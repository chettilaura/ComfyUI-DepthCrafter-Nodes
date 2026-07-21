import os
import torch
import math
import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

from .depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from .depthcrafter.depth_crafter_ppl import DepthCrafterPipeline

# --- REGISTER FOLDER PATHS ---
# Register the "depthcrafter" category so ComfyUI recognizes it
if "depthcrafter" not in folder_paths.folder_names_and_paths:
    # Default lookup in ComfyUI/models/depthcrafter
    base_path = os.path.join(folder_paths.models_dir, "depthcrafter")
    folder_paths.add_model_folder_path("depthcrafter", base_path)

class DepthCrafterNode:
    def __init__(self):
        self.progress_bar = None

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)

    def update_progress(self, *args, **kwargs):
        if self.progress_bar:
            self.progress_bar.update(1)

    def end_progress(self):
        self.progress_bar = None
        
    CATEGORY = "DepthCrafter"

class DownloadAndLoadDepthCrafterModel(DepthCrafterNode):
    @classmethod
    def INPUT_TYPES(s):
        # Get all valid subfolders under the paths registered for "depthcrafter"
        # This populates the dropdown if the user wants to choose between versions
        model_dirs = folder_paths.get_filename_list("depthcrafter")
        
        return {
            "required": {
                "enable_model_cpu_offload": ("BOOLEAN", {"default": True}),
                "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("DEPTHCRAFTER_MODEL",)
    RETURN_NAMES = ("depthcrafter_model",)
    FUNCTION = "load_model"

    def load_model(self, enable_model_cpu_offload, enable_sequential_cpu_offload):
        device = mm.get_torch_device()

        # --- RESOLVE PATH VIA FOLDER_PATHS ---
        # Get the list of base paths registered for this category
        search_dirs = folder_paths.get_folder_paths("depthcrafter")

        unet_path = None
        pretrain_path = None

        # Iterate over configured paths (including those from extra_model_paths.yaml)
        for root_dir in search_dirs:
            if not os.path.exists(root_dir):
                continue
            
            p_unet = os.path.join(root_dir, "tencent_DepthCrafter")
            p_svd = os.path.join(root_dir, "stabilityai_stable-video-diffusion-img2vid-xt")

            if os.path.exists(p_unet) and os.path.exists(p_svd):
                unet_path = p_unet
                pretrain_path = p_svd
                break # Found the complete bundle

        if not unet_path or not pretrain_path:
            raise Exception(
                f"DepthCrafter models not found in ComfyUI model folders.\n"
                f"Please ensure you have:\n"
                f"1. {os.path.join('models', 'depthcrafter', 'tencent_DepthCrafter')}\n"
                f"2. {os.path.join('models', 'depthcrafter', 'stabilityai_stable-video-diffusion-img2vid-xt')}"
            )

        print(f"[DepthCrafter] Loading UNet from: {unet_path}")
        print(f"[DepthCrafter] Loading Pipeline from: {pretrain_path}")

        # Unload existing models to free VRAM
        mm.unload_all_models()
        torch.cuda.empty_cache()

        # Load UNet
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Load pipeline
        pipe = DepthCrafterPipeline.from_pretrained(
            pretrain_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_local_files_only=True,
            low_cpu_mem_usage=True,
        )

        pipe.enable_attention_slicing()
        
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        elif enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)

        return ({"pipe": pipe, "device": device},)

class DepthCrafter(DepthCrafterNode):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "depthcrafter_model": ("DEPTHCRAFTER_MODEL", ),
            "images": ("IMAGE", ),
            "force_size": ("BOOLEAN", {"default": True}),
            "num_inference_steps": ("INT", {"default": 5, "min": 1, "max": 100}),
            "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            "window_size": ("INT", {"default": 110, "min": 1, "max": 200}),
            "overlap": ("INT", {"default": 25, "min": 0, "max": 100}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_maps",)
    FUNCTION = "process"
    
    def process(self, depthcrafter_model, images, force_size, num_inference_steps, guidance_scale, window_size, overlap):
        device = depthcrafter_model['device']
        pipe = depthcrafter_model['pipe']
        
        B, H, W, C = images.shape
        
        if B <= overlap:
            overlap = max(0, B - 1)
        if window_size > B:
            window_size = B

        if force_size:
            width = max(64, round(W / 64) * 64)
            height = max(64, round(H / 64) * 64)

            if width != W or height != H:
                images_for_resize = images.permute(0, 3, 1, 2)
                images_resized = torch.nn.functional.interpolate(
                    images_for_resize, size=(height, width), mode='bilinear', align_corners=False
                )
                images = images_resized.permute(0, 2, 3, 1)
                H, W = height, width

        final_H, final_W = images.shape[1], images.shape[2]
        images = images.permute(0, 3, 1, 2).to(device=device, dtype=torch.float16)
        images = torch.clamp(images, 0, 1)
        
        stride = max(1, window_size - overlap)
        num_windows = math.ceil((B - window_size) / stride) + 1
        
        self.start_progress(num_inference_steps * num_windows)
        
        with torch.inference_mode():
            result = pipe(
                images,
                height=final_H,
                width=final_W,
                output_type="pt",
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                window_size=window_size,
                overlap=overlap,
                decode_chunk_size=1,
                track_time=False,
                progress_callback=self.update_progress,
            )
            
        res = result.frames[0].sum(dim=1) / result.frames[0].shape[1]
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        depth_maps = res.unsqueeze(-1).repeat(1, 1, 1, 3).float()
        
        self.end_progress()
        return (depth_maps,)