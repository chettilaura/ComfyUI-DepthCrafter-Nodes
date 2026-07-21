import os
import torch
import math
import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

from .depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from .depthcrafter.depth_crafter_ppl import DepthCrafterPipeline

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
        print(f"\n[DepthCrafter DEBUG] --- Rez Environment Initialization ---")
        
        # Ottieni i percorsi dalla variabile d'ambiente Rez
        rez_env_val = os.environ.get("COMFY_MODEL_TYPE_DEPTHCRAFTER_MODELS", "")
        
        search_dirs = []
        if rez_env_val:
            search_dirs = rez_env_val.split(os.pathsep)
            print(f"[DepthCrafter DEBUG] Rez variable found with {len(search_dirs)} search paths.")
        
        # Aggiungi percorsi extra da extra_model_paths.yaml se presenti
        extra_paths = folder_paths.get_folder_paths("depthcrafter_models")
        if extra_paths:
            search_dirs.extend(extra_paths)

        # Cerchiamo cartelle che contengano "model_index.json" (pipeline) o "config.json" (unet)
        found_bundles = []
        for root_dir in set(search_dirs):
            if not root_dir or not os.path.exists(root_dir):
                continue
            
            # Scansioniamo le sottocartelle per trovare i "bundle" di modelli
            for item in os.listdir(root_dir):
                full_path = os.path.join(root_dir, item)
                if os.path.isdir(full_path):
                    # Identifichiamo se è una cartella utile
                    if os.path.exists(os.path.join(full_path, "model_index.json")):
                        found_bundles.append(f"Pipeline: {item}")
                    elif os.path.exists(os.path.join(full_path, "config.json")):
                        found_bundles.append(f"UNet: {item}")

        if not found_bundles:
            found_bundles = ["No models found - Check Rez folders"]

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

        # --- LOGICA DI RISOLUZIONE PATH TRAMITE ENV VAR ---
        rez_env_val = os.environ.get("COMFY_MODEL_TYPE_DEPTHCRAFTER_MODELS", "")
        search_dirs = rez_env_val.split(os.pathsep) if rez_env_val else []
        search_dirs.extend(folder_paths.get_folder_paths("depthcrafter_models"))
        
        unet_path = None
        pretrain_path = None

        # Cerchiamo le due cartelle necessarie nei percorsi Rez
        for root_dir in set([os.path.abspath(p) for p in search_dirs if p.strip()]):
            # Cerchiamo la UNet (solitamente cartella tencent_DepthCrafter o simile)
            p_unet = os.path.join(root_dir, "tencent_DepthCrafter")
            # Cerchiamo la pipeline SVD (solitamente stabilityai_stable-video-diffusion...)
            p_svd = os.path.join(root_dir, "stabilityai_stable-video-diffusion-img2vid-xt")

            if os.path.exists(p_unet):
                unet_path = p_unet
            if os.path.exists(p_svd):
                pretrain_path = p_svd

        # Fallback al comportamento standard se non trovati tramite Rez
        if not unet_path or not pretrain_path:
            model_dir = os.path.join(folder_paths.models_dir, "depthcrafter")
            if not unet_path: unet_path = os.path.join(model_dir, "tencent_DepthCrafter")
            if not pretrain_path: pretrain_path = os.path.join(model_dir, "stabilityai_stable-video-diffusion-img2vid-xt")

        print(f"[DepthCrafter DEBUG] UNet Path: {unet_path}")
        print(f"[DepthCrafter DEBUG] Pretrain Path: {pretrain_path}")

        if not os.path.exists(unet_path) or not os.path.exists(pretrain_path):
            raise Exception(
                f"DepthCrafter models not found.\n"
                f"Checked Rez paths: {search_dirs}\n"
                f"Missing either:\n1. {unet_path}\n2. {pretrain_path}"
            )

        # Caricamento effettivo
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )


        # Aggiunto
        mm.unload_all_models()
        torch.cuda.empty_cache()
        
        pipe = DepthCrafterPipeline.from_pretrained(
            pretrain_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_local_files_only=True,
            low_cpu_mem_usage=True,
        )

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



# class DownloadAndLoadDepthCrafterModel(DepthCrafterNode):
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#             "enable_model_cpu_offload": ("BOOLEAN", {"default": True}),
#             "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False}),
#         }}

#     RETURN_TYPES = ("DEPTHCRAFTER_MODEL",)
#     RETURN_NAMES = ("depthcrafter_model",)
#     FUNCTION = "load_model"
#     DESCRIPTION = """
#     Downloads and loads the DepthCrafter model.
#     - enable_model_cpu_offload: If True, the model will be offloaded to the CPU. (Saves VRAM)
#     - enable_sequential_cpu_offload: If True, the model will be offloaded to the CPU in a sequential manner. (Saves the most VRAM but runs slowly)
#     Only enable one of the two at a time.
#     """

#     def load_model(self, enable_model_cpu_offload, enable_sequential_cpu_offload):
#         device = mm.get_torch_device()

#         model_dir = os.path.join(folder_paths.models_dir, "depthcrafter")
#         os.makedirs(model_dir, exist_ok=True)

#         # Paths to models
#         unet_path = os.path.join(model_dir, "tencent_DepthCrafter")
#         pretrain_path = os.path.join(model_dir, "stabilityai_stable-video-diffusion-img2vid-xt")

#         depthcrafter_files_to_download = [
#             "config.json",
#             "diffusion_pytorch_model.safetensors",
#         ]
#         svd_files_to_download = [
#             "feature_extractor/preprocessor_config.json",
#             "image_encoder/config.json",
#             "image_encoder/model.fp16.safetensors",
#             "scheduler/scheduler_config.json",
#             "unet/config.json",
#             "unet/diffusion_pytorch_model.fp16.safetensors",
#             "vae/config.json",
#             "vae/diffusion_pytorch_model.fp16.safetensors",
#             "model_index.json",
#         ]

#         self.start_progress(len(svd_files_to_download) + len(depthcrafter_files_to_download))

#         # Check if models exist, if not download them
#         # from huggingface_hub import hf_hub_download

#         # if not os.path.exists(unet_path):
#         #     print(f"Downloading UNet model to: {unet_path}")
#         #     for path in depthcrafter_files_to_download:
#         #         hf_hub_download(
#         #             repo_id="tencent/DepthCrafter",
#         #             filename=path,
#         #             local_dir=unet_path,
#         #             local_dir_use_symlinks=False,
#         #             revision="c1a22b53f8abf80cd0b025adf29e637773229eca",
#         #         )
#         #         self.update_progress()

#         # if not os.path.exists(pretrain_path):
#         #     print(f"Downloading pre-trained pipeline to: {pretrain_path}")
#         #     for path in svd_files_to_download:
#         #         hf_hub_download(
#         #             repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
#         #             filename=path,
#         #             local_dir=pretrain_path,
#         #             local_dir_use_symlinks=False,
#         #             revision="9e43909513c6714f1bc78bcb44d96e733cd242aa",
#         #         )
#         #         self.update_progress()

#         #PHOTON
#         # Check if models exist. If not, raise an error instead of downloading.
#         if not os.path.exists(unet_path) or not os.path.exists(pretrain_path):
#             raise Exception(
#                 f"Model files not found in {model_dir}. "
#                 "Automatic downloading has been disabled. Please manually place the models in the folder."
#             )

#         # Load the custom UNet model
#         unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
#             unet_path,
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#         )

#         # Load the pipeline
#         pipe = DepthCrafterPipeline.from_pretrained(
#             pretrain_path,
#             unet=unet,
#             torch_dtype=torch.float16,
#             variant="fp16",
#             use_local_files_only=True,
#             low_cpu_mem_usage=True,
#         )

#         # Model setup
#         # try:
#         #     pipe.enable_xformers_memory_efficient_attention()
#         # except Exception as e:
#         #     print(e)
#         #     print("Xformers is not enabled")
#         pipe.enable_attention_slicing()
        
#         if enable_model_cpu_offload:
#             pipe.enable_model_cpu_offload()
#         elif enable_sequential_cpu_offload:
#             pipe.enable_sequential_cpu_offload()
#         else:
#             pipe.to(device)


#         depthcrafter_model = {
#             "pipe": pipe,
#             "device": device,
#         }

#         self.end_progress()

#         return (depthcrafter_model,)

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
    DESCRIPTION = """
    Runs the DepthCrafter model on the input images.
    **WARNING:** The model internally requires image dimensions (width and height)
    to be multiples of 64. Enable 'force_size' to automatically resize the input
    to the nearest valid dimensions, or ensure your input images already meet
    this requirement if 'force_size' is disabled.
    """
    
    def process(self, depthcrafter_model, images, force_size, num_inference_steps, guidance_scale, window_size, overlap):
        device = depthcrafter_model['device']
        pipe = depthcrafter_model['pipe']
        
        B, H, W, C = images.shape
        
        # --- FIX 1: GESTIONE OVERLAP PER SEQUENZE CORTE ---
        # Se i frame totali (B) sono meno dell'overlap, il calcolo fallisce.
        if B <= overlap:
            print(f"DepthCrafter: Riduzione automatica overlap da {overlap} a {B-1} per sequenza corta.")
            overlap = max(0, B - 1)
        
        # Se window_size è più grande dei frame disponibili, forziamolo a B
        if window_size > B:
            window_size = B

        if force_size:
            width = round(W / 64) * 64
            height = round(H / 64) * 64
            width = max(64, width)
            height = max(64, height)

            if width != W or height != H:
                print(f"DepthCrafter: Resizing input from {W}x{H} to {width}x{height}")
                images_for_resize = images.permute(0, 3, 1, 2)
                images_resized = torch.nn.functional.interpolate(
                    images_for_resize,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                )
                images = images_resized.permute(0, 2, 3, 1)
                H, W = height, width

        # --- FIX 2: SICUREZZA DIMENSIONI TENSOR ---
        # Prendiamo le dimensioni direttamente dal tensor finale per evitare errori di arrotondamento
        final_H = images.shape[1]
        final_W = images.shape[2]

        # Permute e cast
        images = images.permute(0, 3, 1, 2)  # [B, C, H, W]
        images = images.to(device=device, dtype=torch.float16)
        images = torch.clamp(images, 0, 1)
        
        # Calcolo finestre (aggiunto controllo per evitare divisione per zero)
        stride = window_size - overlap
        if stride <= 0: stride = 1
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
                decode_chunk_size=1, # <--- AGGIUNGI QUESTA RIGA QUI
                track_time=False,
                progress_callback=self.update_progress,
            )
            
        res = result.frames[0]  # [B, H, W, C]
        
        # Convert to grayscale depth map
        res = res.sum(dim=1) / res.shape[1]  # [B, H, W]
        
        # Normalize depth maps
        res_min = res.min()
        res_max = res.max()
        res = (res - res_min) / (res_max - res_min + 1e-8)
        
        # Convert back to tensor with 3 channels
        depth_maps = res.unsqueeze(-1).repeat(1, 1, 1, 3)  # [B, H, W, 3]
        depth_maps = depth_maps.float()
        
        self.end_progress()
        
        return (depth_maps,)
