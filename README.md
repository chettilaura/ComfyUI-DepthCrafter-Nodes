# DepthCrafter Nodes

**Create consistent depth maps for your videos using DepthCrafter in ComfyUI.**

Original DepthCrafter repo: https://github.com/Tencent/DepthCrafter

DepthCrafter model download available [here](https://huggingface.co/tencent/DepthCrafter/tree/main)
(Model license is limited to non-commercial academic use only)

Recommended minimum VRAM: 8GB

## ⚠️ License notice

This is a fork of [akatz-ai/ComfyUI-DepthCrafter-Nodes](https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes). The upstream `LICENSE` explicitly restricts the DepthCrafter inference code (not just the model weights) to **academic, research, and education purposes only, and prohibits commercial or production use under any circumstances**. This fork is shared publicly purely to document technical changes for portfolio/educational purposes — it carries the same restriction and is not offered for commercial use.

## Fork notes

**What changed, and why:** same theme as the rest of this restructuring work — moving from an auto-downloading node to one that assumes local models and manages VRAM explicitly:
- **Local-only model resolution.** Replaced `huggingface_hub.hf_hub_download(...)` (which pulled ~10 files from two HF repos on first run) with a `folder_paths`-based lookup across registered model directories, so the node works offline once the model bundle is placed on disk.
- **Explicit VRAM clearing before loading**, via `comfy.model_management.unload_all_models()` + `torch.cuda.empty_cache()`.
- **Manual VAE reload workaround** (`depthcrafter/depth_crafter_ppl.py`): in some environments the VAE loaded via `from_pretrained` would decode too many frames at once and exceed the VRAM budget; this fork manually reloads the VAE weights and forces `decode_chunk_size=1`, with a fallback re-injection path in `__call__` if the VAE ends up missing at inference time.
- **Overlap-blend safety clamp**: the sliding-window latent blending now clamps the overlap to what's actually available on the last window, instead of assuming a fixed overlap size (avoids a shape-mismatch crash on short final windows).

See [`dev-history/`](dev-history/) for the original upstream pipeline file and an intermediate refactor step, kept for reference.

## Updates:
(04/16/2025): Disabled forced xFormers in model loading stage to prevent issues with higher dimension inputs (e.g. 960x960) from hitting batch size cap.

(04/04/2025): Replaced **max_size** with **force_rate** parameter to automatically attempt re-sizing input resolution to match closest multiple of 64.
- Updated requirements.txt with accelerate
- Added warning message in node description about 64 pixel resolution multiple constraint.

(11/27/2024): Updated to support DepthCrafter v1.0.1 inference configuration.

(10/25/2024): Added enable_model_cpu_offload and enable_sequential_cpu_offload options to model loader. **Only enable one at a time!**
- **enable_model_cpu_offload**: Can save +25% of VRAM with little impact to speed by offloading models to cpu when no longer needed for inference.
- **enable_sequential_cpu_offload**: Can save +37% of VRAM at the expense of slower inference speed by moving all models to CPU.

## 🖥️ Custom Environment
I created a custom ComfyUI environment for testing out DepthCrafter nodes:

**akatzai/comfy-env-depthcrafter:latest**

Create a new environment and copy and paste the link above into the "Custom Image" field in my Environment Manager tool:
https://github.com/akatz-ai/ComfyUI-Environment-Manager

Make sure to select the **Basic** environment type to access the included workflow!

## ⭐ Example Workflow:
![depthcrafter](https://github.com/user-attachments/assets/d7e50363-b489-4c01-8e52-c7f654cdd37a)



## 📦 Included Nodes:
- **DownloadAndLoadDepthCrafterModel**: Will fetch the model files need to run DepthCrafter and save them under models/depthcrafter.
- **DepthCrafter**: Renders out depthmap videos given the following inputs:
  - **depthcrafter_model**: (input from the first node)
  - **images**: (single or batch),
  - **max_res**: the maximum resolution of the input images, supports increments of 64 pixels. (Larger resolutions require more VRAM)
  - **max_inference_steps**: more steps may result in less artifacts in the output, but will take longer to render.
  - **guidance_scale**: (1 - 1.2 recommended)
  - **window_size**: the length of the context window for DepthCrafter. You can lower this to save on VRAM at the expense of taking longer to render (75-110 recommended)
  - **overlap**: how much to overlap each context window to render longer videos > 110 frames. (25 recommended)

## 🔧 Installation and Usage

1. ComfyUI Manager:

- This node pack is available to install via the [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager). You can find it in the Custom Nodes section by searching for "DepthCrafter" and clicking on the entry called "DepthCrafter Nodes".

2. Clone the repository:
- Navigate to ComfyUI/custom_nodes folder in terminal or command prompt.
- Clone the repo using the following command:
```bash
git clone https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes.git
```
- Restart ComfyUI

## Manual Model Installation

If you have trouble using the automatic download feature of the "DownloadAndLoadDepthCrafterModel" node, you can manually download the necessary files like so:
1. Create the model directories for Depthcrafter:
   - In your models/ folder you should create a new directory called depthcrafter/
   - Inside of models/depthcrafter/ you should create 2 additional directories called tencent_DepthCrafter/ and stabilityai_stable-video-diffusion-img2vid-xt/

Result:

    models/
    
      depthcrafter/
      
        tencent_DepthCrafter/
        
        stabilityai_stable-video-diffusion-img2vid-xt/

2. Now navigate to https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/tree/main
   - Download the following files and directories and place them inside of model/depthcrafter/stabilityai_stable-video-diffusion-img2vid-xt/:
     ```
      feature_extractor/preprocessor_config.json,
      image_encoder/config.json,
      image_encoder/model.fp16.safetensors,
      scheduler/scheduler_config.json,
      unet/config.json,
      unet/diffusion_pytorch_model.fp16.safetensors,
      vae/config.json,
      vae/diffusion_pytorch_model.fp16.safetensors,
      model_index.json,
     ```
     *** Note: Make sure you actually have/create the subdirectories feature_extractor/, image_encoder/, scheduler/, unet/, and vae/ for the files above! ***

3. Navigate to https://huggingface.co/tencent/DepthCrafter/tree/main:
   - Download the following files and place the inside of model/depthcrafter/tencent_DepthCrafter/:
     ```
     config.json
     diffusion_pytorch_model.safetensors
     ```

After running the node with the above files and directories installed you should be able to run DepthCrafter.

