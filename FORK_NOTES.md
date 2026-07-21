# Fork notes

This is a fork of [akatz-ai/ComfyUI-DepthCrafter-Nodes](https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes).

## ⚠️ License notice

The upstream `LICENSE` explicitly restricts the DepthCrafter inference code (not just the model weights) to **academic, research, and education purposes only, and prohibits commercial or production use under any circumstances**. This fork is shared publicly purely to document technical changes for portfolio/educational purposes — it carries the same restriction and is not offered for commercial use.

## What changed, and why

Same theme as the rest of this restructuring work — moving from an auto-downloading node to one that assumes local models and manages VRAM explicitly:

- **Local-only model resolution.** Replaced `huggingface_hub.hf_hub_download(...)` (which pulled ~10 files from two HF repos on first run) with a `folder_paths`-based lookup across registered model directories, so the node works offline once the model bundle is placed on disk.
- **Explicit VRAM clearing before loading**, via `comfy.model_management.unload_all_models()` + `torch.cuda.empty_cache()`.
- **Manual VAE reload workaround** (`depthcrafter/depth_crafter_ppl.py`): in some environments the VAE loaded via `from_pretrained` would decode too many frames at once and exceed the VRAM budget; this fork manually reloads the VAE weights and forces `decode_chunk_size=1`, with a fallback re-injection path in `__call__` if the VAE ends up missing at inference time.
- **Overlap-blend safety clamp**: the sliding-window latent blending now clamps the overlap to what's actually available on the last window, instead of assuming a fixed overlap size (avoids a shape-mismatch crash on short final windows).

See [`dev-history/`](dev-history/) for the original upstream pipeline file and an intermediate refactor step, kept for reference.

## Credits

- Original ComfyUI wrapper: [akatz-ai/ComfyUI-DepthCrafter-Nodes](https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes)
- Underlying model: [Tencent/DepthCrafter](https://github.com/Tencent/DepthCrafter)
