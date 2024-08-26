## Convert Pretrained Checkpoint

We provide a script for converting pre-trained weight from [`.safetensors`,`.bin`] to `.ckpt` in `./tools/weight_conversion/convert_weight.py`.

### step1. Download the [Cambrian-1](https://huggingface.co/nyu-visionx) and Vision Towers pre-train weights from huggingface.

- `Cambrian-8B`                       : https://huggingface.co/nyu-visionx/cambrian-8b
- `siglip/CLIP-ViT-SO400M-14-384`     : https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384
- `openai/clip-vit-large-patch14-336` : https://huggingface.co/openai/clip-vit-large-patch14-336
- `facebook/dinov2-giant-res378`      : https://huggingface.co/facebook/dinov2-giant
- `clip-convnext-XXL-multi-stage`     : https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup

### step2. Convert weight to MindSpore `.ckpt` format.

```shell
# convert cambrian-8b and 4 vision towers weight
python tools/weight_conversion/convert_weight.py \
  --cambrian_folder /PATH TO/nyu-visionx/cambrian-8b \
  --siglip_folder /PATH TO/timm/ViT-SO400M-14-SigLIP-384 \
  --openai_folder /PATH TO/openai/clip-vit-large-patch14-336 \
  --dinov2_folder /PATH TO/facebook/dinov2-giant \
  --convnext_folder /PATH TO/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup \
  \
  --mindspore_checkpoint_path "cambrian-8b.ckpt" \
```
