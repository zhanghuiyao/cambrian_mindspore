<div align="center">

#  🪼 *Cambrian-1 on MindSpore*

</div>


> *Here is [Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs](https://arxiv.org/abs/2406.16860) implemented with [MindSpore](https://www.mindspore.cn/), reference to [Official Implementation](https://github.com/cambrian-mllm/cambrian) by New York University.*

<div align="center">
<p>
    <img src="images/cambrian.png" alt="Cambrian" width="500" height="auto">
</p>
</div>

> *Fun fact: vision emerged in animals during the Cambrian period! This was the inspiration for the name of our project, Cambrian.*

<br>


## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
    - [Preprocess pretrain weight](#preprocess-pretrain-weight)
    - [Inference](#inference)
    - [Training](#training)
- [Features and TodoList](#features-and-todolist)


## Installation

1. Clone this repository and navigate to into the codebase

```bash
git clone https://github.com/zhanghuiyao/cambrian_mindspore.git
cd cambrian_mindspore
export PYTHONPATH=$PWD:$PYTHONPATH
```

2. Install [MindSpore](https://www.mindspore.cn/install/) and CANN

3. Install Others Packages

```bash
pip install -r requirements.txt
```

## Quick Start

### Preprocess pretrain weight

1. Download

- `cambrian` pretrain weight from https://huggingface.co/nyu-visionx/cambrian-8b
- `siglip/CLIP-ViT-SO400M-14-384` pretrain weight from https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384
- `openai/clip-vit-large-patch14-336` pretrain weight from https://huggingface.co/openai/clip-vit-large-patch14-336
- `facebook/dinov2-giant-res378` pretrain weight from https://huggingface.co/facebook/dinov2-giant
- `clip-convnext-XXL-multi-stage` pretrain weight from https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup

2. Convert weight to MindSpore format

```shell
python tools/weight_conversion/convert_weight.py \
  --cambrian_folder "path to cambrian folder" \
  --siglip_folder "path to siglip/CLIP-ViT-SO400M-14-384 folder" \
  --openai_folder "path to openai/clip-vit-large-patch14-336 folder" \
  --dinov2_folder "path to facebook/dinov2-giant-res378 folder" \
  --convnext_folder "path to clip-convnext-XXL-multi-stage folder" \
  --mindspore_checkpoint_path "cambrian-8b.ckpt"
```

### Inference

```bash
python inference.py --checkpoint_path ./cambrian-8b.ckpt
```

### Training

```bash
# Replace your dataset path in `*.sh`, default is two-images toy-dataset

# pretrain
bash scripts/cambrian/pretrain_cambrian_8b.sh

# finetune
bash scripts/cambrian/finetune_cambrian_8b.sh
```


## Features and TodoList

- [x] Cambrian Framework
  - [x] Infer
  - [x] Pretrain - Training Spatial Vision Aggregator
  - [x] Finetune - Instruction Tuning
- [x] Third party library adaptation
  - [x] `transformers` -> `cambrian/transformers`
  - [x] `open_clip` -> `cambrian/open_clip`
  - [x] `timm` -> `cambrian/timm`
- [x] Base Modules
  - [x] Connector
    - [x] SVA, Spatial Vision Aggregator
    - [x] Projector (LLaVA style 2-layer MLP)
  - [x] Vision Encoders
    - [x] OpenAI CLIP ViT-L/14@336
    - [x] SigLIP ViT-SO400M/14@384
    - [x] OpenCLIP ConvNeXt-XXL@1024
    - [x] DINOv2 ViT-L/14@518
  - [ ] LLMs
    - [ ] Phi-3B
    - [x] LLaMA-3-Instruct-8B
    - [ ] Vicuna-1.5-13B
    - [ ] Hermes-2-Yi-34B
- [ ] (TODO) support KV-Cache
- [ ] (TODO) Benchmarking: CV-Bench
- [ ] (TODO) Targeted Data Engine
- [ ] (Not yet release) Evaluation 

