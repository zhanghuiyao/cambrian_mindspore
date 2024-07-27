<div align="center">

#  🪼 *Cambrian-1*:<br> A Fully Open, Vision-Centric Exploration of Multimodal LLMs

</div>


> *Here is [Cambrian-1](https://arxiv.org/abs/2406.16860) implemented with [MindSpore](https://www.mindspore.cn/), reference to [Official Implementation](https://github.com/cambrian-mllm/cambrian) by New York University.*

<div align="center">
<p>
    <img src="images/cambrian.png" alt="Cambrian" width="500" height="auto">
</p>
</div>

> *Fun fact: vision emerged in animals during the Cambrian period! This was the inspiration for the name of our project, Cambrian.*

<br>


## Release

coming soon !!!


## Features

- [ ] Cambrian Model
  - [x] Infer
  - [ ] Pretrain - Training Spatial Vision Aggregator
  - [ ] Finetune - Instruction Tuning
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
- [ ] Targeted Data Engine
- [ ] Benchmarking
- [ ] Evaluation (Not yet release)


## Contents
- [Quick Start]()
- [Cambrian Weights](#cambrian-weights)
    - [Weight Convert]()
- [Cambrian Instruction Tuning Data](#cambrian-instruction-tuning-data)
- [Train](#train)
- [Evaluation](#evaluation)
- [Demo](#demo)

