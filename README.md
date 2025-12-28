# DCS-Font
We propose DCS-Font for few-shot font generation—a diffusion-based framework featuring a Content-Style Feature Enhancement Encoder (CSFE-Encoder) and Multi-Scale Adaptive U-Net Modulation (MAUM) module to ensure content accuracy and style consistency.

## Installation
### Prerequisites (Recommended)

- OS: Linux  
- Python: 3.9  
- PyTorch: 1.13.1  
- CUDA: 11.7  

---

### Environment Setup

#### Clone this repository

```bash
git clone https://github.com/2023fl/DCS-Font.git
cd DCS-Font
```
Step 1: Create a conda environment and activate it.
```bash
conda create -n DCS-Font python=3.9 -y
conda activate DCS-Font
```
Step 2: Install related version Pytorch.
```bash
# Example. Please install the corresponding version of Pytorch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
Step 3: Install the required packages.
```bash
pip install -r requirements.txt
```
## Training
### Data Construction
The training data files tree should be (The data examples are shown in directory data_examples/train/):
```bash
├──data_examples
│   └── train
│       ├── ContentImage
│       │   ├── char0.png
│       │   ├── char1.png
│       │   ├── char2.png
│       │   └── ...
│       └── TargetImage.png
│           ├── style0
│           │     ├──style0+char0.png
│           │     ├──style0+char1.png
│           │     └── ...
│           ├── style1
│           │     ├──style1+char0.png
│           │     ├──style1+char1.png
│           │     └── ...

```
## Training
```bash
sh scripts/train_phase_1.sh
```
- data_root: The data root, as ./data_examples
- output_dir: The training output logs and checkpoints saving directory.
- resolution: The resolution of the UNet in our diffusion model.
- style_image_size: The resolution of the style image, can be different with resolution.
- content_image_size: The resolution of the content image, should be the same as the resolution.
- channel_attn: Whether to use the channel attention in the MCA block.
- train_batch_size: The batch size in the training.
- max_train_steps: The maximum of the training steps.
- learning_rate: The learning rate when training.
- ckpt_interval: The checkpoint saving interval when training.
- drop_prob: The classifier-free guidance training probability.

## Sampling
Put your re-training checkpoint folder ckpt to the root directory, including the files unet.pth, content_encoder.pth, and style_encoder.pth.
```bash
 CUDA_VISIBLE_DEVICES=GPUID python sample_fyy.py
```
