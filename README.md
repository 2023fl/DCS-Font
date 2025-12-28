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
## Datasets
You can collect your own fonts from the following web sites (for non-commercial purpose):
- https://www.foundertype.com/index.php/FindFont/index (acknowledgement: DG-Font refers this web site)
- https://chinesefontdesign.com/
- Any other web sites providing non-commercial fonts
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
 CUDA_VISIBLE_DEVICES=GPUID python sample.py
```
- ckpt_dir: The model checkpoints saving directory.
- content_image_path: The content/source image path.
- style_image_path: The style/reference image path.
- save_image: set True if saving as images.
- save_image_dir: The image saving directory, the saving files including an out_single.png and an out_with_cs.png.
- device: The sampling device, recommended GPU acceleration.
- guidance_scale: The classifier-free sampling guidance scale.
- num_inference_steps: The inference step by DPM-Solver++.

- character_input: If set True, use character string as content/source input.
- content_character: The content/source content character string.
### Important Notice
This code is directly related to a manuscript currently submitted to *The Visual Computer*.
## Citation
```bash
If you find this code useful for your research, please consider citing the following manuscript:
Y. Fang, et al., “DCS-Font: Dynamic Feature Enhancement for Few-Shot Font Generation” 
```
currently submitted to *The Visual Computer*.
