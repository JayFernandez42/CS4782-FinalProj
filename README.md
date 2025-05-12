# Visual Question Generation using Transformer Decoders and Image Pretraining

## INTRODUCTION
-----------------------------------------------------------------------------------------------------

The objective of this project is to generate natural, context-aware questions about images—a task known as Visual Question Generation (VQG). 

Inspired by the work in *Generating Natural Questions About an Image* (Mostafazadeh et al., 2016), we explore the construction of a modern VQG pipeline using transformer-based decoders and pretrained vision encoders (ResNet and CLIP-ViT). 

Unlike classification tasks, VQG demands both semantic understanding and language fluency. We extend beyond the original paper by introducing transfer learning methods, dataset augmentation, and structured question generation workflows using PyTorch.

**This project covers key machine learning components:**

A) Building image-question datasets using GQA, VQG-Bing, COCO, and Flickr

B) Training transformer models on raw images or ResNet/CLIP-encoded tensors

C) Using pretraining from VQA-style datasets to improve fluency and syntax

D) Performing inference from arbitrary images using a trained question generator


## PROJECT SETUP INSTRUCTIONS
-----------------------------------------------------------------------------------------------------

1) Clone the repository and place all images and dataset CSVs into the `data/` folder

2) Install dependencies listed in `requirements.txt`

3) Open and run `ImageToQuestion.ipynb` for the full training and evaluation pipeline

**Directory structure:**


## REFERENCES
-----------------------------------------------------------------------------------------------------

Mostafazadeh, N., et al. Generating Natural Questions about an Image. arXiv, 2016. [https://arxiv.org/abs/1603.06059](url)

Hudson, D. A., and Manning, C. D. GQA: A New Dataset for Real-World Visual Reasoning. CVPR, 2019. [https://cs.stanford.edu/people/dorarad/gqa/](url)

He, K., et al. Deep Residual Learning for Image Recognition. CVPR, 2016. [https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py](url)

Radford, A., et al. Learning Transferable Visual Models From Natural Language Supervision. arXiv, 2021. [https://github.com/openai/CLIP](url)

Sennrich, R., Haddow, B., and Birch, A. Improving Neural Machine Translation Models with Monolingual Data. arXiv, 2015. [https://arxiv.org/abs/1511.06709](url)
