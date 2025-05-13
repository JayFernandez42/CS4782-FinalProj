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

## Chosen Result
-----------------------------------------------------------------------------------------------------

The original paper benchmarked various generative models, including retrieval-based methods, maximum entropy pipelines, and a gated recurrent neural network (GRNN) trained to ask questions. Their evaluation demonstrated that GRNN models most effectively captured human-like question patterns, particularly on their event-centric Bing and Flickr datasets.

We aimed to replicate this result: specifically, the ability of an end-to-end generative model (in our case, a Transformer-based decoder) to generate abstract, context-aware questions about an image. Our central goal was to replicate their qualitative insight: "models can generate plausible questions, but there's still a gap to human naturalness."

## GITHUB CONTENT
-----------------------------------------------------------------------------------------------------
├── **model.py** - Encapsulate the transformer architecture (with use_resnet toggles and token embedding)             
├── **train.py** - Contains the `train_model()` loop used across Transformer and GRNN architectures             
├── **eval.py** - Includes both `test_loss()` for evaluating test performance and `generate_question()` for generating text from a model and image              
├── **dataset.py** - Customized `VQGTensorDataset` class that can work with raw .jpg images or .pt tensor files           
├── **utils.py** - Provides support functions for vocabulary construction, token indexing, and preprocessing; includes `build_vocab()` used across training and evaluation             
├── **config.py** - Stores centralized hyperparameters and device setup for reproducibility and clean tuning             
├── **prepare_gqa_for_vqg.py** - Extracts image-question pairs from the GQA dataset for use in VQG-style fine-tuning  

## RE-IMPLEMENTATION
-----------------------------------------------------------------------------------------------------
We used a simple Encoder-Decoder Model where input our image through a CNN and passed the resulting embedding through a series of GRU blocks. Our dataset which we used from the paper consisted of MS COCO, Flickr, and Bing which was a total of 15,000 iamges. However, a major challenge that we had was a lot of the images such as MS COCO were corrupted or the links no longer worked so we had to manually remove them. The evaluation metric we used was BLEU which is an n-gram overlap metric that measures how much our model generated question overlaps with the assigned human generated quesiton; however, we also noticed that it is generally hard to get a high score for this and another challenge is truly being able to compare human and machine gnerated questions.



## PROJECT SETUP INSTRUCTIONS
-----------------------------------------------------------------------------------------------------

1) Clone the repository and place all images and dataset CSVs into the `data/` folder

2) Install dependencies listed in `requirements.txt` (pip install -r requirements.txt)

3) Open and run `ImageToQuestion.ipynb` for the full training and evaluation pipeline




## SAMPLE INFERENCE
-----------------------------------------------------------------------------------------------------

Once trained, a model can generate a natural-language question from a single image using `generate_question()`
in eval.py. Image inputs are accepted as raw .jpg files.

| Input 1 | Input 2 |
|--------|---------|
| <img width="791" alt="Screenshot 2025-05-12 at 6 07 54 PM" src="https://github.com/user-attachments/assets/8d49ea0b-1db6-4868-8edd-3790c2c50084" /> | <img width="785" alt="Screenshot 2025-05-12 at 6 09 37 PM" src="https://github.com/user-attachments/assets/e52c7bce-151b-4c70-9516-cb09fa821bf5" /> |

## RESULTS/INSIGHTS
-----------------------------------------------------------------------------------------------------
Once again, we could not fully recreate the results under the conditions outlined in the paper due to data limitations. However, aside from this face our model outperformed or performed just as well as the original paper in terms of metrics such as BLEU. We find that in general our questions that we produce have around a 10-15% overlap with the human generated questions, which is a sign of a good result. These results are comparable with those of the paper. However, an insight we gained from this is that it is hard to truly generate the complexity or uniqueness of a human question.

## CONCLUSION
-----------------------------------------------------------------------------------------------------
Our implementation successfully reproduces and extends the original paper's findings using modern deep learning techniques. While transformer-based architectures and pretrained vision encoders offer improvements in question quality and diversity, the core challenge identified in the original paper remains: generating truly human-like questions is difficult. Our results suggest that future work should focus on larger-scale pretraining, more sophisticated evaluation metrics, and integration with conversational systems.


## REFERENCES
-----------------------------------------------------------------------------------------------------

Mostafazadeh, N., et al. Generating Natural Questions about an Image. arXiv, 2016. [https://arxiv.org/abs/1603.06059](url)

Hudson, D. A., and Manning, C. D. GQA: A New Dataset for Real-World Visual Reasoning. CVPR, 2019. [https://cs.stanford.edu/people/dorarad/gqa/](url)

He, K., et al. Deep Residual Learning for Image Recognition. CVPR, 2016. [https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py](url)

Radford, A., et al. Learning Transferable Visual Models From Natural Language Supervision. arXiv, 2021. [https://github.com/openai/CLIP](url)

Sennrich, R., Haddow, B., and Birch, A. Improving Neural Machine Translation Models with Monolingual Data. arXiv, 2015. [https://arxiv.org/abs/1511.06709](url)

## ACKNOWLEDGEMENTS
-----------------------------------------------------------------------------------------------------

This project was a part of a final project for a course at Cornell University called CS 4782/5782: Intro to Deep Learning, hosted by Killian Q. Weinberger and Jennifer J. Sun. We recognize their ability to allocate google cloud compute for the course, and thank all institutions accordingly.

