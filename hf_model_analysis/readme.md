# HF Model Analysis

- **DISCLAIMER**: This code was written and tested in December 2024, the code might break if HF changes their website or
  api
- as part of our paper we analyze and compare publicly available models
- we follow the following workflow
    - given we base model by its hugging face ID, we:

    - (1) crawl derived models (models are marked as fine-tuned models from the specified base model)
    - (2) compare the crawled models with their base model to generate _comparison files_
    - (3) generate _similarity reports_

- for steps (1) and (2) we use the [comparison_experiments.py](comparison_experiments.py) script
- for step (3) we use the [similarity_analysis.py](similarity_analysis.py) script
- because of their size we do not add the comparison files or the similarity reports to our repo
  - comparison files can be downloaded here: https://nextcloud.hpi.de/f/22515579
  - similarity reports can be downloaded here: https://nextcloud.hpi.de/f/22515582



## Analyzed tasks
- Object Detection (6 models)
  - detr-resnet-50, conditional-detr-resnet-50
  - table-transformer-detection, table-transformer-structure-recognition (ResNet???)
  - detr-resnet-101
  - deformable_detr_resnet_50
- Image Feature Extraction (7 models)
  - dinov2-base, dinov2-large
  - vit16base, vit16large, vit16huge
  - yolos-small, yolos-tiny
- Image Classification (6 Model)
  - **TODO**: maybe some more models here 
  - mobilenet_v2_1.0_224
  - google-efficientnet-b0
  - microsoft/resnet-18, microsoft/resnet-50, microsoft/resnet-101, microsoft/resnet-152
- (Image Segmentation) (5 Models)
  - all nvidia segformer architectures
- (Image to text models)
  - **TODO**: maybe some more models here

## Overview of results 
- **google-mobilenet_v2_1.0_224**
    - task type: Image classification
    - 33 models in total (33/38 worked)
    - 1 only new classifier, 1 seems like only last layer but batch norm not updated, 1 same but first 10 layers/blocks
      frozen
    - rest fully fine-tuned
- **google-efficientnet-b0**
    - task type: Image classification
    - 6 models in total (6/10 worked)
    - all fully fine-tuned
- **microsoft/resnet-18**
    - task type: Image classification
    - 22 models in total (22/24 worked)
    - 2 only last layer, 4 only BN layers, rest fully fine-tuned
- **microsoft/resnet-50**
    - task type: Image classification
    - 112 models in total (112/135 worked)
    - 13 only BN layers, rest fully fine-tuned
- **microsoft/resnet-101**
    - task type: Image classification
    - 8 models in total (8/9 worked)
    - all fully fine-tuned
- **microsoft/resnet-152**
    - task type: Image classification
    - 9 models in total (9/9 worked)
    - 1 only last layer, rest fully fine-tuned

- **facebook-detr-resnet-50**
    - task type: Object Detection
    - 420 models in total (420/455 worked, maybe also only partially analyzed)
    - 13 last layers adjusted
    - 1 different backbone
    - 353 first stage of ResNet frozen
    - rest all fine-tuned
- **microsoft-conditional-detr-resnet-50**
    - task type: Object Detection
    - 47 models in total (47/49 worked)
    - 42 models first stage of resnet 50 frozen
    - rest core weights full fine-tuned
- **microsoft-table-transformer-detection**
    - task type: Object Detection
    - 12 models in total (12/12 worked)
    - all models first stage frozen (resnet ???)
- **microsoft-table-transformer-structure-recognition**
    - task type: Object Detection
    - 19 models in total (19/19 worked)
    - all models first stage frozen (resnet ???)
- **facebook-detr-resnet-101**
    - task type: Object Detection
    - 10 models in total (10/10 worked)
    - all show significant overlap
    - first stage of ResNet101 frozen (55 first layers)
    - only one of the models fully fine-tuned
    - **for analysis only consider backbone model**
- **deformable_detr_resnet_50**
    - task type: Object detection
    - 17 models in total (17/17 worked)
    - 16 models, resNet50 backbone first stage frozen
    - no model fully fine-tuned
- **facebook-detr-resnet-50-dc5**
    - task type: Object detection
    - 9 models in total (9/9 worked)
    - 7 models, resNet50 backbone first stage frozen
    - 1 fully fine-tuned

- **facebook-dinov2-base**
    - task type: Image feature extraction
    - 27 models in total (27/28 worked)
    - 3 models fully frozen, 23 models fully fine-tuned, 1 base model
- **facebook-dinov2-large**
    - task type: Image feature extraction
    - 10 models in total (10/10 worked)
    - 6 models fully frozen, 1 model (seems fully frozen but different embedding leyer), 2 models fully adjusted
- **google-vit-base-patch16-224-in21k**
    - task type: Image feature extraction
    - 1468 models in total (1468/1769 worked)
    - 37 models, only last 2 layers adjusted
    - 1 trained block 10 and 11, until 9 all frozen
    - 2 LORA so in between adjusted
    - 2 trained from block 6, until 5 frozen
    - 1 froze only block 9 - 11, rest trained
    - rest fully fine-tuned
- **google-vit-large-patch16-224-in21k**
    - task type: Image feature extraction
    - 32 models in total (32/33 worked)
    - 1 model only last layer, rest fully fine-tuned
- **google-vit-huge-patch14-224-in21k**
    - task type: Image feature extraction
    - 6 models in total (6/6 worked)
    - all models fully fine-tuned
- **hustvl-yolos-small**
    - task type: Image feature extraction
    - 14 models in total (14/14 worked)
    - all models fully fine-tuned
- **hustvl-yolos-tiny**
    - task type: Image feature extraction
    - 12 models in total (12/12 worked)
    - all models fully fine-tuned

- **nvidia-segformer-b0-finetuned-ade-512-512**
    - task type: Image segmentation
    - 39 models in total (25/39 worked)
    - all models fully fine-tuned
- **nvidia-segformer-b1-finetuned-ade-512-512**
    - task type: Image segmentation
    - 18 models in total (18/19 worked)
    - all models fully fine-tuned
- **nvidia-segformer-b2-finetuned-ade-512-512**
    - task type: Image segmentation
    - 7 models in total (7/7 worked)
    - all models fully fine-tuned
- **nvidia-segformer-b1-finetuned-cityscapes-1024-1024**
    - task type: Image segmentation
    - 8 models in total (8/8 worked)
    - all models fully fine-tuned
- **nvidia-segformer-b5-finetuned-cityscapes-1024-1024**
    - task type: Image segmentation
    - 5 models in total (5/5 worked)
    - all models fully fine-tuned

- **microsoft-trocr-large-printed**
    - task type: Image to text
    - 5 models in total (5/5 worked)
    - all models fully fine-tuned
- **microsoft-trocr-base-printed**
    - task type: Image to text
    - 8 models in total (8/9 worked)
    - all models fully fine-tuned
- **microsoft-trocr-base-stage1**
    - task type: Image to text
    - 14 models in total (14/16 worked)
    - all models fully fine-tuned
- **Salesforce-blip-image-captioning-base**
    - task type: Image to text
    - 10 models in total (10/12 worked)
    - 2 models all models backbone fully frozen, rest fully fine-tuned
- **naver-clova-ix-donut-base-finetuned-cord-v2**
    - task type: Image to text
    - 28 models in total (28/29 worked)
    - all fully fine-tuned
- **nlpconnect-vit-gpt2-image-captioning**
    - task type: Image to text
    - 11 models in total (11/11 worked)
    - 2 models no adjustments, rest fully fine-tuned
- **microsoft-git-base**
    - task type: Image to text
    - 103 models in total (103/107 worked)
    - 8 models no/minimal adjustments, rest fully fine-tuned
- **naver-clova-ix-donut-base**
    - task type: Image to text
    - 350 models in total (350/368 worked)
    - 4 models encoder identical, rest fully fine-tuned






















