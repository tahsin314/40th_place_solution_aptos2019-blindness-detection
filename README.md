# Aptos 2019 Blindness Detection: 37th Place Solution (Updated)
Aravind Eye Hospital in India provided a large set of retina images taken using fundus photography under a variety of imaging conditions. A clinician has rated each image for the severity of diabetic retinopathy on a scale of 0 to 4.
The task of this competition was to train a network that can detect the level of severity.

### Our approach
I treated the task as a regression problem. I experimented with several Resnet, SE-Resnet, Resnet, ResNext and SE-Resnext models until
I found out that [Efficientnet](https://arxiv.org/abs/1905.11946) model performs better than most
of the pretrained models in this competition. Our final silver winning solution was an ensemble of 5 different
Efficientnet models. Here is an overview:

- **Preprocessing**:   Only Circle Cropping from [this](https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resnet50-binary-cropped) kernel.
- **Augmentations**: Horizontal and Vertical Flip, 360 rotation, zoom 20%-25%, lighting 50%. Used Fast.ai transformations.

- **Training**:  Mostly followed [this](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/100815#latest-582005) method. Did 5+10 training using `2015` data as train and `2019` as validation set. Then I split the `2019` data into 80:20 ratio using `sklearn`'s Stratified Split function and finetuned the previously trained model over it for 15 epochs. I took the best model based on validation loss which didn't seem to be overfitted and used it for submission. I used a fixed seed so that I can compare different model performances over the same val set. Didn't use CV or TTA.
I wrote a `pytorch` and a `fastai` version of my codes for the competition.

**I mostly used my `fastai` codes throughout the competition. It requires a lot of work to cleanup my 
codes and add a detailed explanation of my workflow. I guess I will keep updating this repo.** 

### Approaches That Didn't Work For Me
Classification and Ordinal Regression 

### Scores (TL;DR)
Here is the summary of my final models and their ensemble scores: 

| Model           | Image Size | Val Kappa | Public Kappa | Private Kappa |
|-----------------|------------|-----------|--------------|---------------|
| Efficientnet B2 | 256        | 0\.922    | 0\.807       | 0\.918        |
| Efficientnet B1 | 256        | 0\.926    | 0\.804       | 0\.921        |
| Efficientnet B0 | 256        | 0\.919    | 0\.816       | 0\.914        |
| Efficientnet B3 | 256        | 0\.921    | 0\.812       | 0\.917        |
| Efficientnet B5 | 300        | 0\.920     | 0\.802       | 0\.916        |
| Ensemble        | ----      | 0\.932    | 0\.826       | 0\.926        |
 
### Resources 
- Competition [Link](https://www.kaggle.com/c/aptos2019-blindness-detection)
- Pytorch Sample [Code](https://www.kaggle.com/abhishek/very-simple-pytorch-training-0-59)
- `Weighted Kappa` and `Metric Optimizer`  [Link](https://www.kaggle.com/tanlikesmath/intro-aptos-diabetic-retinopathy-eda-starter/notebook)
- Data augmentation library *Albumentation* [example](https://www.kaggle.com/leighplt/pytorch-tutorial-dataset-data-preparetion-stage)
- Preprocessing [Technique](https://www.kaggle.com/ratthachat/aptos-simple-preprocessing-decoloring-cropping) (borrowed from last competition winner)
- Discussion on our [strategy](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108017#latest-622514 ).
-  All gold medal [solutions and discussions](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108307#latest-624716)


### Requirements
- torch
- torchvision
- fastai
- sklearn
- numpy
- tqdm

### PyTorch 
- **Steps**
  - Create a directory `data/new_data` and keep the 2019 APTOS competition [data](https://www.kaggle.com/c/aptos2019-blindness-detection/data) there.
  - Run `cyclic_lr.py` and find a suitable learning rate.
  - Run `cd pytorch_version` and then `train.py`. It will run a 5 fold cross-validation with 10 epochs for each fold.  
  -  You can experiment with `albumentations` augmentations by using `DRDatasetAlbumentation` as your Dataset class.
  - The code monitors both `val_loss` and `kappa` scores and saves model based on them. In my experience, `kappa` score is unstable and often doesn't
  seem to be correlated with `val_loss`. The safe option here is to choose and save your model based on the `val_loss` only.

### Fastai
- **Steps**


**Will be added soon**