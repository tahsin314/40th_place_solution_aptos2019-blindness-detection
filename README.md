# Aptos 2019 Blindness Detection: 37th Place Solution
Aravind Eye Hospital in India provided a large set of retina images taken using fundus photography under a variety of imaging conditions. A clinician has rated each image for the severity of diabetic retinopathy on a scale of 0 to 4.
The task of this competition was to train DNN that can detect the level of severity.

### Resources 
- Competition [Link](https://www.kaggle.com/c/aptos2019-blindness-detection)
- Pytorch Sample [Code](https://www.kaggle.com/abhishek/very-simple-pytorch-training-0-59)
- `Weighted Kappa` and `Metric Optimizer`  [Link](https://www.kaggle.com/tanlikesmath/intro-aptos-diabetic-retinopathy-eda-starter/notebook)
- Data augmentation library *Albumentation* [example](https://www.kaggle.com/leighplt/pytorch-tutorial-dataset-data-preparetion-stage)
- Preprocessing [Technique](https://www.kaggle.com/ratthachat/aptos-simple-preprocessing-decoloring-cropping) (borrowed from last competition winner)
- Discussion on our [strategy](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108017#latest-622514 ).


I wrote a `pytorch` and a `fastai` version of my codes for the competition. I mostly used my `fastai` codes throughout the competition. It requires a lot of work to cleanup my 
codes and add a detailed explanation of my workflow. I guess I will keep updating this repo.

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
  - Run `train.py`. It will run a 5 fold cross-validation with 10 epochs for each fold.  
  -  You can experiment with `albumentations` augmentations by using `DRDatasetAlbumentation` as your Dataset class.
  - The code monitors both `val_loss` and `kappa` scores and saves model based on them. In my experience, `kappa` score is unstable and often doesn't
  seem to be correlated with `val_loss`. The safe option here is to choose and save your model based on the `val_loss` only.
   
### Steps

- [x] Cross Validation
- [x] Learning Rate Finder
- [x] Data Augmentation (Write custom tranform classes)
  * Flip(Horizontal and Vertical)
  * Rotate
  * Zoom
  * Warp
  * Lightning
  * Symmetric Warp (Optional)
- [x] Cyclic Learning Rate ()
- [ ] Implement *Kappa* and *Focal Loss* (If possible)
- [ ] TTA (Will not be practical since it will take too much time to generate prediction
during submission)
- [ ] Explore multi-label classification