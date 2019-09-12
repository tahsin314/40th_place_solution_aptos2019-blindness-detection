import cv2
from PIL import Image
import numpy as np


def crop_image(img, tol=7):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image1(img, tol=7):
    # img is image data
    # tol  is tolerance

    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


def open_aptos2019_image(fn, convert_mode, after_open) -> Image:
    SIZE = sz
    image = cv2.imread(fn)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = crop_image(image)
    image = cv2.resize(image, (SIZE, SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), SIZE / 10), -4, 128)
    return Image(pil2tensor(image, np.float32).div_(255))


def load_ben_color(fn, convert_mode, after_open) -> Image:
    sigmaX = 10
    IMG_SIZE = sz
    image = cv2.imread(fn)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return Image(pil2tensor(image, np.float32).div_(255))


def load_ben_color_pre(fn, sz, aug=True):
    sigmaX = 10
    IMG_SIZE = sz
    try:
        image = cv2.imread(fn)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        if aug == True:
            image = apply_album(image, IMG_SIZE)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
        return image
    except:
        print(fn)
        pass


def open_aptos2019_image_pre(fn):
    SIZE = sz
    image = cv2.imread(fn)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = crop_image(image)
    image = cv2.resize(image, (SIZE, SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), SIZE / 10), -4, 128)
    return image