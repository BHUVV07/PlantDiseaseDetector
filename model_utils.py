
import numpy as np
from PIL import Image
import cv2
from skimage.feature import hog

def preprocess_image_for_ml(pil_img, size=(128,128)):
    """Convert PIL image to feature vector for classical ML:
       - resize
       - color histogram (RGB)
       - HOG features (grayscale)
    """
    img = pil_img.copy()
    img = img.resize(size)
    arr = np.array(img)
    # color histograms
    chans = cv2.split(arr)
    hist_features = []
    for ch in chans:
        hist = cv2.calcHist([ch], [0], None, [32], [0,256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        hist_features.extend(hist.tolist())
    # HOG on grayscale
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hog_feat = hog(gray, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
    feat = np.concatenate([np.array(hist_features), hog_feat])
    return feat
