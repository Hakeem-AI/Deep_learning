import numpy as np
import cv2
import glob
from sklearn.utils import shuffle
cv_img = []
cv_y = []
W1  = np.random.random(1024) * 2
W2  = np.random.random(1024) * 0.5
th = np.array([W1,W2])
for i in glob.glob('alphaData/*.png'):
    n = np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE))
    x = n.flatten()/255
    cv_img.append(x)
    cv_y.append([1, 0])

for i in glob.glob('numData/*.png'):
    n = np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE))
    x = n.flatten()/255
    cv_img.append(x)
    cv_y.append([0, 1])



X, Y = shuffle(cv_img, cv_y)


