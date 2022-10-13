from knn import X
from knn import th
from knn import Y
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
l = len(X)
W = th
alph = np.zeros((26, 2))
numr = np.zeros((10, 2))
o = 0
p = 0
K = 1
L = 1
b = np.array([1, 1])
dist1 = []
dist2 = []
dist3 = []
dist4 = []


def gettest():
    n1 = np.array(cv2.imread('A.png', cv2.IMREAD_GRAYSCALE))
    n1 = n1.flatten() / 255
    n2 = np.array(cv2.imread('N.png', cv2.IMREAD_GRAYSCALE))
    n2 = n2.flatten() / 255
    scor1 = farword(n1, W, b) / 222
    scor2 = farword(n2, W, b) /222
    return scor1,scor2
def farword(x,w,b):
    h = np.dot(w, x)
    h = h + b
    return  h


def distman(c1,c2,test1,test2):

    for i in range(len(c1)):
        dist1.append(math.dist(test1, c1[i]))
        dist2.append(math.dist(test2, c1[i]))
    for i in range(len(c2)):
        dist3.append(math.dist(test1, c2[i]))
        dist4.append(math.dist(test2, c2[i]))

    if min(dist1) >= min(dist3):
        print("test image 1 belongs to class1")
    else:
        print("test image 1 belongs to class2")
    if min(dist2) >= min(dist4):
        print("test image 2 belongs to class1")
    else:
        print("test image 2 belongs to class2")

for i in range(l):
    out =  farword(X[i], W, b)
    if Y[i] == [0, 1]:
        numr[o] = out / 200
        o = o +1
    elif Y[i] == [1, 0]:
        alph[p] = out / 235
        p = p +1



s1, s2 = gettest()



distman(alph,numr,s1,s2)
x1 = alph[:, 0]
y1 = alph[:, 1]
x2 = numr[:, 0]
y2 = numr[:, 1]
x3 = s1[0]
y3 = s1[1]
x4 = s2[0]
y4 = s2[1]
plt.scatter(x1, y1, marker='D')
plt.scatter(x2, y2, marker= '*')
plt.scatter(x3, y3, marker= 'D')
plt.scatter(x4, y4, marker= '*')
plt.show()


