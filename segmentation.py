import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

 

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 32):
        params = {'ksize': (ksize, ksize), 'sigma': 0.0225, 'theta': theta, 'lambd': 15.0,
                  'gamma': 0.01, 'psi': 0, 'ktype': cv2.CV_32F}
        
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern, params))
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern, params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def GaborFilter(img):
    filters = build_filters()
    p = process(img, filters)
    return p



def Watershed(img, img12):
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.23*dist_transform.max(),255,0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers+1

    markers[unknown==255] = 0

    # img2 = cv2.imread(s,1)
    img2 = img12
    img2 = cv2.medianBlur(img2,5)
    markers = cv2.watershed(img2,markers)
    img2[markers == -1] = [255,0,0]

    return img2

# ret, img = cv2.threshold(img, 90, 180, cv2.THRESH_BINARY)
# cv2.imshow('a',img)
# img3 = GaborFilter(img)


# img3 = Watershed(img3)

# # cv2.imshow('s',img3)
# plt.imshow(img3,'gray')
# plt.title('Marked')
# plt.xticks([]),plt.yticks([])
# plt.show()
# cv2.waitKey(0)