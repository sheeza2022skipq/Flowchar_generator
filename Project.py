# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:17:54 2022

@author: Sheeza
"""

import cv2
import numpy as np
from skimage import util 
from utils import get_four_points

img = cv2.imread('graph.jpeg')
height, width = img.shape[:2]
print("width: ", width)
print("height: ", height)

img = cv2.resize(img, (720,800))
height, width = img.shape[:2]
print("width: ", width)
print("height: ", height)


# Display Image

cv2.imshow('image',img)
cv2.waitKey(0)

# Defining points
src_points = get_four_points(img)
dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]]) 
print(dst_points)

# Find homography using least square method (by default)
M_ls, _ = cv2.findHomography(src_points, dst_points)

dest_ls = cv2.warpPerspective(img, M_ls, (width, height))

# Drawing source points on original image

for i in range(0, 4):
    cv2.circle(img, (int(src_points[i][0]), int(src_points[i][1])), 10, (0, 255, 0), cv2.FILLED)
    

# Display Image

graph_img = dest_ls
Combined_img = cv2.hconcat([img, graph_img]) 
cv2.imshow('Homography image',Combined_img)
cv2.waitKey(0)



#convert into grayscale
gray_img = cv2.cvtColor(graph_img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale Homography image',gray_img)
cv2.waitKey(0)


thresh_otsu, binary_image = cv2.threshold(gray_img.astype(np.uint8), 0, 255,
                                         cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#draw contour
ret, binary = cv2.threshold(binary_image, 127,255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt= contours[0]
graph_image = cv2.drawContours(binary_image,[cnt],0,(0,255,0),2)


cv2.imshow('Contour on Homography image',graph_image)
cv2.waitKey(0)



# Apply Guassian blur to smooth the image
img_smooth = cv2.GaussianBlur(graph_image, (3, 3), sigmaX = 0, sigmaY = 0) 
img_smooth = cv2.fastNlMeansDenoising(img_smooth)

# Plot the image

cv2.imshow('Denoise image',img_smooth)
cv2.waitKey(0)

#   Checking image shape
r,c = img_smooth.shape

#   Creating same size empty image to hold result for binarization
binarize_img= np.zeros((r,c),np.uint8)


#   Selecting initial threshold value as the mean value of the original image.
THRESH = img_smooth.mean()

for i in range(r):
    for j in range(c):
        if(img_smooth[i,j] >= THRESH):
            binarize_img[i,j] = 255
        else:
            binarize_img[i,j]=0

cv2.imshow('Binarize image',binarize_img)
cv2.waitKey(0)
cv2.imwrite("Binarize image.jpg", binarize_img)

#inverted image

inverted_img = util.invert(binarize_img)
cv2.imshow('Inverted Binarize image',inverted_img)
cv2.waitKey(0)
cv2.imwrite("inverted Binarized_Image.jpg", inverted_img)

#Dilation Inverted Binarize image
kernel = np.ones((3,3), np.uint8)
coins_erosion = cv2.erode(inverted_img, kernel, iterations=3)
coins_dilation = cv2.dilate(inverted_img, kernel, iterations=3)
cv2.imshow('Dilation Inverted Binarize image',coins_dilation)
cv2.waitKey(0)

cv2.imwrite("Dilation Inverted Binarize image.jpg", coins_dilation)



# Canny Edge Detection
edges = cv2.Canny(image = coins_dilation, threshold1 = 100, threshold2 = 100)
cv2.imshow('Edges of Dilation Inverted Binarize image',edges)
cv2.waitKey(0)
cv2.imwrite("Edges of Dilation Inverted Binarize image.jpg", edges)