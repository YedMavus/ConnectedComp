import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from histogram import Histogram


def Otsu(image, type='BetweenClassVariance', variance_out=False):
    if type == 'BetweenClassVariance':
        result = OtsuThreshold_BC(image)
    else:
        result = OtsuThreshold_WC(image)
    
    if variance_out:
        optimal_threshold, variance = result
    else:
        optimal_threshold = result
        variance = None

    print(f'Optimal Threshold using {type} = {optimal_threshold}')
    return BinarizeImage(image, optimal_threshold), variance


def WithinClassVariance(image, t):
    height, width = image.shape
    size = float(height * width)
    
    # Calculate histogram and global mean
    histogram = Histogram(image)
    histogram = histogram / size

    # Weight (probability) for class 0 and class 1
    w_0 = np.sum(histogram[:t + 1])
    w_1 = np.sum(histogram[t + 1:])
    
    if w_0 == 0 or w_1 == 0:
        return float('inf')  # To avoid invalid variance calculation and ensure proper minimization (hence inf and not 0)

    # Mean for class 0 and class 1
    u_0 = np.sum(np.arange(t + 1) * histogram[:t + 1]) / w_0
    u_1 = np.sum(np.arange(t + 1, 256) * histogram[t + 1:]) / w_1

    # Variance for class 0 and class 1
    var_0 = np.sum((np.arange(t + 1) - u_0)**2 * histogram[:t + 1]) / w_0
    var_1 = np.sum((np.arange(t + 1, 256) - u_1)**2 * histogram[t + 1:]) / w_1

    return w_0 * var_0 + w_1 * var_1


def OtsuThreshold_BC(image):
    height, width = image.shape
    size = float(height * width)
    
    # Calculate histogram
    histogram = Histogram(image)
    
    # Normalize the histogram
    histogram = histogram / size
    
    # Cumulative sum of the histogram (weight for class 0)
    w_0 = np.cumsum(histogram)
    w_1 = 1 - w_0  # Weight for class 1
    
    # Cumulative sum of the weighted histogram (mean for class 0)
    sum_all = np.cumsum(np.arange(256) * histogram)
    u_0 = sum_all / w_0
    u_1 = (sum_all[-1] - sum_all) / w_1
    
    # Avoid division by zero in mean calculations
    u_0[w_0 == 0] = 0
    u_1[w_1 == 0] = 0

    # Calculate between-class variance
    variances = w_0 * w_1 * (u_0 - u_1) ** 2
    best_threshold = np.argmax(variances)
    return best_threshold, variances
 


def OtsuThreshold_WC(image):
    # Calculate within class variance for all thresholds
    variances = np.array([WithinClassVariance(image, t) for t in range(256)])
    # Threshold that minimizes the variance
    best_threshold = np.argmin(variances)    
    return best_threshold, variances


def BinarizeImage(image, threshold):
    # Create a binary image based on the threshold
    binary_image = np.zeros_like(image)
    binary_image[image > threshold] = 1
    return binary_image
