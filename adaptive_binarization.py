import otsu
import cv2
import numpy as np
from math import floor

def adaptive_binarization(image, block_mul=1):
    '''adaptive_binarization(image, block_mul=1) returns the binarized image obtained using adaptive binarization. It inputs a grayscale image "image"
      and by default applies Otsu's algorithm on the entire image (ie block size 1). NXN block size can be specified by passing the argument 
      block_mul = N'''
    height, width = image.shape
    block_height, block_width = floor(height/block_mul), floor(width/block_mul)
    binarized_image = np.zeros_like(image)

    # Iterate through each block
    for i in range(0, height, block_height):
        for j in range(0, width, block_width):
            # Extract the current block
            block = image[i:i + block_height, j:j + block_width]
            
            # Apply Otsu's method to the block
            block_threshold,_ = otsu.OtsuThreshold_BC(block)
            binarized_block = otsu.BinarizeImage(block, block_threshold)
            
            # Place the binarized block in the output image
            binarized_image[i:i + block_height, j:j + block_width] = binarized_block

    return binarized_image