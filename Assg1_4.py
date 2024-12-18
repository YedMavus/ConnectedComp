import os
from math import floor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import otsu
from adaptive_binarization import adaptive_binarization
import time

def count_characters(image):
    start_time = time.time()
    I = adaptive_binarization(image,2) #2X2 chosen by manual observation
    height, width = I.shape

    #The loop ignores the edge pixels since characters don't appear there, will think of edge cases later

    #Set region index counter to 1
    k = 1
    R = np.zeros_like(I)

    #i = rows (along height), j = cols (along width)
    for i in range(height):
        #Ignore image extremes
        if i==0 or i==height:
            continue
        for j in range(width):
            #Ignore image extremes
            if j==0 or j==width:
                continue

            if I[i, j] == 0:  # If the pixel is part of a character
                # Gather the labels of the neighbors
                neighbors = []
                if I[i-1, j] == 0:
                    neighbors.append(R[i-1, j])
                if I[i, j-1] == 0:
                    neighbors.append(R[i, j-1])
                if I[i-1, j-1] == 0:
                    neighbors.append(R[i-1, j-1])
                if I[i-1, j+1] == 0:
                    neighbors.append(R[i-1, j+1])

                if len(neighbors)==0: # new region to be initialized
                    R[i,j]=k
                    k+=1
                else:
                    minimum=min(neighbors) # pixel part of existing region(s)
                    R[i,j]=minimum
                    for region_index in neighbors: #This loop is independent of size of image so is of O(1)
                        R=np.where(R==region_index,R[i,j],R)
    value, count = np.unique(R, return_counts=True)

    # Exclude background label '0'
    value = value[1:]  # Exclude background
    count = count[1:]  # Exclude background
    num_characters = len(value) 
    print(f'Number of characters detected including punctuation: {num_characters}')
    # Excluding punctuations

    # Determine the maximum region size
    max_pixels = np.max(count)

    # Set a threshold as a percentage of the maximum region size
    threshold_percentage = 20  # For example, 10% of the largest region size
    threshold = threshold_percentage / 100 * max_pixels
    value = value[count>threshold]
    num_characters = len(value)
    print(f'Number of characters detected excluding punctuation: {num_characters}')

    #Time before displaying
    end_time = time.time()
    print(f"assg_4 took {end_time-start_time:.4f} seconds to run")

def assg_4():
    
    script_dir = os.path.dirname(__file__)
    image_path = os.path.join(script_dir, 'images', 'quote.png')
    # script_dir = 'C:\\Users\\suvam\\OneDrive - Indian Institute of Science\\PG Sem 1\\DIP\\Assignment 1'
    # image_path = os.path.join(script_dir, 'images', 'quote.png')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    count_characters(image)
