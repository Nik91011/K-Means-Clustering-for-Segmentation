"""
K-Means Segmentation Problem
The goal of this task is to segment image using k-means clustering.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are allowed to add your own functions if needed.
You should design you algorithm as fast as possible. To avoid repetitve calculation, you are suggested to depict clustering based on statistic histogram [0,255]. 
You will be graded based on the total distortion, e.g., sum of distances, the less the better your clustering is.
"""


import utils
import numpy as np
import json
import time


def kmeans(img,k):
    """
    Implement kmeans clustering on the given image.
    Steps:
    (1) Random initialize the centers.
    (2) Calculate distances and update centers, stop when centers do not change.
    (3) Iterate all initializations and return the best result.
    Arg: Input image;
         Number of K. 
    Return: Clustering center values;
            Clustering labels of all pixels;
            Minimum summation of distance between each pixel and its center.  
    """
    # TODO: implement this function.
    p = img.flatten().reshape(-1, 1)
    b_c = None
    b_sd = float('inf')
    hist, bins = np.histogram(p, bins=256, range=(0,256))
    c_1 = np.array([bins[np.argmax(hist)], bins[len(bins) - 1 - np.argmax(hist[::-1])]])
    p_c = np.zeros_like(c_1)
    
    while not np.all(c_1 == p_c):
        d_l = np.abs(p - c_1.reshape(1, -1))
        l_1 = np.argmin(d_l, axis=1)
        p_c = c_1.copy()
        for i in range(k):
            c_p = p[l_1 == i ]
            if len(c_p) == 0:
                c_1[i] = bins[np.argmax(hist)]
            else:
                c_1[i] = np.mean(c_p)
    s = np.min(d_l, axis=1)
    s_1 = np.sum(s)
    b_sd = int(s_1)
    b_c = [int(center) for center in c_1]

    return b_c, l_1, b_sd


def visualize(centers,labels):
    """
    Convert the image to segmentation map replacing each pixel value with its center.
    Arg: Clustering center values;
         Clustering labels of all pixels. 
    Return: Segmentation map.
    """
    # TODO: implement this function.

    image = np.array([centers[label] for label in labels])
    s_g = image/255.0
    s = int(np.sqrt(len(labels))), int(np.sqrt(len(labels)))
    return s_g.reshape(s)

     
if __name__ == "__main__":
    img = utils.read_image('./images/lenna.png')
    k = 2

    start_time = time.time()
    centers, labels, sumdistance = kmeans(img,k)
    result = visualize(centers, labels)
    end_time = time.time()

    running_time = end_time - start_time
    print(running_time)

    centers = list(centers)
    with open('t1.json', "w") as jsonFile:
        jsonFile.write(json.dumps({"centers":centers, "distance":sumdistance, "time":running_time}))
    utils.write_image(result, 't1.png')
