import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from matplotlib import style
# from sci_utilities import is_outlier
import pandas as pd

style.use("ggplot")
from sklearn.cluster import MiniBatchKMeans
from keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import pickle
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageEnhance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

label = [0, 1]
array_label = np.asarray(label)
nucleus_array = []
index = 0
colors = ['r.', 'g.', 'b', '']


def pca():
    dying = []
    extreme = []

    outputs_array_in = open('sample_3.array', 'rb')
    output_array = pickle.load(outputs_array_in)

    labels_array_in = open('label_2.array', 'rb')
    labels_array = pickle.load(labels_array_in)

    output_array1 = []
    imgs = []

    for i in output_array:
        for j in i:
            mean_pi, std_pi = cv2.meanStdDev(j)
            output_array1.append(std_pi[0][0])
            imgs.append(j)

    for img in output_array1:
        if img > 120:
            output_array1.remove(img)
    # optimal_bins = np.histogram_bin_edges(output_array1, bins='fd')

    q3, q1 = np.percentile(output_array1, [75, 25])
    iqr = q3 - q1
    h = 2 * iqr * (len(output_array1) ** (-1/3))
    optimal_bins = int((np.amax(output_array1) - np.amin(output_array1))/h)

    possible_hist = [1, 1.5, 2, 2.5, 3]

    saved_hist = []
    for i in range(len(possible_hist)):
        optimal_bins = int(possible_hist[i] * optimal_bins)
        plt.title('PI Standard Deviation of H2O2-stimulated Nuclei Images (2650 images)', fontsize=10)
        plt.xlabel("PI Standard Deviation")
        plt.ylabel("# of Images")
        plt.hist(output_array1, bins=optimal_bins, range=[0, 120])
        saved = plt.savefig("histogram " + str(possible_hist[i]) + "x.png")
        saved_hist.append(saved)


    return possible_hist, output_array1, optimal_bins


a, b, c = pca()
print(a, b, c)

def kmeans_clustering(X):
    total_clusters = len(np.unique(array_label))

    kmeans = MiniBatchKMeans(n_clusters= total_clusters)

    kmeans.fit(X)

    labels = kmeans.labels_

    centroids = kmeans.cluster_centers_

    index = 0
    index1 = 0
    index2 = 0

    print(labels)
    for i in labels:
        if i == 0:
            index += 1
        elif i == 1:
            index1 += 1
        elif i == 2:
            index2 += 1
    print(str(index) + " : 0 ," + str(index1) + " : 1 ," + str(index2) + " : 2")

    return centroids, labels

def show_cluster(centroids, labels, X):
    colors = ["g.", "r.", "c.", "y."]

    x = []
    y = []

    for i in range(len(X)):
        #print("coordinate:", X[i], "label:", labels[i])
        #plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
        x.append(X[i][0])
        y.append(X[i][1])
    for i in range(len(x)):
        plt.plot([i], x[i], "g.", markersize=10)
        plt.plot([i], y[i], 'r.', markersize=10)



    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
    plt.title("Weights")
    plt.show()
