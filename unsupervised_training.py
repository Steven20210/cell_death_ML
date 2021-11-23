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

    # outputs_array_in = open('layer_4_output_all2.array', 'rb')
    # output_array = pickle.load(outputs_array_in)
    outputs_array_in = open('sample_3.array', 'rb')
    output_array = pickle.load(outputs_array_in)

    labels_array_in = open('label_2.array', 'rb')
    labels_array = pickle.load(labels_array_in)

    # output_array1 = np.asarray(output_array)
    output_array1 = []
    imgs = []

    for i in output_array:
        for j in i:
            mean_pi, std_pi = cv2.meanStdDev(j)
            output_array1.append(std_pi[0][0])
            imgs.append(j)
            # output_array1.append(std_pi[0][0])
            # imgs.append(j)
    # for data in output_array1:
    #     data_mean, data_std = cv2.meanStdDev(data)
    #
    #     cut_off = data_std * 3
    #
    #     lower_bound, upper_bound = data_mean - cut_off, data_mean + cut_off

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
        # hist, axs = plt.subplots(1, len(possible_hist))
        # axs[1, i].set_title('PI Standard Deviation of H2O2-stimulated Nuclei Images (2650 images)', fontsize=10)
        # axs[1, i].set_xlabel("PI Standard Deviation")
        # axs[1, i].set_ylabel("# of Images")
        # axs[1, i].hist(output_array1, bins=optimal_bins, range=[0, 120])
        plt.title('PI Standard Deviation of H2O2-stimulated Nuclei Images (2650 images)', fontsize=10)
        plt.xlabel("PI Standard Deviation")
        plt.ylabel("# of Images")
        plt.hist(output_array1, bins=optimal_bins, range=[0, 120])
        saved = plt.savefig("histogram " + str(possible_hist[i]) + "x.png")
        saved_hist.append(saved)
        # plt.show()
    # hist, bin_edges = np.histogram(output_array1)
    # for i in range(len(output_array1)):
    #     if output_array1[i] > 36:
    #         print(output_array1[i])
    #         plt.imshow(imgs[i])
    #         plt.show()

    return possible_hist, output_array1, optimal_bins

    # output_array1 = np.asarray(output_array1)

    # output_array1 = output_array1.reshape(output_array1.shape[-5], -1)

    # outputs_array_1 = np.transpose(output_array1, (1, 0, 2))

    # for x in outputs_array_1:
    # for x in output_array1:
    # transformed = StandardScaler().fit_transform(x)

    # components = PCA(n_components=2)
    #
    #
    # # principalComponents = components.fit_transform(transformed)
    #
    # principalComponents = components.fit_transform(output_array1)
    #
    # variance = components.explained_variance_ratio_
    #
    # print(variance)
    # principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    #
    # print(principalDf)
    # for i in range(len(principalComponents)):
    #     # plt.plot(principalComponents[i][0], principalComponents[i][1], colors[labels_array[i]], markersize=5)
    #     plt.plot(principalComponents[i][0], principalComponents[i][1], 'g.', markersize=5)
    #
    # plt.xlabel("PCA1 - " + str(variance[0] * 100) + " %")
    # plt.ylabel("PCA2 - " + str(variance[1] * 100) + " %")
    # plt.title('PCA Plot of the Output Values of Layer 4 of the Cell Death Classification Neural Network', fontsize=10)
    # for i in range(len(principalDf)):
    #
    #     # if 0 <= principalDf.iloc[i][0] <= 15:
    #     #     #healthy_close.append(i)
    #     #     extreme.append(i)
    #     if principalDf.iloc[i][0] <= 0:
    #         extreme.append(i)
    #     # elif principalDf.iloc[i][0] > 30:
    #     #     healthy_far.append(i)
    #     else:
    #         dying.append(i)
    # plt.legend(['dying cells', 'healthy cells'])


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

    # x = np.asarray(x)
    # x = x.reshape(-1, 1)
    # y = np.asarray(y)
    #
    # cov = np.cov(x, y)
    #
    # print(cov)
    # reg = LinearRegression()
    # reg.fit(x, y)
    #
    # reg_predict = reg.predict(x)
    # plt.plot(x, reg_predict)
    # print(reg.coef_)




    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
    plt.title("Weights")
    plt.show()