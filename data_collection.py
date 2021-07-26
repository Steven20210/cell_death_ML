import random
import os
import cv2
import pickle
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import collections
from pandas import DataFrame
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image, ImageEnhance, ImageStat

sample_label_array = []
shuffle_array = []
sample_array = []
label_array = []

pi_array = []
areas_array = []
std_array = []

pi_std_array = []
# Calcein STD array
c_std_array = []

# Stores the centroids of each image:
parent_image_centroid_dictionary = {}


# Make sure to implement if the cells resemble a circle later

def sort_files(parent_folder, stimuli):
    file_array = []

    dapi_files = []
    pi_files = []
    c_files = []

    dapi_dictionary = {}
    pi_dictionary = {}
    c_dictionary = {}

    first_folder = os.listdir(parent_folder)[0]
    image_directory = os.path.join(parent_folder, first_folder)
    image_files = os.listdir(image_directory)
    for img_file in image_files:
        imgs_array = [os.path.join(image_directory, img_file)]
        sub_folder_total = len(os.listdir(parent_folder))
        for index in range(2, sub_folder_total):
            sub_folder = "TimePoint_" + str(index)
            temp_file = os.path.join(parent_folder, sub_folder, img_file)
            if os.path.exists(temp_file):
                imgs_array.append(temp_file)
        file_array.append(imgs_array)

    stimulus_index = -5
    for img_array in file_array:
        for img in img_array:
            try:
                if img[-1] != 'D':
                    if img[stimulus_index] == '1':
                        dapi_files.append(img_array)
                    elif img[stimulus_index] == '3':
                        pi_files.append(img_array)
                    # elif img[stimulus_index] == '2':
                    #     c_files.append(img_array)
            except:
                continue

    assign_key(dapi_files, dapi_dictionary)
    assign_key(pi_files, pi_dictionary)
    # assign_key(c_files, c_dictionary)

    for key in pi_dictionary:
        if key in dapi_dictionary:
            continue
        else:
            del pi_dictionary[key]
            # del c_dictionary[key]

    for key in dapi_dictionary:
        if key in pi_dictionary:
            pass
        else:
            del dapi_dictionary[key]
            # del c_dictionary[key]

    # for key in c_dictionary:
    #     if key in pi_dictionary and dapi_dictionary:
    #         pass
    #     else:
    #         del dapi_dictionary[key]
    #         del pi_dictionary[key]

    sorted_dapi = collections.OrderedDict(sorted(dapi_dictionary.items()))
    sorted_pi = collections.OrderedDict(sorted(pi_dictionary.items()))
    # sorted_c = collections.OrderedDict(sorted(c_dictionary.items()))

    sorted_dapi, possible_label = sort_stimulants(sorted_dapi, stimuli)
    sorted_pi, possible_label_pi = sort_stimulants(sorted_pi, stimuli)
    # sorted_c, possible_label_c = sort_stimulants(sorted_c, stimuli)

    return sorted_pi, sorted_dapi, possible_label


def assign_key(stim_arr, stim_dic):
    for img_array in stim_arr:
        for img in img_array:
            try:
                if img[-9] == '1':
                    stim_dic[img[-13] + img[-12] + img[-11] + img[-9] + img[-8]] = img_array
                else:
                    stim_dic[img[-13] + img[-12] + img[-11] + img[-8]] = img_array
            except:
                continue


# Appends the images in groups of 3 (group by stimulants) because that's how they were organized
def sort_stimulants(dictionary, stimuli):
    blank = []
    staurosporin = []
    h2o2 = []
    nigericin = []

    blank_label = ['1', '2', '3']
    staurosporin_label = ['4', '5', '6']
    h2o2_label = ['7', '8', '9']
    nigericin_label = ['1']
    for file_name, imgs in zip(dictionary.keys(), dictionary.values()):
        if file_name[-3] == nigericin_label[0]:
            nigericin.append(imgs)
        else:
            if file_name[-2] in blank_label:
                blank.append(imgs)
            elif file_name[-2] in staurosporin_label:
                staurosporin.append(imgs)
            elif file_name[-2] in h2o2_label:
                h2o2.append(imgs)
            else:
                nigericin.append(imgs)

    dic = {'blank': blank, 'staurosporin': staurosporin, 'h2o2': h2o2, 'nigericin': nigericin}

    if stimuli == 'all':
        img_array = [blank, staurosporin, h2o2, nigericin]
    else:
        img_array = [dic[stimuli][:10], staurosporin[:10]]

    # This will create an array detailing the possible labels that could be associated with each nuclei
    # (aka 0, 1, 2, 3) each of them will corresponds with healthy, staurosporin_dying, etc.

    possible_labels = []
    for i in range(len(img_array) * 2):
        possible_labels.append(i)

    return img_array, possible_labels


def store_arrays(possible_labels):
    count_arr = []
    org_arr = []

    # k_means_clustering(pi_std_array, c_std_array)

    # Returns the count of each label as well as stores the imgs in org_array
    for label in range(len(possible_labels)):
        label_count = sum([num_label.count(label) for num_label in sample_label_array])
        img_label_array = []
        for file, img_label, area in sample_label_array:
            if img_label == label:
                img_label_array.append([file, img_label, area])
                # del sample_label_array[file, img_label]
        count_arr.append([label, label_count])
        org_arr.append(img_label_array)

    # random.shuffle(org_arr[1])

    # Removes the outliers from the dataset
    area_array = []
    for img_label_array in org_arr:
        for img_label in img_label_array:
            area_array.append(img_label[2])

    area_array = np.asarray(area_array)
    data_mean, data_std = cv2.meanStdDev(area_array)

    cut_off = data_std * 3

    lower_bound, upper_bound = data_mean - cut_off, data_mean + cut_off

    # Removes the outliers within the dataset (if the area is outside of 98 percentile-tile)
    for index in range(len(org_arr)):
        for img_label in reversed(range(len(org_arr[index]))):
            if org_arr[index][img_label][2] > upper_bound or org_arr[index][img_label][2] < lower_bound:
                del org_arr[index][img_label]  # org_arr[index].remove(img_label)

    # Updates the number of images per label
    for label in range(len(possible_labels)):
        count_arr[label][1] = len(org_arr[label])

    # Find the dataset with the lowest number of imgs
    lowest_num = count_arr[0][1]
    for label_index in reversed(range(len(count_arr))):
        if count_arr[label_index][1] == 0:
            del org_arr[label_index]
        if count_arr[label_index][1] != 0:
            if count_arr[label_index][1] < lowest_num:
                lowest_num = count_arr[label_index][1]

    # Slices the array with a greater number of imgs so that the dataset is even
    for index in range(len(org_arr)):
        org_arr[index] = org_arr[index][:lowest_num]

    # for label_index in range(1, len(count_arr), 2):
    #     if count_arr[label_index][1] > count_arr[label_index - 1][1]:
    #         num = count_arr[label_index - 1][1]
    #         org_arr[label_index] = org_arr[label_index][:num]
    #     else:
    #         num = count_arr[label_index][1]
    #         org_arr[label_index - 1] = org_arr[label_index - 1][:num]

    # Verifying images
    # for stim in org_arr:
    #     for index in range(0, len(stim), 5):
    #         plt.imshow(stim[index][0][0])
    #         plt.imshow(stim[index][0][1])
    #         plt.imshow(stim[index][0][2])
    #         plt.imshow(stim[index][0][3])
    #         print(stim[index][1])

    # Splitting the nuclei img and label into separate arrays
    for stim in org_arr:
        for files, labels, area in stim:
            for file in files:
                # Augment the data
                temp_array = []
                temp_array.append([file, labels])
                temp_array.append([np.array(tf.image.flip_left_right(file)), labels])
                # # temp_array.append([np.array(tf.image.random_brightness(file, max_delta=0.5)), labels])
                temp_array.append([np.array(tf.image.rot90(file)), labels])
                temp_array.append([np.array(tf.image.random_flip_left_right(file)), labels])
                temp_array.append([np.array(tf.image.random_flip_up_down(file)), labels])
                for img in temp_array:
                    shuffle_array.append(img)
            # squeezed_file = np.squeeze(file)
            # plt.imshow(squeezed_file)
            # plt.show()
            # print(labels)

    random.shuffle(shuffle_array)

    # dapi_arr = []
    # for file_label in shuffle_array:
    #     file_label[0][0]
    for file_label in shuffle_array:
        if type(file_label[0]) != list:
            sample_array.append(file_label[0])
            label_array.append(file_label[1])
        else:
            sample_array.append(file_label[0][0])
            label_array.append(file_label[1])

    print(sample_array)


def calculate_centroid(c, index, centroid_number, centroid_array, image_centroid_dictionary):
    # calculate the moments of the binary image
    moments = cv2.moments(c)

    # calculate the coordinates of the center of the image
    if moments["m00"] != 0:
        cx = int(moments['m10'] / moments["m00"])
        cy = int(moments['m01'] / moments["m00"])
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if w > 50 and h > 50 and area < 100000:
            # appends to external list of centroids
            if index == 0:
                # Create ID for cell
                image_centroid_dictionary['centroid {}'.format(centroid_number)] = [[cx, cy, c]]
            else:
                # if it is another cell in the time point, then append to a list of arrays to be sorted out later
                centroid_array.append([cx, cy, c])
    else:
        pass


def compare_centroids(cx, cy, possible_centroids):
    smallest_distance = int()
    img = []
    for centroid in possible_centroids:
        # calculates the distance between the centroids using Pythagorean's theorem
        dist = math.sqrt((int(centroid[0]) - int(cx)) * (int(centroid[0]) - int(cx)) + (int(centroid[1]) - int(cy)) * (
                int(centroid[1]) - int(cy)))
        if dist < smallest_distance or smallest_distance == int():
            smallest_distance = dist
            img = [centroid[0], centroid[1], centroid[2]]
    # Returns the x, y coordinates as well as the img array of the next cell within the frame
    return img


def kmeans_preprocessing(img, img_array):
    # Determine STD
    arr = img.astype('float')
    arr[arr == 0] = None
    std_img = np.nanstd(arr)

    img_array.append(std_img)


def k_means_clustering(pi_std_arr, c_std_arr):
    arr = []
    x = []
    y = []

    for i in range(len(pi_std_arr)):
        arr.append([pi_std_arr[i], c_std_arr[i]])

    total_clusters = 2

    kmeans = MiniBatchKMeans(n_clusters=total_clusters)

    kmeans.fit(arr)

    centroids = kmeans.cluster_centers_

    for index in arr:
        x.append(index[0])
        y.append(index[1])

    pi_values = {'size': x,
                 'bright pixels': y}

    df = DataFrame(pi_values, columns=['size', 'bright pixels'])
    plt.scatter(df['size'], df['bright pixels'], c=kmeans.labels_.astype(float), s=10, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
    plt.xlim(0, 120)
    # plt.ylim()
    plt.title("# of bright pixels per PI image in relation to image size (staurosporin)")
    plt.xlabel("Size of PI Image")
    plt.ylabel("# of Bright Pixels")
    plt.show()


def apply_mask(img, contour, x, y, w, h):
    # Applying a mask to the image to isolate ROI
    # Create Mask
    zero_mask = np.zeros(img.shape).astype(img.dtype)

    white = [1]
    cv2.drawContours(zero_mask, [contour[-1]], 0, white, -1)

    res_img = zero_mask * img

    res_img = res_img[y:y + h, x:x + w]

    return res_img


def process_image(dapi, pi, index, labels, centroid_array, image_centroid_dictionary, threshold_value, stimuli):
    mean, std_dapi = cv2.meanStdDev(dapi)
    if std_dapi > 30:
        # plt.imshow(pi)
        # Converts the color in the image to grey scale
        img = cv2.cvtColor(dapi, cv2.COLOR_GRAY2BGR)
        # Normalizes/converts the 16-bit image to an 8-bit image
        img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dapi_img = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)

        dapi_img = cv2.GaussianBlur(dapi_img, (9, 9), 0)

        ret, mask = cv2.threshold(dapi_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Contours are curves joining all continuous points while in our case having the same intensity
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # assigns new identity to each centroid
        centroid_number = 0

        # Smoothens the PI img
        pi_smoothed = cv2.GaussianBlur(pi, (9, 9), 0)

        for a in contours:
            calculate_centroid(a, index, centroid_number, centroid_array, image_centroid_dictionary)
            centroid_number += 1

        if index >= 1:
            # for each cell in an image
            for key in image_centroid_dictionary.keys():  # FIX
                # appends the list of possible arrays to the dictionary
                image_centroid_dictionary[key].append(centroid_array)
                next_cell = compare_centroids(image_centroid_dictionary[key][-2][0],
                                              image_centroid_dictionary[key][-2][1],
                                              image_centroid_dictionary[key][-1])
                # pushes the cell within the next frame to the end of the array
                image_centroid_dictionary[key][-1] = next_cell

                '''Current format after Timepoint 2
                image dictionary = {ID: [[x1, y1, centroid], [x2, y2, centroid], ...} 
                
                What we want: 
                image dictionary = {ID: [[x1, y1, segmented image, label]...}
                
                How to implement:
                Instead of appending to image label array, append to the dictionary directly 
                Maybe you can use dictionary.values() 
                
                dying = False 
                for each key in dictionary: 
                    for arr in key.values():
                        if 0 in arr:
                            dying = True 
                            break
                            
                if dying == True:
                    for arr in dictionary.value():
                        arr[-1] = 0
                At this point each of the labels should be changed to 0 if a dying cell exists in their timepoint
                '''

        for nuclei_id in image_centroid_dictionary.keys():
            for nuclei in image_centroid_dictionary[nuclei_id]:

                x, y, w, h = cv2.boundingRect(nuclei[-1])

                res_dapi = apply_mask(dapi, nuclei, x, y, w, h)
                res_pi = apply_mask(pi, nuclei, x, y, w, h)
                # res_c = apply_mask(c, nuclei, x, y, w, h)

                area = w * h

                # Converts pixel values of 0 to NaN to not influence std calculation
                arr = res_dapi.astype('float')
                arr[arr == 0] = None
                std_dapi_img = np.nanstd(arr)

                # Ensuring that there is a cell there in the DAPI image
                if std_dapi_img > 100:

                    std_array.append(std_dapi_img)
                    #
                    # kmeans_preprocessing(res_pi, pi_std_array)
                    # kmeans_preprocessing(res_c, c_std_array)

                    squeezed_dapi = cv2.merge((res_dapi, res_dapi, res_dapi), -1)
                    dapi_img_resized = cv2.resize(squeezed_dapi, (86, 86), -1)

                    squeezed_pi = cv2.merge((res_pi, res_pi, res_pi), -1)
                    pi_img_resized = cv2.resize(squeezed_pi, (86, 86), -1)

                    if stimuli != 'all':
                        augmented_images = [res_pi, res_dapi, res_dapi]
                    else:
                        augmented_images = [dapi_img_resized]
                    # augmented_images = [res_pi, pi_img, res_dapi, res_dapi]

                    # augmented_images = [[res_dapi]]

                    arr_pi = res_pi.astype('float')
                    arr_pi[arr_pi == 0] = None
                    std_pi_img = np.nanstd(arr_pi)

                    if std_pi_img > threshold_value:
                        # plt.imshow(res_pi)
                        # plt.imshow(pi_img)
                        # plt.imshow(res_dapi)
                        # plt.imshow(res_dapi)
                        # plt.show()

                        sample_label_array.append([augmented_images, labels[0], area])
                        pi_array.append(res_pi)

                        # store_arrays()

                    else:
                        # plt.imshow(res_pi)
                        # plt.imshow(pi_img)
                        # plt.imshow(res_dapi)
                        # plt.imshow(res_dapi)
                        # plt.show()
                        sample_label_array.append([augmented_images, 1, area])
                        pi_array.append(res_pi)

                    # store_arrays()


    else:
        pass


def generate_training_data(directory, stimuli='all', dictionary={}):
    pi_values, dapi_values, possible_labels = sort_files(directory, stimuli)

    for stim_index in range(len(dapi_values)):
        mod_index = stim_index + 1
        mod_index = (mod_index * 2)
        pi_k_means_array = []
        # labels[0] = stim_dying, labels[1] = stim_healthy
        labels = [possible_labels[mod_index - 2], possible_labels[mod_index - 1]]
        # for parent_img_index in range(len(dapi_values[stim_index])):
        # for parent_img_index in range(1):

        thresh_hold = 0

        # assigning threshold values
        try:
            if stim_index == 0:
                thresh_hold = dictionary['blank']
            elif stim_index == 1:
                thresh_hold = dictionary['staurosporin']
            elif stim_index == 2:
                thresh_hold = dictionary['h2o2']
            elif stim_index == 3:
                thresh_hold = dictionary['nigericin']
        except:
            thresh_hold = 38

        for parent_img_index in range(len(dapi_values[stim_index])):
            # Stores the centroids of each image (without an ID):
            pi_frame_array = []
            centroid_array = []
            image_centroid_dictionary = {}
            index = 0
            for img_frame_index in range(2):
                # for img_frame_index in range(len(dapi_values[stim_index][parent_img_index])):
                dapi_img = cv2.imread(dapi_values[stim_index][parent_img_index][img_frame_index], -1)
                pi_img = cv2.imread(pi_values[stim_index][parent_img_index][img_frame_index], -1)
                # c_img = cv2.imread(c_values[stim_index][parent_img_index][img_frame_index], -1)
                process_image(dapi_img, pi_img, index, labels, centroid_array,
                              image_centroid_dictionary, thresh_hold, stimuli)
                index += 1

    store_arrays(possible_labels)

    array_out = open("sample_3.array", "wb")
    pickle.dump(sample_array, array_out)
    array_out.close()

    array_out = open("label_3.array", 'wb')
    pickle.dump(label_array, array_out)
    array_out.close()

    array_out = open("pi.array", 'wb')
    pickle.dump(pi_array, array_out)
    array_out.close()

    print(label_array)


generate_training_data(r"C:\Users\Kerr Lab\Desktop\CD-ML-myNewBranch\60_x_frames", stimuli='blank')
