import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
from keras import models
import pickle
import time
import imageio
from PIL import Image
import skimage as sk
import os
import cv2

images_files = "sample_2.array"
images_labels = "label_2.array"
categories = ['dyingcell', 'healthycell']
nucleus_array = []
model = tf.keras.models.load_model('.mdl_wts.hdf5')
directory = r'C:\Users\Steven\Documents\GitHub\CD-ML\Graphs'

index = 0

def run(index):
    # img_path = r'C:\Users\Steven\Documents\GitHub\CD-ML\healthycell'
    # for img in os.listdir(img_path):
    #     index += 1
    #     with open(os.path.join(img_path, img), 'r') as f:
    #         data_array, images = create_array(f)
    #     predict_image(data_array)
    #     export_files(model, index, images)
        # print_images(images)
    open_array = open('sample_3.array', 'rb')
    loaded_array = pickle.load(open_array)
    predict_image(loaded_array)

def load_images():
    false_healthy_images = open('false healthy', 'rb')
    false_dying_images = open('false dying', 'rb')
    true_healthy_images = open('true healthy', 'rb')
    true_dying_images = open('true dying', 'rb')

    loaded_f_healthy = pickle.load(false_healthy_images)
    loaded_f_dying = pickle.load(false_dying_images)
    loaded_t_healthy = pickle.load(true_healthy_images)
    loaded_t_dying = pickle.load(true_dying_images)

    # for i in range(len(loaded_f_healthy)):
    #     imageio.imwrite(r'C:\Users\Steven\Documents\GitHub\CD-ML\60x_confusion_matrix_img\f_healthy{}.png'.format(i), loaded_f_healthy[i])
    # for i in range(len(loaded_t_healthy)):
    #     imageio.imwrite(r'C:\Users\Steven\Documents\GitHub\CD-ML\60x_confusion_matrix_img\t_healthy{}.png'.format(i), loaded_t_healthy[i])
    # for i in range(len(loaded_f_dying)):
    #     imageio.imwrite(r'C:\Users\Steven\Documents\GitHub\CD-ML\60x_confusion_matrix_img\f_dying{}.png'.format(i), loaded_f_dying[i])
    # for i in range(len(loaded_t_dying)):
    #     imageio.imwrite(r'C:\Users\Steven\Documents\GitHub\CD-ML\60x_confusion_matrix_img\t_dying{}.png'.format(i), loaded_t_dying[i])

    # for i in loaded_f_healthy:
    #     healthy_scaled = cv2.normalize(i, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    #     cv2.namedWindow('myImage', cv2.WINDOW_NORMAL)
    #     cv2.imshow('myImage', healthy_scaled)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # for i in loaded_t_dying:
    #     dying_scaled = cv2.normalize(i, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    #     cv2.namedWindow('myImage', cv2.WINDOW_NORMAL)
    #     cv2.imshow('myImage', dying_scaled)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # for i in loaded_t_healthy:
    #     healthy_scaled = cv2.normalize(i, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    #     cv2.namedWindow('myImage', cv2.WINDOW_NORMAL)
    #     cv2.imshow('myImage', healthy_scaled)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    for i in loaded_f_dying:
        healthy_scaled = cv2.normalize(i, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
        cv2.namedWindow('myImage', cv2.WINDOW_NORMAL)
        cv2.imshow('myImage', healthy_scaled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def run_file():
    true_dying = 0
    true_healthy = 0
    false_dying = 0
    false_healthy = 0

    print(model.summary())

    index = 0

    false_dying_array = []
    false_healthy_array = []
    true_dying_array = []
    true_healthy_array = []

    open_array = open(images_files, 'rb')
    loaded_array = pickle.load(open_array)

    open_labels = open(images_labels, 'rb')
    loaded_label = pickle.load(open_labels)

    start = time.time()

    for i in range(len(loaded_array[:4000])):
        expanded_array = np.expand_dims(loaded_array[i], axis=0)
        classes = np.argmax(model.predict(expanded_array), axis=-1)

        if classes[0] == loaded_label[i]:
            if classes[0] == 0:
                true_dying += 1
                index += 1
                if index % 4 == 0:
                    squeezed_true_dying = np.squeeze(expanded_array)
                    true_dying_array.append(squeezed_true_dying)
            else:
                true_healthy += 1
                index += 1
                if index % 4 == 0:
                    squeezed_true_healthy = np.squeeze(expanded_array)
                    true_healthy_array.append(squeezed_true_healthy)
        else:
            if classes[0] == 1:
                index += 1
                # print("false healthy")
                squeezed_healthy = np.squeeze(expanded_array)
                healthy_scaled = cv2.normalize(squeezed_healthy, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

                if index % 4 == 0:
                    false_healthy_array.append(squeezed_healthy)
                # cv2.namedWindow('myImage', cv2.WINDOW_NORMAL)
                # cv2.imshow('myImage', healthy_scaled)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # plt.show()
                false_healthy += 1

            else:
                index += 1
                # print("false dying")
                squeezed_dying = np.squeeze(expanded_array)
                dying_scaled = cv2.normalize(squeezed_dying, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
                if index % 4 == 0:
                    false_dying_array.append(squeezed_dying)


                # cv2.namedWindow('myImage', cv2.WINDOW_NORMAL)
                # cv2.imshow('myImage', dying_scaled)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                false_dying += 1

    print("true dying: " + str(true_dying))
    print("false dying: " + str(false_dying))
    print("true healthy: " + str(true_healthy))
    print("false healthy: " + str(false_healthy))

    end = time.time()

    print((end - start)/4000)

    healthy_out = open('false dying', 'wb')
    pickle.dump(false_dying_array, healthy_out)
    healthy_out.close()

    dying_out = open('false healthy', 'wb')
    pickle.dump(false_healthy_array, dying_out)
    dying_out.close()

    healthy_out = open('true dying', 'wb')
    pickle.dump(true_dying_array, healthy_out)
    healthy_out.close()

    dying_out = open('true healthy', 'wb')
    pickle.dump(true_healthy_array, dying_out)
    dying_out.close()

def predict_image(inputArray):
    all_array = []
    for i in inputArray[:500]:
        array = []
        array.append(i)
        array = np.asarray(array)
        export_files(model, array)

def create_array(imageFile):
    if type(imageFile) == str:
        img = image.load_img(imageFile, target_size=(86, 86))
    else:
        img = image.load_img(imageFile.name, target_size=(86, 86))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    images = np.vstack([img_array])
    return img_array, images

def export_files(model, images):


    inputs = model.input
    #outputs = [model.layers[i].output for i in range(len(model.layers))]
    outputs = model.layers[4].output
    models = tf.keras.Model(inputs, outputs)
    # model = outputs to each layer

    all_layers_predictions = models.predict(images)
    #print(all_layers_predictions)

    nucleus_array.append(all_layers_predictions)



    #array_out = open("layer_4_output{}.array".format(index), "wb")
    array_out = open("layer_4_output_60x_focused_sample.array", "wb")
    pickle.dump(nucleus_array, array_out)
    array_out.close()



def print_images(images):
    layer_outputs = [layer.output for layer in model.layers[:2]] # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

    activations = activation_model.predict(images)

    layer_names = []
    for layer in model.layers[:12]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    images_per_row = 14

    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.clf()
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

run(index)
# run_file()
#print(weights)
# load_images()

















