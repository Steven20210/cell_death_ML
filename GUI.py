import PIL.Image
from PIL import ImageTk as itk
import cv2
from tkinter import *
from tkinter import ttk
import os
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

HEIGHT = 700
WIDTH = 800
categories = ['dyingcell', 'healthycell']

nucleus_array = []
stim_thresh = {}
# model = tf.keras.models.load_model('cell_cnn.h5')
index = 0


class Page(Frame):
    def __init__(self):
        Frame.__init__(self)


class GUI(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        master.title("Machine Learning Cell Death")

        self.canvas = Canvas(master, height=HEIGHT, width=WIDTH)
        self.canvas.grid()

        # When choosing a color go to html color picker
        self.ui_notebook = ttk.Notebook(master)
        self.ui_notebook.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)

        # Designing Home Tab
        self.home_tab = self.make_window(self.ui_notebook, 'Home')
        self.scroll = Scrollbar(self.home_tab)
        self.t = Text(self.home_tab, height=30, width=75)
        self.t.pack(side=LEFT, fill=Y)
        self.scroll.pack(side=RIGHT, fill=Y)
        self.scroll.config(command=self.t.yview)
        self.t.config(yscrollcommand=self.scroll.set)
        intro = '''
            Hello!
            Welcome to Cell Death Classification Program!

            Here are a list of the tabs and what they do within our program: 

            Make Hist Tab: 
            This window helps to make histograms based off of the 
            standard deviation of the brightness
            of the pixels within your images. Here, the number of bins 
            within the histogram
            can be controlled to better locate the threshold values
             of your dataset.

            Preprocessing Tab:
            This tab sorts your cell images into individual images, as well as 
            assigns them a label based off of the threshold value 
            provided in the previous tab.

            Training tab:
            Uses the VGG16 neural network model to train on your provided data. 

            Testing tab: 
            Creates a covariance matrix on the performance of the network on 
            your data. If you would like, you could import a pre-trained 
            network with "x" validation accuracy to classify the cells within
            your dataset.  

            '''
        self.t.insert(END, intro)

        self.possible_hist = [1]

        # Designing the Make Histogram Tab
        self.make_hist_tab = self.make_window(self.ui_notebook, 'Determine Threshold')
        self.determine_thresh()

        # Designing Preprocessing Tab
        self.preprocess_tab = self.make_window(self.ui_notebook, 'Preprocessing')
        self.get_dir = Entry(self.preprocess_tab, font=30)
        self.get_dir.place(relwidth=0.65, relheight=0.1, relx=0.25)
        self.get_dir_button = Button(self.preprocess_tab, text="Get Image Path", font=40,
                                     command=lambda: self.run_preprocessing(self.get_dir.get(), self.preprocess_tab,
                                                                            self.get_thresh.get()))
        self.get_dir_button.place(relwidth=0.25, relheight=0.1)

        # Designing Training Tab
        self.training_tab = self.make_window(self.ui_notebook, 'Training')
        self.train_button = Button(self.training_tab, text="Train Model", font=40,
                                   command=lambda: self.run_training())
        self.train_button.place(relwidth=0.65, relheight=0.1, relx=0.25)

        # Designing the Testing Tab
        self.test = self.make_window(self.ui_notebook, 'Testing')
        self.entry = Entry(self.test, font=30)
        self.entry.place(relwidth=0.65, relheight=0.1, relx=0.25)
        self.entry_button = Button(self.test, text="Get Image Path", font=40,
                                   command=lambda: self.get_entry(self.entry.get()))
        self.entry_button.place(relwidth=0.25, relheight=0.1)

        # Designing the Master Page
        self.label = Label(master, text="Machine Learning Cell Death", font=('Times New Roman', 20))
        self.label.place(relx=0.05, rely=0.05)

        self.indexd = 0
        self.indexh = 0
        # self.hists, self.array, self.optimal_bins = PCA

        self.windows = []

        self.imgs = []

    def make_window(self, parent, tab_name):
        tab = Frame(parent, width=300, height=300)
        tab.pack(fill='both', expand=1)
        parent.add(tab, text=tab_name)
        return tab

    def run_training(self):
        from training import run_program
        run_program()

    def train_network(self):

        self.train_button.grid()
        # file = 'model_stats.png'
        # img = itk.PhotoImage(file=file)
        # Label(self.train_win, image=img).grid()

    def run_preprocessing(self, entry, win, stim='stim'):
        # exec(open('data_collection.py').read ())
        from data_collection import generate_training_data

        generate_training_data(entry, stimuli=stim, dictionary=stim_thresh)

        finish = Label(win, text="Preprocessing Complete", font=30)
        finish.place(anchor=NW)

    def pca(self):

        outputs_array_in = open('pi.array', 'rb')
        output_array = pickle.load(outputs_array_in)

        std_arr = []
        imgs = []

        for img in output_array:
            # Converts the 0 into NaN so that the Std will not be influenced
            arr = img.astype('float')
            arr[arr == 0] = None
            std_pi = np.nanstd(arr)
            std_arr.append(std_pi)
            imgs.append(img)


        q3, q1 = np.percentile(std_arr, [75, 25])
        iqr = q3 - q1
        h = 2 * iqr * (len(std_arr) ** (-1 / 3))
        optimal_bins = int((np.amax(std_arr) - np.amin(std_arr)) / h)

        return std_arr, optimal_bins, imgs

    def get_entry(self, entry):
        self.run(index, entry)
        self.label = Label(self.test, text="Healthy Cells: " + str(self.indexh), font=('Times New Roman', 20),
                           bg='#34ebcf')
        self.label.place(rely=0.25)
        self.labeld = Label(self.test, text="Dying Cells: " + str(self.indexd), font=('Times New Roman', 20),
                            bg='#34ebcf')
        self.labeld.place(rely=0.5)

    def dropdown_menu(self, page, options_list, option_title):

        clicked = StringVar()
        clicked.set(option_title)

        dir_entry = Entry(page, font=30)
        dir_label = Label(page, text="- Enter Image Directory", font=30)
        drop = OptionMenu(page, clicked, *options_list)
        submit = Button(page, text='Submit Stimuli', command=lambda: self.config_stimuli(clicked.get(), page, dir_entry.get()))


        return drop, submit, dir_entry, dir_label

    def make_2nd_frame(self, parent):
        win = Frame(parent, width=300, height=300)
        win.pack(fill='both', expand=1)

        return win

    def win_children(self, win):
        list = win.winfo_children()

        # Return list of widgets
        for item in list:
            if item.winfo_children():
                list.extend(item.winfo_children())

        # Forget Widgets
        for widget in list:
            widget.pack_forget()

    def config_stimuli(self, stim, win, dir):

        # Clears the widgets off of the screen
        self.win_children(win)

        # Runs the preprocessing for the stimuli
        # self.run_preprocessing(dir, win, stim=stim)

        # Places thresholding widgets onto the screen
        thresh = self.thresh_widgets(win, stim)

        # Saves the thresh and stim for future preprocessing:
        stim_thresh[stim] = thresh

        # self.run_preprocessing(dir, win, thresh=thresh, stim=stim)

    def increase_hist(self, stim):
        for factor in self.possible_hist:
            factor *= 1.5
            self.possible_hist[0] = factor
        file_name_arr, std_arr, imgs = self.make_hist(stim)
        self.show_hist(file_name_arr)

    def decrease_hist(self, stim):
        for factor in self.possible_hist:
            factor *= 0.5
            self.possible_hist[0] = factor
        file_name_arr, std_arr, imgs = self.make_hist(stim)

        self.show_hist(file_name_arr)

    def make_hist(self, stim):
        std_arr, optimal_bins, imgs = self.pca()
        file_name_arr = []
        for i in range(len(self.possible_hist)):
            optimal_bins_fin = int(self.possible_hist[i] * optimal_bins)
            plt.clf()
            plt.title(('PI standard deviation of {stims} stimulated nuclei images (2650 images)').format(
                stims=stim, fontsize=5))
            plt.xlabel(("PI Standard Deviation"))
            plt.ylabel("# of Images")
            plt.hist(std_arr, bins=optimal_bins_fin, range=[0, 120])
            file_name = "histogram " + "x.png"
            plt.savefig(file_name)
            file_name_arr.append(file_name)
        return file_name_arr, std_arr, imgs

    def show_hist(self, file_name_arr):
        for file in file_name_arr:
            img = itk.PhotoImage(file=file)
            hist = Label(self.make_hist_tab, image=img)
            hist.image = img
            hist.place(anchor=NW)

    def back(self, win):
        win.destroy()
        self.determine_thresh()

    def thresh_widgets(self, win, stim):

        file_name_arr, std_arr, imgs = self.make_hist(stim)
        self.show_hist(file_name_arr)

        increase_button = Button(win, text="Increase Bins", font=40,
                                 command=lambda: self.increase_hist(stim))
        decrease_button = Button(win, text="Decrease Bins", font=40,
                                 command=lambda: self.decrease_hist(stim))
        back_button = Button(win, text="Back", font=40,
                             command=lambda: self.back(win))
        get_thresh = Entry(self.make_hist_tab, font=15)
        thresh_button = Button(self.make_hist_tab, text="Thresh Value", font=5)

        buttons_y = 0.9
        get_thresh.place(rely=buttons_y, relx=0.3, relheight=0.05, relwidth=0.15)
        thresh_button.place(relx=0.45, rely=buttons_y, relheight=0.05, relwidth=0.20)
        increase_button.place(rely=buttons_y, relx=0.67)
        decrease_button.place(rely=buttons_y, relx=0.05)
        back_button.place(rely=buttons_y, relx=0.9)

        for img_index in range(len(std_arr)):
            if std_arr[img_index] > 40:
                plt.clf()
                plt.imshow(imgs[img_index])
                plt.show()
            else:
                plt.clf()
                plt.imshow(imgs[img_index])
                plt.show()

        return get_thresh.get()

    def determine_thresh(self):
        # Creates Second Frame
        win = self.make_2nd_frame(self.make_hist_tab)

        # Creates dropdown menu for stimuli
        options = ['staurosporin', 'nigericin', 'h2o2', 'blank']
        title = 'Stimuli'
        drop, submit, dir, dir_label = self.dropdown_menu(win, options, title)
        dir.pack()
        dir_label.place(in_=dir, relx=1, x=2, rely=0)
        drop.pack()
        submit.pack()

    def run(self, index, input):
        img_path = input
        for img in os.listdir(img_path):
            index += 1
            with open(os.path.join(img_path, img), 'r') as f:
                # img_array = f.read()
                data_array, images = self.create_array(f)
                self.predict_image(data_array)

    def create_array(self, imageFile):
        if type(imageFile) == str:
            img = image.load_img(imageFile, target_size=(28, 28))
        else:
            img = image.load_img(imageFile.name, target_size=(28, 28))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        images = np.vstack([img_array])
        return img_array, images

    def predict_image(self, inputArray):
        classes = np.argmax(model.predict(inputArray), axis=-1)
        prediction = (categories[classes[0]])

        if prediction == 'dyingcell':
            self.indexd += 1
        else:
            self.indexh += 1
        return self.indexd, self.indexh


root = Tk()
my_gui = GUI(root)
root.mainloop()