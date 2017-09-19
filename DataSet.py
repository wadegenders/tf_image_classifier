import os
import numpy as np

class DataSet():
    def __init__(self, classes_folder_path):
        #classes_folder_path is contains folders labeled as the categories and containing images
        self.classes_folder_path = classes_folder_path
        self.classes_list = os.listdir(self.classes_folder_path)
        #enumerate classes with an integer from 0-(n-1) for n classes, used for one-hot vectors
        self.classes = {c:i for c, i in zip(self.classes_list, range(len(self.classes_list)))}
        self.n_classes = len(self.classes)

    def get_class_img_fnames(self, c):
        class_folder = self.classes_folder_path+"/"+c+"/"
        return [class_folder+f for f in os.listdir(self.classes_folder_path+"/"+c)]

    def get_class_fnames(self):
        class_fnames = {c:[] for c in self.classes}
        one_hot_template = np.zeros(self.n_classes)
        for c in self.classes:
            print("FOUND CLASS FOLDER "+str(c))
            one_hot = np.copy(one_hot_template)
            one_hot[self.classes[c]] = 1
            for f in self.get_class_img_fnames(c):
                class_fnames[c].append((f, np.expand_dims(one_hot, 0)))
        return class_fnames

    def get_classes_list(self):
        return self.classes_list
