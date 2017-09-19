from DataSet import DataSet
import multiprocessing
from random import shuffle
from PIL import Image
import numpy as np
from timeit import default_timer as timer

class Trainer():
    def __init__(self, dataset, flatten, test_p, even_class_n):
        self.dataset = dataset
        self.normalizer = 255.0

        ###all class filenames in a dict, key is class name, value is an array holding tuples of fname and one hot correct label
        self.class_fnames = self.dataset.get_class_fnames()

        n_class_samples = [len(self.class_fnames[c]) for c in self.class_fnames]
        print("CLASS SAMPLES")
        print(n_class_samples)

        ###shuffle the order of the class fnames
        for c in self.class_fnames:
           shuffle(self.class_fnames[c])

        if test_p < 0.0 or test_p > 1.0:
            test_p = 0.15

        ###determine the number of test data points given the desired test proportion
        self.train_data = []
        self.test_data = []
        if even_class_n is True:
            min_samples = min(n_class_samples)
            #min_samples = 800
            print("MIN SAMPLES "+str(min_samples))
            self.test_n = int(min_samples*test_p)
            print("TEST N per class "+str(self.test_n))
            self.train_n = min_samples - self.test_n
            print("TRAIN N per class "+str(self.train_n))
            ###get minimum number of data points from each class, seperate into test and train sets
            for c in self.class_fnames:
                self.train_data.extend(self.class_fnames[c][:self.train_n])    
                self.test_data.extend(self.class_fnames[c][self.train_n:self.test_n+self.train_n])

        shuffle(self.train_data)
        shuffle(self.test_data)
        ###set function whether to flatten image for traditional feedforward or keep shape for convolutional
        if flatten is True:
            self.get_img = self.get_img_data_flatten
        else:
            self.get_img = self.get_img_data
        #if class_samples.count(class_samples[0]) == len(class_samples)
        print("TOTAL TEST SAMPLES "+str(len(self.test_data)))
        print("TOTAL TRAIN SAMPLES "+str(len(self.train_data)))
        self.j = 0

    def get_img_data(self, fname):
        return np.expand_dims(np.array(Image.open(fname))/self.normalizer, 0)

    def get_img_data_flatten(self, fname):
        return np.expand_dims(np.array(Image.open(fname)).flatten()/self.normalizer, 0)

    def get_train_batch(self, batch_size):
        if self.j+batch_size < len(self.train_data):                                                                                 
            batch = self.train_data[self.j:self.j+batch_size] 
            self.j += batch_size
            epoch_over = False
        else: 
            ###at end of train data, use up last of data, reset iterator, and shuffle train data
            batch = self.train_data[self.j:]  
            self.j = 0
            epoch_over = True
            shuffle(self.train_data)
            #print("test_over")
        train_batch, one_hots = self.fnames_to_batch(batch)
        return train_batch, one_hots, epoch_over 

    def get_test_batch(self, batch_size):
        if self.j+batch_size < len(self.test_data):                                                                                 
            batch = self.test_data[self.j:self.j+batch_size] 
            self.j += batch_size
            epoch_over = False
        else: 
            ###at end of train data, use up last of data, reset iterator, 
            batch = self.test_data[self.j:]  
            self.j = 0
            epoch_over = True
        train_batch, one_hots = self.fnames_to_batch(batch)
        return train_batch, one_hots, epoch_over 

    def fnames_to_batch(self, fnames_and_one_hot):
        batch = []
        one_hots = []
        
        #get image data using PILLOW 
        for f in fnames_and_one_hot:
            batch.append(self.get_img(f[0]))
            one_hots.append(f[1])

        batch = np.concatenate(batch, 0)
        one_hots = np.concatenate(one_hots, 0)
        return batch, one_hots

    def get_batch_and_fnames(self, batch_size):
        if self.j+batch_size < len(self.test_data):                                                                                 
            batch = self.test_data[self.j:self.j+batch_size] 
            self.j += batch_size
            epoch_over = False
        else: 
            ###at end of train data, use up last of data, reset iterator, 
            batch = self.test_data[self.j:]  
            self.j = 0
            epoch_over = True
        train_batch, one_hots = self.fnames_to_batch(batch)
        #return image data and fnames
        return train_batch, [b[0] for b in batch], epoch_over 
