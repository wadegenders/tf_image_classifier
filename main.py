import os
import argparse
from timeit import default_timer as timer
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from DataSet import DataSet
from Trainer import Trainer

class ConvNet():
    def __init__(self, img_w, img_h, channels, n_classes, lr):
        self.lr = lr
        ###input data and label placeholders
        self.inputs = tf.placeholder(shape=[None, img_w, img_h, channels], dtype=tf.float32, name='inputs')
        self.one_hot_labels = tf.placeholder(shape=[None, n_classes], dtype=tf.float32, name='one_hot_labels')
        ###graph model
        self.conv1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.inputs, num_outputs=16, kernel_size=[5,5], padding='VALID')
        self.pool1 = slim.max_pool2d(inputs=self.conv1, kernel_size=[2, 2], padding='VALID' )
        self.conv2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.pool1, num_outputs=32, kernel_size=[3,3], padding='VALID')
        self.pool2 = slim.max_pool2d(inputs=self.conv2, kernel_size=[2, 2], padding='VALID')
        self.flatten = slim.flatten(self.pool2)
        self.dense = slim.fully_connected(activation_fn=tf.nn.relu, inputs=self.flatten, num_outputs=128)
        self.output = slim.fully_connected(activation_fn=None, inputs=self.dense, num_outputs=n_classes)
        self.outputs = tf.cast(self.output, dtype=tf.float64, name="outputs")
        self.cross_entropy_logit = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.one_hot_labels)
        ###define cross entropy loss
        self.loss = tf.reduce_mean(self.cross_entropy_logit)
        ###define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

class FFNet():
    def __init__(self, input_size, n_classes, lr):
        self.lr = lr
        ###input data and label placeholders
        self.inputs = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
        self.one_hot_labels = tf.placeholder(shape=[None, n_classes], dtype=tf.float32)
        ###graph model
        self.dense1 = slim.fully_connected(activation_fn=tf.nn.relu, inputs=self.inputs, num_outputs=128)
        self.dense2 = slim.fully_connected(activation_fn=tf.nn.relu, inputs=self.dense1, num_outputs=128)
        self.output = slim.fully_connected(activation_fn=None, inputs=self.dense2, num_outputs=n_classes)
        self.cross_entropy_logit = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.one_hot_labels)
        ###define cross entropy loss
        self.loss = tf.reduce_mean(self.cross_entropy_logit)
        ###define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

def write_to_tensorboard(label, data, t, summary_writer): 
    ###add stats to tensorboard                                   
    summary = tf.Summary()
    summary.value.add(tag=label, simple_value=float(data))
    summary_writer.add_summary(summary, t)
    summary_writer.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', help='load already saved model, argument should be folder path where model is stored')
    parser.add_argument('--train', action='store_true', help='train on labeled data, argument should be folder path with contains folders, each folder contains all images of same category')
    #parser.add_argument('--test', action='store_true', help='test model on images, argument should be folder path containing images to be labeled by model')
    parser.add_argument('--data', default='./data', help='filepath containing data, either to be trained or tested on')
    parser.add_argument('--epochs', default=5, help='number of training epochs')
    parser.add_argument('--batchsize', default=48, help='number of data samples per batch')
    parser.add_argument('--flatten', action='store_true', help='flatten data to be 1-dimensional')
    parser.add_argument('--testp', default=0.15, help='proportion of data to be excluded during training to test as validation')
    args = parser.parse_args()

    classes_folder = args.data
    data = DataSet(classes_folder)
    flatten_image = True if args.flatten else False
    even_class_p = True
    trainer = Trainer(data, flatten_image, args.testp, even_class_p)
    net = ConvNet(330, 788, 3, 2, 0.0001)
    tb_dir = "tensorboard_train"
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    #tensorboard stats
    if args.train:
        ###train loop variables
        ###tensorboard stuff
        summary_writer = tf.summary.FileWriter(tb_dir)
        epochs = 0
        total_epochs = args.epochs
        batchs = 0
        total_batchs = 0
        epoch_over = False
        ###using a bigger batch size on GPU will speed up training, bounded by GPU memory
        train_batch_size = args.batchsize
        test_batch_size = 64
        ###start TF session
        with tf.Session() as sess:
            saver = tf.train.Saver()
            # Required to get the filename matching to run.
            sess.run(tf.global_variables_initializer())
            ###loop for specified num of epochs
            while epochs < total_epochs:
                start = timer()
                print("START TRAINING EPOCH "+str(epochs))
                ###train on images until no more images left (ie use entire dataset once)
                while not epoch_over:
                    train_batch, one_hots, epoch_over = trainer.get_train_batch(train_batch_size)
                    _, loss, c_ent = sess.run([net.optimizer, net.loss, net.cross_entropy_logit], feed_dict={net.inputs:train_batch, net.one_hot_labels:one_hots})
                    write_to_tensorboard("Loss/Loss", loss, batchs+total_batchs, summary_writer)
                    batchs += 1
                    if batchs % 15 == 0:
                        print("TRAIN IMAGE #"+str(train_batch_size*batchs))
                    if epoch_over is True:
                        print("END EPOCH "+str(epochs)) 
                end = timer()
                total_batchs += batchs
                print("TRAIN TIME ELAPSED "+str(end-start))

                ###test on unseen data, init variables
                print("BEGIN TESTING")
                start = timer()
                batchs = 0 
                epoch_over = False
                accuracy = []
                ###begin testing/validation loop, use entire validation dataset
                while not epoch_over:
                    test_batch, one_hots, epoch_over = trainer.get_test_batch(test_batch_size)
                    #get predicted class of test data from model 
                    prediction = sess.run([net.output], feed_dict={net.inputs:test_batch})[0]
                    accuracy.append(np.mean(np.equal(np.argmax(prediction, axis=1), np.argmax(one_hots, axis=1)).astype("float32")))
                    batchs += 1
                    if batchs % 5 == 0:
                        print("TEST IMAGE #"+str(test_batch_size*batchs))
                    if epoch_over is True:
                        print("TEST EPOCH OVER")
                epoch_over = False
                batchs = 0
                test_accuracy = np.mean(accuracy)
                print("TEST ACCURACY "+str(test_accuracy))
                write_to_tensorboard("Accuracy/Test Data", test_accuracy, epochs, summary_writer)
                end = timer()
                print("TEST TIME ELAPSED "+str(end-start))
                #save model after each training epoch
                saver.save(sess, model_dir+'/model', global_step=epochs)
                epochs += 1
        print("FINISHED")
