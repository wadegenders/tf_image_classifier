# tf_image_classifier
Simple, out-of-the-box multiclass Tensorflow image classifier. 

# Notes
Images must be organized into class folders as subdirectories in a 'data_dir' folder.
```
data_dir/
data_dir/image_class_1
data_dir/image_class_2
.
.
.
data_dir/image_class_N
```

`image_class_1` should be a folder containing images of only images in class 1.

# Running the code
To train, run:
```
python main.py --train --data /PATH/TO/data_dir --w 100 --h 100
```
Modify `--w` and `--h` to desired values for resizing images width and height

To view training loss and test accuracy, run Tensorboard:

```
tensorboard --logdir='./tensorboard_train'
```

# Dependencies
* Python 3.5.x
* Tensorflow 1.2.x
* Pillow 4.2
* Numpy 1.13
