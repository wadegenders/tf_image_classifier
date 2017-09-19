# tf_image_classifier
Simple, out-of-the-box multiclass image classification in Tensorflow. 

#Notes
Images must be organized into class folders as subdirectories in a 'data_dir' folder.
```
data_dir/
data_dir/image_class_1
data_dir/image_class_2
.
.
.
```
#Running the code
To train, run:
```
python main.py --train --data /PATH/TO/data_dir
```

#Dependencies
Python 3.5.x
Tensorflow 1.2.x
Pillow 4.2
Numpy 1.13
