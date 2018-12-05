# facial-landmark-dnn
Facial landmark detection using Convolutional Neural Networks for Mobile Device

### [ Data Preprocessing ]
- Raw-Data
    1. [300W](https://ibug.doc.ic.ac.uk/resources/300-W/)
    2. [300VW](https://ibug.doc.ic.ac.uk/resources/300-VW/)
    3. [Ibug](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
    4. Our own Korean face dataset
    
- Dataset in Google Drive: [Image Dataset](https://drive.google.com/file/d/1I-azq5nKd-8YOLiKz2xVxJMXhvjhtW6s/view?usp=sharing)
[Landmarks Dataset](https://drive.google.com/file/d/1J7MC48fQtB_AVSFo4gefqfApw3dsWBak/view?usp=sharing)
```python
# Example: Load Dataset
X = np.load(PATH + "basic_dataset_img.npz")
y = np.load(PATH + "basic_dataset_pts.npz")
X = X['arr_0']
y = y['arr_0'].reshape(-1, 136)
```

### [ Modeling ]
- [VGG_like_model](./modeling/train_basic_models.py) 
- [Mobilenet_based_model](./modeling/train_mobilenets.py)

### [ Result ]
- Face detector: [opencv ssd detector](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
- Facial Landmark detector: Mobilenet based model

[Youtube Video 1](https://www.youtube.com/watch?v=0CwBjoHU5os)
[Youtube Video 2](https://www.youtube.com/watch?v=ovMdKCaAuMc)

### [ Converter for Mobile ]
- [IOS - Keras to CoreML](./testing/convert.py)

- Android (TODO)
    1. Import model in Tensorflow
        - Convert Keras to Tensorflow [use this code](https://github.com/amir-abdi/keras_to_tensorflow/blob/master/keras_to_tensorflow.ipynb)
        - Build Android app and call tensflow. check [this tutorial](https://omid.al/posts/2017-02-20-Tutorial-Build-Your-First-Tensorflow-Android-App.html) and [this official demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android) from google to learn how to do it.
    2. Import model in Java
        - deeplearning4j a java library allow to import keras model: [tutorial link](https://deeplearning4j.org/model-import-keras)
        - Use deeplearning4j in Android: it is easy since you are in java world. check [this tutorial](https://deeplearning4j.org/android)

### [ Reference ]
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [Yinguobing github](https://github.com/yinguobing)




