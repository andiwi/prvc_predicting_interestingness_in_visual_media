# Application for predicting interestingness in visual media
This application should be submitted to the http://www.multimediaeval.org/mediaeval2017/
Implemented during the 'Praktikum aus Visual Computing' at the TU Wien.

#How to deploy
As package, dependency and environment management tool, I use conda.
You need miniconda to install the application [https://conda.io/miniconda.html](https://conda.io/miniconda.html)

To deploy: 
```python 
$ conda env create
$ source activate prvc (for linux, OSX)
or
$ activate prcv (for windows)
```

To start the application:
 
```python
python main.py
```

in the main.py you have 5 parameters to set
* ```runname``` ... 1,2,3
* ```do_preprocessing``` ...True or False; set to True only at your first run on the dataset
* ```calc_features``` ...True or False; calculates the selected features (skips calculation if file for image and feature already exists)
* ```use_second_dev_classification_method``` ...True or False; classifies with second order deviation method


To evaluate the results trec_eval tool is used. Instructions can be found at https://github.com/multimediaeval/2017-Predicting-Media-Interestingness-Task/wiki/Evaluation


## Chainer CV install problem
To install ChainerCV you may have to do the following steps on windows:

1, Download Visual C++ for Python from https://www.microsoft.com/download/details.aspx?id=44266
2, My installation path is 'C:\Users\username\AppData\Local\Programs\Common\Microsoft\Visual C++ for Python\9.0\VC\include'
3, Copy stdint.h to installation path. (download it from: https://github.com/mattn/gntp-send/blob/master/include/msinttypes/stdint.h)