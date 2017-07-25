# Application for predicting interestingness in visual media
This application should be submitted to the http://www.multimediaeval.org/mediaeval2017/
Implemented during the 'Praktikum aus Visual Computing' at the TU Wien by Andreas Wittmann (e1225854@student.tuwien.ac.at)

#How to deploy
I use conda as package, dependency and environment management tool.
You can use miniconda to install the application [https://conda.io/miniconda.html](https://conda.io/miniconda.html)

Install instructions: 
```python 
$ conda env create
$ source activate prvc (for linux, OSX)
or
$ activate prcv (for windows)
```

## Chainer CV install problem
To install ChainerCV you may have to do the following steps on windows:

1. Download Visual C++ for Python from https://www.microsoft.com/download/details.aspx?id=44266
2. My installation path is 'C:\Users\username\AppData\Local\Programs\Common\Microsoft\Visual C++ for Python\9.0\VC\include'
3. Copy stdint.h to installation path. (download it from: https://github.com/mattn/gntp-send/blob/master/include/msinttypes/stdint.h)

#Run the application
To start the application:
 
```python
python main.py
```

in the main.py you have 5 parameters to set
* ```feature_names``` ...select (uncomment) the feature names which should be used
* ```runname``` ... 1,2,3
* ```do_preprocessing``` ...True or False; set to True only at your first run on the dataset (crops images)
* ```calc_features``` ...True or False; calculates the selected features (skips calculation if file for image and feature already exists)
* ```use_second_dev_classification_method``` ...True or False; classifies with second order deviation method

features are stored in a subdirectory './devset|testset/Features_From_TUWien'

A result file (ready for submission) is stored in the root directory.

## Evaluation
The trec_eval tool is used for evaluation.
Detailed instructions can be found at https://github.com/multimediaeval/2017-Predicting-Media-Interestingness-Task/wiki/Evaluation

Convert result file for the trec_eval tool:
```python
python results_to_trec.py me17in_groupname_image_runname.txt
```

Run evaluation (in trec_eval directory):
```python
./trec_eval -M10 ./data/testset-image.qrels ./data/me17in_groupname_image_runname.txt.trec
```

