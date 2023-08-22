# Breast Cancer Prediction

Breast cancer is a disease that causes a high mortality rate in women. This disease is caused by breast tissue cells that grow abnormally and continuously. Family history of breast cancer, menopausal age, age of first child pregnancy, nulliparous parity, breastfeeding history, obesity, and activity are some of the high risk factors that can cause breast tumor cancer (Hero 2021).

code files:

- breastcancer_predict.ipynb (used imagedatagenerator)
- using_tfdata_pipeline.ipynb (used tfdata pipeline)
- transfer_learning.ipynb (used efficientnetV2B1 transfer learning, for final result)

# Project Goals

This project aims to built a image detection tools based on Convolutional Neural Network model architecture to classify benign and malignant breast tumors from histopathologic breast cancer images.

# About the Data

The dataset used for this projects has been taken from BreakHis breast cancer histopathology image dataset derived from the collaboration of The Laboratory of Vision, Robotics and Imaging, Department of Informatics of the Federal University of Parana with P&D Laboratory Pathological Anatomy and Cytopathology, Parana, Brazil.

source: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

The BreakHis dataset consists of 9,109 microscopic images of breast cancer collected from 82 patients using four magnifying factors of 40x, 100x, 200x, and 400x. These images are divided into two sample classes of 2,480 benign and 5,429 malignant which are 700 x 460 pixels in size with PNG format.

# Data Preprocessing

1. Load images

Dataset can be accesses through fold.csv dataframe that contains images information that consist of magnitude, group (train, test), and path, with total of 7909 images of benign and malignant breast cancer.

2. Data Splitting

Through fold.csv dataframe, the data is divided into three parts. First, data is divided to training set by 5506 and test set by 2403 based on 'grp' value. Then training set is divided randomly to train set by 4404 and validation set by 1102.

3. Data Augmentation

Applies a set of image augmentation transformations using the Albumentations library. These transformations include:

- Rotation,
- Horizontal and vertical flips,
- Random brightness and contrast adjustments, and
- Blurring

# Transfer Learning

The architecture that will be used is one of the pretrained models, EfficientNetv2B1 with freezed weights. And then add code below to rebuild the top:

```
inputs = layers.Input(shape=IMAGE_SHAPE)
x = Flatten()(pretrained_model.output)
x = Dense(32, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(pretrained_model.input, outputs)
```

# Result

- Loss and ROC AUC plotting over epochs during the training of a CNN model using EfficientNetv2B1 transfer learning.

  ![alt text](https://github.com/anggapark/breast-cancer-predict/blob/main/asset/transfer_lr_model_result.png?raw=true)

- Confusion Matrix and ROC (Receiver Operating Characteristic ) Curve

  ![alt text](https://github.com/anggapark/breast-cancer-predict/blob/main/asset/cm_roc_curve.png?raw=true)

# Deployment

Breast Cancer Classifier Deployment:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://breast-cancer-predict-deploy.streamlit.app)
