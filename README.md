# Machine Learning course project, Fall 2025, EEG Project, Hosseinzadeh, Vaziri, 

Group G45

Mohammad Hossein Hosseinzadeh - 403203557

Farshad Vaziri - 403206179


# 🧠 Motor Imagery EEG Classification Pipeline

### CSP + SVM/KNN/LDA/RF on "BCI Competition IV Dataset 1"

``` text

This project provides a complete pipeline for classification of EEG signals related to Motor
Imagery (imagined left/right hand movement).

The pipeline includes the following steps:

    Loading and preprocessing EEG signals

    Band-pass filtering (8--30 Hz)

    Feature extraction using Common Spatial Patterns (CSP)

    Training classification models

    Evaluation with ROC, AUC, Accuracy, Precision, Recall, F1

    Running the pipeline on both files

    Displaying the results in a comparative table

  
```

------------------------------------------------------------------------

## 📂 Project structure

``` text
├── MotorImagery_Classification_Pipeline.ipynb
│
├── BCICIV_calib_ds1a.mat
├── BCICIV_calib_ds1c.mat
│
└── README.md
```


------------------------------------------------------------------------

## 🧩 Steps description

### 🎛 1. Band-pass filtering (8--30 Hz)

### 🧠 2. Feature extraction with CSP

### 🤖 3.Classification models

#### 🔹 SVM-RBF
#### 🔹 LDA
#### 🔹 KNN
#### 🔹 Random Forest

------------------------------------------------------------------------

## 📊 Sample compare results for 2 datasets


| Model          | Accuracy_ds1a | Accuracy_ds1c | MeanAccuracy |
|----------------|----------|----------|---------|
| SVM-RBF        | 0.92     | 0.82     | 0.87    |
| LDA            | 0.90     | 0.84     | 0.87    |
| KNN            | 0.96     | 0.80     | 0.88    |
| Random Forest  | 0.90     | 0.86     | 0.88    |


