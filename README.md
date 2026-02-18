# ML.Fall2025.EEG.Prj, Hosseinzadeh, Vaziri, Machine Learning course project, Fall 2025

Group G45

Mohammad Hossein Hosseinzadeh - 403203557

Farshad Vaziri - 403206179


# 🧠 Motor Imagery EEG Classification Pipeline

### CSP + SVM/KNN/LDA/RF on BCI Competition IV Dataset 1

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

## ⚙️Installation and running the project

### 1️⃣ Installing dependencies

``` bash
pip install numpy scipy scikit-learn matplotlib
```

### 2️⃣ Running the pipeline on each file

``` python
out_a = run_pipeline("BCICIV_calib_ds1a.mat")
out_c = run_pipeline("BCICIV_calib_ds1c.mat")
```

### 3️⃣ Plotting ROCROC

``` python
plot_roc(out_a["results"], "ROC – ds1a")
plot_roc(out_c["results"], "ROC – ds1c")
```

### 4️⃣Comparative results table

``` python
df_results
```

------------------------------------------------------------------------

## 🧩 Steps description

### 🎛 11. Band-pass filtering (8--30 Hz)

X_filtered(t) = Bandpass(X(t), 8--30 Hz)

### 🧠 22. Feature extraction with CSP

W = argmax(W^T C1 W / W^T C2 W)

Features:

fi = log(var(Wi^T X) / Σ var(Wj^T X))

### 🤖 3.Classification models

#### 🔹 SVM-RBF

K(xi, xj) = exp(-γ\|\|xi - xj\|\|²)

#### 🔹 LDA

y = w\^T x + b

#### 🔹 KNN

-  Distance: Euclidean or Cosine\
-  Number of neighbors is tuned using Cross-Validation

#### 🔹 Random Forest

-   Multiple decision trees + Bagging
-   Important parameters: number of trees, leaf size

------------------------------------------------------------------------

## 📊 Sample results (Pipeline)


| Model          | Accuracy_ds1a | Accuracy_ds1c | MeanAccuracy |
|----------------|----------|----------|---------|
| SVM-RBF        | 0.76     | 0.84     | 0.80    |
| LDA            | 0.76     | 0.92     | 0.84    |
| KNN            | 0.76     | 0.82     | 0.79    |
| Random Forest  | 0.74     | 0.82     | 0.78    |


