# ML.Fall2025.EEG.Prj.Hosseinzadeh.Vaziri
پروژه درس یادگیری ماشین پاییز 1404


گروه G45

محمد حسین حسین زاده 403203557

فرشاد وزیری - 403206179


# 🧠 Motor Imagery EEG Classification Pipeline

### CSP + SVM/KNN/LDA/RF on BCI Competition IV Dataset 1

این پروژه یک پایپلاین کامل برای **طبقه‌بندی سیگنال EEG مربوط به Motor
Imagery** (تصور حرکت دست چپ/راست) ارائه می‌دهد.\


پایپلاین شامل مراحل زیر است:

-   بارگذاری و پیش‌پردازش سیگنال EEG\
-   فیلترگذاری باندپسی (۸--۳۰ Hz)\
-   استخراج ویژگی با **Common Spatial Patterns (CSP)**\
-   آموزش مدل‌های طبقه‌بندی\
-   ارزیابی با ROC، AUC، Accuracy، Precision، Recall، F1\
-   اجرای پایپلاین روی هر دو فایل\
-   نمایش نتایج در قالب جدول مقایسه‌ای

------------------------------------------------------------------------

## 📂 ساختار پروژه

``` text
├── MotorImagery_Classification_Pipeline.ipynb
│   
└── README.md
```

------------------------------------------------------------------------

## ⚙️ نصب و اجرای پروژه

### 1️⃣ نصب وابستگی‌ها

``` bash
pip install numpy scipy scikit-learn matplotlib
```

### 2️⃣ اجرای پایپلاین روی هر فایل

``` python
out_a = run_pipeline("BCICIV_calib_ds1a.mat")
out_c = run_pipeline("BCICIV_calib_ds1c.mat")
```

### 3️⃣ رسم ROC

``` python
plot_roc(out_a["results"], "ROC – ds1a")
plot_roc(out_c["results"], "ROC – ds1c")
```

### 4️⃣ جدول مقایسه‌ای نتایج

``` python
df_results
```

------------------------------------------------------------------------

## 🧩 توضیح مراحل

### 🎛 1. فیلترگذاری باندپسی (۸--۳۰ Hz)

X_filtered(t) = Bandpass(X(t), 8--30 Hz)

### 🧠 2. استخراج ویژگی با CSP

W = argmax(W\^T C1 W / W\^T C2 W)

ویژگی‌ها:

fi = log(var(Wi\^T X) / Σ var(Wj\^T X))

### 🤖 3. مدل‌های طبقه‌بندی

#### 🔹 SVM-RBF

K(xi, xj) = exp(-γ\|\|xi - xj\|\|²)

#### 🔹 LDA

y = w\^T x + b

#### 🔹 KNN

-   فاصله: Euclidean یا Cosine\
-   تعداد همسایه‌ها با Cross-Validation تنظیم می‌شود

#### 🔹 Random Forest

-   چندین درخت تصمیم + Bagging\
-   پارامترهای مهم: تعداد درخت‌ها، اندازهٔ برگ‌ها

------------------------------------------------------------------------

## 📊 نتایج نمونه (Pipeline)

  Model           Acc_ds1a   Acc_ds1c   MeanAcc
  --------------- ---------- ---------- ---------
  SVM-RBF         0.76       0.50       0.63
  LDA             0.76       0.52       0.64
  KNN             0.76       0.50       0.63
  Random Forest   0.74       0.50       0.62



