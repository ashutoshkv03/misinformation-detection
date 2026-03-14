# Misinformation Detection using Supervised Machine Learning

## Overview

This project implements a **supervised machine learning pipeline for detecting misinformation (rumors) in textual data**. The system processes social media text, performs feature engineering using TF-IDF, trains multiple classification models, and selects the best-performing model using cross-validated evaluation.

The goal of the project is to **automatically classify whether a given text is a rumor (misinformation) or a non-rumor (true information)**.

The implementation includes data preprocessing, feature extraction, model training, model selection, evaluation, and result analysis.

---

# Project Structure

```
misinfo_detection/
│
├── data/
│   ├── raw/
│   │   └── pheme_dataset.csv
│   └── processed/
│       └── cleaned_dataset.csv
│
├── models/
│   ├── best_model.pkl
│   └── model_metadata.json
│
├── notebooks/
│   └── misinformation_detection_pipeline.ipynb
│
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── confusion_matrix_values.txt
│   ├── metrics.json
│   ├── misclassified_examples.csv
│   └── top_features.txt
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── utils.py
│
├── reports/
│   └── project_report.md
│
├── requirements.txt
├── README.md
└── run_project.py
```

---

# Dataset

The dataset used in this project is derived from the **PHEME rumor dataset**, which contains social media posts labeled as either **rumor** or **non-rumor**.

### Data files

```
data/raw/pheme_dataset.csv
data/processed/cleaned_dataset.csv
```

The processed dataset contains the following columns:

| Column       | Description                                   |
| ------------ | --------------------------------------------- |
| text         | Original social media text                    |
| cleaned_text | Preprocessed text used for feature extraction |
| label        | Target variable (0 = non-rumor, 1 = rumor)    |

---

# Methodology

The project follows a typical supervised learning pipeline.

## 1. Data Preprocessing

Text data is cleaned using the following steps:

* lowercasing
* removal of URLs
* removal of punctuation
* stopword filtering
* whitespace normalization

The cleaning logic is implemented in:

```
src/utils.py
src/data_preprocessing.py
```

---

## 2. Feature Engineering

Text is converted into numerical features using **TF-IDF vectorization**.

The vectorizer captures term frequency and inverse document frequency to represent the importance of words across the corpus.

Implemented in:

```
src/feature_engineering.py
```

---

## 3. Model Training

Multiple supervised learning algorithms are trained and compared.

The models evaluated include:

* Logistic Regression
* Multinomial Naive Bayes
* Linear Support Vector Machine (SVM)

Each model is trained using a **scikit-learn pipeline** that integrates TF-IDF feature extraction with the classifier.

---

## 4. Model Selection

Model selection is performed using **GridSearchCV** with **5-fold cross-validation**.

The primary evaluation metric used for model selection is:

```
F1 Score
```

This metric balances **precision and recall**, making it suitable for misinformation detection tasks.

---

## 5. Model Evaluation

The selected best model is evaluated on a **held-out test dataset**.

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Additional evaluation outputs include:

* confusion matrix
* classification report
* misclassified examples
* top predictive features

---

# Results

All evaluation outputs are automatically generated and saved in:

```
results/
```

### Generated outputs

| File                        | Description                               |
| --------------------------- | ----------------------------------------- |
| metrics.json                | evaluation metrics                        |
| classification_report.txt   | detailed classification report            |
| confusion_matrix.png        | visualization of prediction performance   |
| confusion_matrix_values.txt | confusion matrix values                   |
| misclassified_examples.csv  | incorrectly classified examples           |
| top_features.txt            | most influential words for classification |

---

# Best Model

The best model selected through cross-validation is saved in:

```
models/best_model.pkl
```

Metadata describing the model and training configuration is stored in:

```
models/model_metadata.json
```

---

# Running the Project

## 1. Install dependencies

Create a virtual environment and install the required packages:

```
pip install -r requirements.txt
```

---

## 2. Run the notebook

Open and execute the notebook:

```
notebooks/misinformation_detection_pipeline.ipynb
```

Run all cells to reproduce the entire pipeline.

The notebook will automatically:

* train models
* perform model selection
* evaluate the best model
* save all results

---

# Technologies Used

* Python
* scikit-learn
* pandas
* numpy
* matplotlib
* Jupyter Notebook

---

# Key Features of the Project

* End-to-end machine learning pipeline
* text preprocessing and cleaning
* TF-IDF feature engineering
* comparison of multiple classifiers
* cross-validated model selection
* detailed performance evaluation
* reproducible experiment pipeline

---

# Future Improvements

Potential extensions for this project include:

* transformer-based models (BERT)
* ensemble learning methods
* real-time rumor detection APIs
* explainable AI methods (SHAP / LIME)
* deployment as a web application

---

# Author

Venkata Ashutosh Kande
Master's in Computer Science
Santa Clara University


