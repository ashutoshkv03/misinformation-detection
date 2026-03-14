# Misinformation Detection using Machine Learning

## Overview

Misinformation spreads rapidly on social media platforms and can significantly influence public opinion. This project implements an **end-to-end machine learning pipeline** that automatically detects misinformation (rumors) in textual data using supervised learning techniques.

The system processes social media text, converts it into numerical representations using **TF-IDF feature engineering**, trains multiple machine learning models, and selects the best-performing classifier using cross-validated model selection.

The final model predicts whether a given text is:

* **Rumor / Misinformation**
* **Non-Rumor / True Information**

This project demonstrates practical machine learning techniques such as **text preprocessing, feature engineering, model comparison, hyperparameter tuning, and evaluation analysis**.

---

# Key Features

* End-to-end **machine learning pipeline**
* Automated **text preprocessing**
* **TF-IDF feature engineering** for text representation
* Training and comparison of multiple classifiers:

  * Logistic Regression
  * Multinomial Naive Bayes
  * Linear Support Vector Machine (SVM)
* **GridSearchCV model selection**
* Performance evaluation using multiple metrics
* Misclassification analysis for model interpretation
* Automatic generation of experiment outputs

---

# Machine Learning Pipeline

```text
Raw Text Data
      ↓
Text Cleaning & Preprocessing
      ↓
TF-IDF Feature Extraction
      ↓
Train/Test Split
      ↓
Model Training
(Logistic Regression / Naive Bayes / SVM)
      ↓
Hyperparameter Tuning (GridSearchCV)
      ↓
Model Selection (Best F1 Score)
      ↓
Model Evaluation
      ↓
Save Best Model + Results
```

---

# Project Structure

```text
misinfo_detection/
│
├── data/
│   ├── raw/
│   │   └── pheme_dataset.csv
│   └── processed/
│       └── cleaned_dataset.csv
│
├── notebooks/
│   └── misinformation_detection_pipeline.ipynb
│
├── src/
│   ├── utils.py
│   └── feature_engineering.py
│
├── models/
│   ├── best_model.pkl
│   └── model_metadata.json
│
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── confusion_matrix_values.txt
│   ├── metrics.json
│   ├── misclassified_examples.csv
│   └── top_features.txt
│
├── reports/
│   └── project_report.md
│
├── README.md
└── requirements.txt
```

---

# Dataset

The dataset used in this project is derived from the **PHEME Rumor Dataset**, which contains social media posts labeled as either:

* **Rumor**
* **Non-Rumor**

Each example consists of textual content along with a binary label.

Example dataset structure:

| Column       | Description                                |
| ------------ | ------------------------------------------ |
| text         | Original social media text                 |
| cleaned_text | Preprocessed version used for training     |
| label        | Target variable (0 = non-rumor, 1 = rumor) |

---

# Models Used

This project compares three commonly used classifiers for text classification.

### Logistic Regression

A linear probabilistic classifier that predicts rumor probability using learned feature weights.

### Multinomial Naive Bayes

A probabilistic model based on **Bayes' theorem** that estimates class probabilities from word frequency distributions.

### Linear Support Vector Machine (SVM)

A margin-based classifier that finds the optimal decision boundary separating rumor and non-rumor texts.

---

# Model Selection

Model selection is performed using:

**GridSearchCV with cross-validation**

Evaluation metric used for model comparison:

**F1 Score**

F1 score is particularly useful for misinformation detection because it balances **precision and recall**.

---

# Results

The pipeline automatically generates evaluation outputs including:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Misclassified examples
* Top predictive features

All results are saved inside:

```text
results/
```

Example generated outputs:

| File                       | Description                                  |
| -------------------------- | -------------------------------------------- |
| metrics.json               | Model evaluation metrics                     |
| classification_report.txt  | Detailed classification report               |
| confusion_matrix.png       | Visualization of prediction performance      |
| misclassified_examples.csv | Incorrect predictions for analysis           |
| top_features.txt           | Most influential features for classification |

---

# Best Model

The best performing model is automatically selected and saved.

Saved files:

```text
models/best_model.pkl
models/model_metadata.json
```

The metadata file stores:

* selected model name
* best hyperparameters
* cross-validation score
* comparison results of all models

---

# How to Run the Project

### 1. Clone the repository

```bash
git clone <repository_url>
cd misinfo_detection
```

---

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the notebook

Open and run:

```text
notebooks/misinformation_detection_pipeline.ipynb
```

Execute all cells to reproduce the entire pipeline.

The notebook will automatically:

* train models
* select the best classifier
* evaluate performance
* generate all results

---

# Technologies Used

* Python
* scikit-learn
* pandas
* numpy
* matplotlib
* Jupyter Notebook

---

# Future Improvements

Potential extensions for this project include:

* Transformer-based models (BERT)
* Real-time misinformation detection API
* Explainable AI methods (SHAP / LIME)
* Social network propagation analysis
* Deployment as a web application

---

# Author

Venkata Ashutosh Kande
Master's in Computer Science
Santa Clara University
