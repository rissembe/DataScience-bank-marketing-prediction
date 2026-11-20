# Bank Marketing Prediction

**Binary Classification using Logistic Regression and Decision Tree**

## 1. Project Overview

This project uses the **Bank Marketing** dataset from the UCI Machine Learning Repository to predict whether a customer will subscribe to a bank term deposit (`yes` or `no`).

The objective is to compare two supervised machine learning models:
- Logistic Regression
- Decision Tree

**Research question:** Can we predict whether a bank customer will subscribe to a term deposit using information about the customer and previous marketing contacts?

## 2. Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

The project uses the smaller `bank.csv` dataset, containing **4,521 observations and 17 columns** (including the target variable `y`). It includes customer information, banking history, and marketing campaign data.

The target variable is `y`:
- `yes` (1): the customer subscribed to a term deposit.
- `no` (0): the customer did not subscribe.

The Python script automatically downloads `bank.csv` from UCI if the file is not already present in the script's folder.

## 3. Methodology

The analysis is divided into five parts:

1. **Dataset:** import the data and inspect its structure, missing values, and descriptive statistics.
2. **Descriptive analysis:** examine subscription rates, age, job, education, and correlations among numerical variables.
3. **Data preparation:** encode the target as 0/1, remove `duration`, and split the data into training (80%) and test (20%) sets using stratified sampling.
4. **Model estimation:** train Logistic Regression and Decision Tree classifiers. Numerical variables are standardized and categorical variables are one-hot encoded within scikit-learn pipelines.
5. **Results comparison:** evaluate both models using Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices, and ROC curves.

The variable `duration` is excluded from model training because call duration is not available before a call takes place. The dataset also contains `unknown` categorical values, which are retained as categories in this implementation.

## 4. Tools and Requirements

The project was developed in **Python 3** using **Spyder**. It requires the following packages:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

These dependencies are also listed in `requirements.txt`. The standard-library modules used by the script do not need separate installation.

## 5. Installation and Execution

Clone or download this repository. From the project folder, install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

Alternatively, if you do not have the `requirements.txt` file, install the packages directly:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

Open `Bank_Marketing_Project_EN.py` in Spyder (or run it with Python):

```bash
python Bank_Marketing_Project_EN.py
```

An internet connection is required on the first run if `bank.csv` is not already available next to the script.

## 6. Outputs

The script prints dataset summaries, classification reports, and a table comparing the two models. It also displays:

- Term deposit subscription distribution
- Customer age distribution
- Subscription rates by job and education
- Correlation matrix
- Model performance comparison
- Confusion matrices for both models
- ROC curves
- First levels of the decision tree

The `yes` and `no` classes are imbalanced, so **Accuracy alone is not sufficient**. Recall and F1-score for the `yes` class are important when comparing the models.

## 7. Limitations

This is an initial comparison based on one 80/20 train-test split. The project does not yet include cross-validation or systematic hyperparameter tuning. Results should therefore be interpreted as performance on this test split, not as guaranteed performance on future customers.

## 8. References

- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---

**Academic project — Data Science**
