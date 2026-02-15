# Adult Income Classification using Machine Learning Models

## Problem Statement
The objective of this project is to predict whether an individual earns more than $50K per year based on demographic and employment-related attributes. Multiple machine learning classification models are implemented, evaluated, and compared. An interactive Streamlit web application is developed to demonstrate predictions and performance metrics.

---

## Dataset Description

The Adult Income dataset is a well-known real-world classification dataset used for income prediction.

- Total Instances: ~48,000
- Number of Features: 14
- Target Variable: income
  - <=50K ( equal or greater than 50k )
  - '>' 50K ( less than 50k )
- Feature Types:
  - Numerical: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country

### Data Preprocessing Steps
- Missing values marked as '?' were removed
- Categorical features were converted using Label Encoding
- Features were scaled using StandardScaler
- Dataset was split into training and testing sets

---

## Machine Learning Models Used

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Ensemble Model – Random Forest  
6. Ensemble Model – XGBoost  

---

## Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8259 | 0.8485 | 0.7121 | 0.4698 | 0.5661 | 0.4786 |
| Decision Tree Classifier | 0.8084 | 0.7438 | 0.6005 | 0.6187 | 0.6095 | 0.4826 |
| K-Nearest Neighbor Classifier | 0.8216 | 0.8477 | 0.6447 | 0.5837 | 0.6127 | 0.4982 |
| Naive Bayes Classifier | 0.8013 | 0.8493 | 0.6711 | 0.3484 | 0.4587 | 0.3799 |
| Ensemble Model - Random Forest | 0.8579 | 0.9029 | 0.7449 | 0.6269 | 0.6808 | 0.5940 |
| Ensemble Model - XGBoost | 0.8676 | 0.9214 | 0.7626 | 0.6564 | 0.7055 | 0.6236 |

---

## Observations on Model Performance

### Logistic Regression
Logistic Regression served as a strong baseline model with good accuracy and AUC. However, the recall was relatively lower, indicating that some high-income cases were missed.

### Decision Tree Classifier
The Decision Tree achieved good recall and was able to capture many positive cases. However, its lower AUC suggests overfitting and weaker generalization.

### K-Nearest Neighbor Classifier
KNN showed balanced performance across all metrics and benefited from feature scaling. It performed moderately well but was not the best performer.

### Naive Bayes Classifier
Naive Bayes is computationally efficient and fast but assumes feature independence. This led to lower recall and F1 score compared to other models.

### Ensemble Model – Random Forest
Random Forest significantly improved overall performance. By combining multiple decision trees, it achieved strong accuracy, AUC, and MCC while reducing overfitting.

### Ensemble Model – XGBoost
XGBoost delivered the best performance across all evaluation metrics. It achieved the highest accuracy, AUC, F1 score, and MCC, demonstrating its ability to capture complex relationships in the dataset.

---

## Streamlit Application Features

The deployed Streamlit application provides an interactive interface with:

- Test CSV dataset Download option
- CSV dataset upload option
- Model selection dropdown
- Display of evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - MCC
  - AUC
- Classification report visualization
- Confusion matrix display
- Model descriptions for educational understanding
