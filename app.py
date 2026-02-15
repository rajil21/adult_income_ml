import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

st.set_page_config(page_title="Adult Income Classifier", layout="wide")

st.title("Adult Income Prediction using Multiple ML Models")

st.markdown(
"""
This application compares multiple machine learning classifiers for predicting
whether an individual's income exceeds $50K using the Adult Income dataset.
"""
)

st.write("Predict whether a person earns >50K or <=50K")




# Model selection
model_display = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree Classifier",
        "K-Nearest Neighbor Classifier",
        "Naive Bayes Classifier",
        "Ensemble Model - Random Forest",
        "Ensemble Model - XGBoost"
    ]
)

model_map = {
    "Logistic Regression": "lr",
    "Decision Tree Classifier": "dt",
    "K-Nearest Neighbor Classifier": "knn",
    "Naive Bayes Classifier": "nb",
    "Ensemble Model - Random Forest": "rf",
    "Ensemble Model - XGBoost": "xgb"
}

model_info = {
    "Logistic Regression":
        "A linear model used for binary classification. Works well as a strong baseline and is easy to interpret.",

    "Decision Tree Classifier":
        "A tree-based model that splits data based on feature importance. Easy to visualize but can overfit.",

    "K-Nearest Neighbor Classifier":
        "A distance-based model that classifies samples based on the closest training examples. Sensitive to scaling.",

    "Naive Bayes Classifier":
        "A probabilistic classifier based on Bayes' theorem with independence assumption. Fast and efficient.",

    "Ensemble Model - Random Forest":
        "An ensemble of multiple decision trees that improves accuracy and reduces overfitting.",

    "Ensemble Model - XGBoost":
        "A powerful gradient boosting algorithm that builds trees sequentially to optimize performance."
}

model_name = model_map[model_display]

st.markdown("### Selected Model")
st.write(model_display)

st.info(model_info[model_display])

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])


if uploaded_file is None:
    st.info("Please upload adult.csv to start prediction.")
else:
    df = pd.read_csv(uploaded_file)

    # Handle missing values
    df = df.replace('?', pd.NA)
    df = df.dropna()

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("income", axis=1)
    y = df["income"]

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Load model
    model = joblib.load(f"models/{model_name}.pkl")

    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y, y_pred):.3f}")
    col3.metric("Recall", f"{recall_score(y, y_pred):.3f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("F1 Score", f"{f1_score(y, y_pred):.3f}")
    col5.metric("MCC", f"{matthews_corrcoef(y, y_pred):.3f}")
    col6.metric("AUC", f"{roc_auc_score(y, y_prob):.3f}")

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)