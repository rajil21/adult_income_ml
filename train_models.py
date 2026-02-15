import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import *

# Loading the data set
df = pd.read_csv("adult.csv")

# Preprocessing data

# Remove missing records
df = df.replace('?', pd.NA)
df = df.dropna()

# Encoding categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Target column
X = df.drop("income", axis=1)
y = df["income"]

# Split records
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)

# Scaling training and test data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Models configuration
models = {
    "lr": LogisticRegression(max_iter=1000),
    "dt": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "nb": GaussianNB(),
    "rf": RandomForestClassifier(),
    "xgb": XGBClassifier(eval_metric='logloss')
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    joblib.dump(model, f"models/{name}.pkl")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append([
        name,
        accuracy_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred)
    ])

cols = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
print(pd.DataFrame(results, columns=cols))
