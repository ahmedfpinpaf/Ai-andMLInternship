# iris_classification.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 1. Load data
iris = load_iris()
X = iris.data          # features: sepal length/width, petal length/width
y = iris.target        # labels: 0=setosa, 1=versicolor, 2=virginica

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)
print(df.head())

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale features (helps Logistic Regression converge; optional for Decision Tree)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model — Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# --- To use Decision Tree instead, comment above and uncomment below ---
# model = DecisionTreeClassifier(max_depth=3, random_state=42)
# model.fit(X_train, y_train)   # no scaling needed for trees

# 5. Predict & evaluate
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))