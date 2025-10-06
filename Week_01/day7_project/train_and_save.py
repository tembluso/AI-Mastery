# train_and_save.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

"""
1.Load Titanic dataset.

2.Choose useful features (sex, pclass, age, fare, alone).

3.Split into train/test sets.

4.Define preprocessing rules for numeric + categorical features.

5.Train a Logistic Regression inside a Pipeline.

6.Evaluate accuracy on the test set.

7.Save the full pipeline (preprocessing + model) into one file for later use.
"""

# 1) Load
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

# 2) Select features (keep it simple + robust)
features = ["sex", "pclass", "age", "fare", "alone"]
target = "survived"
df = df[features + [target]]

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

# 4) Preprocess

num_feats = ["age", "fare"]
cat_feats = ["sex", "pclass", "alone"]  # treat pclass & alone as categorical for stability

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_feats),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_feats),
    ]
)

# 5) Model
clf = LogisticRegression(max_iter=500)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", clf)
])

pipe.fit(X_train, y_train)
acc = accuracy_score(y_test, pipe.predict(X_test))
print(f"Test accuracy: {acc:.3f}")

# 6) Save pipeline (preprocess + model together)
# joblib saves Python objects (like your whole pipeline) to a file.
joblib.dump(pipe, "titanic_pipeline.joblib")
print("Saved titanic_pipeline.joblib")
