# train_and_save.py
import io, zipfile, requests, joblib
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def load_sms_spam():
    # Robust fetch with fallback: UCI zip → local path
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open("SMSSpamCollection") as f:
            df = pd.read_csv(f, sep="\t", header=None, names=["label","message"], encoding="utf-8")
    except Exception:
        # Fallback: if you already downloaded a copy next to this script
        df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label","message"], encoding="utf-8")
    return df

df = load_sms_spam()
X = df["message"]
y = df["label"].map({"ham":0, "spam":1})

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=3000)),
    ("clf", MultinomialNB())
])

cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(f"5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(pipe, "spam_nb_pipeline.joblib")
print("Saved model to spam_nb_pipeline.joblib")
