import os
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

if not csv_files:
    raise FileNotFoundError("Aucun CSV trouvé dans data/processed")

df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

X = df.drop("label", axis=1)
y = df["label"]

# =========================================================
# SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# MODELS
# =========================================================

models = {
    "LogisticRegression": Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=0.0001,
        max_iter=50
    ))
    ]),


    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            C=0.01,
            gamma=10
        ))
    ]),

    "RandomForest": RandomForestClassifier(
        n_estimators=5,
        max_depth=1,
        min_samples_split=50,
        min_samples_leaf=25,
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=50
        ))
    ])
}

# =========================================================
# TRAIN & EVAL
# =========================================================

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"{name} -> Accuracy: {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds, labels=["UP", "DOWN", "NONE"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["UP", "DOWN", "NONE"])
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# =========================================================
# SAVE BEST
# =========================================================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

joblib.dump(best_model, os.path.join(MODEL_DIR, "gesture_model.pkl"))

print("\n✅ Meilleur modèle :", best_model_name)
print("✅ Modèle sauvegardé : data/models/gesture_model.pkl")