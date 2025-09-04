import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import joblib
import os

INPUT_LABELED = "./dataset/outputs/sample_labeled.csv"
MODEL_OUT = "./ml/model.pkl"

LEVELS = ["title", "H1", "H2", "H3", "H4"]

def extract_features(df):
    df["text_length"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df["is_uppercase"] = df["text"].apply(lambda x: int(str(x).isupper()))
    df["is_titlecase"] = df["text"].apply(lambda x: int(str(x).istitle()))

    features = df[[
        "font_size", "x_position", "y_position", "page",
        "text_length", "word_count", "is_uppercase", "is_titlecase"
    ]].fillna(0)

    return features

def train_model():
    df = pd.read_csv(INPUT_LABELED)
    df = df[df["label"].notna()]

    X = extract_features(df)
    y = df["label"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    # Calculate precision, recall, and support (frequency)
    precision, recall, _, support = precision_recall_fscore_support(
        y_val, y_pred, labels=[le.transform([lvl])[0] for lvl in LEVELS if lvl in le.classes_], zero_division=0
    )

    # Print table header
    print("\n| Level  | Precision | Recall | Frequency |")
    print("|--------|-----------|--------|-----------|")
    for i, lvl in enumerate([lvl for lvl in LEVELS if lvl in le.classes_]):
        print(f"| {lvl:<6} |   {precision[i]:.2f}    |  {recall[i]:.2f} |   {support[i]:>5}   |")
    print("|--------|-----------|--------|-----------|")
    print(f"\nAccuracy: {acc*100:.2f}%\n")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump((model, le), MODEL_OUT)

if __name__ == "__main__":
    train_model()
