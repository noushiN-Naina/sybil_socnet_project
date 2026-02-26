import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "bot_detection_data.csv")

N_BLOCKS = 71
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ============================================================
# 1. LOADING & BASIC PREPROCESSING
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print("Raw shape:", df.shape)
    return df


def encode_verified(df: pd.DataFrame) -> pd.DataFrame:
    df["Verified"] = (
        df["Verified"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0})
        .fillna(0)
        .astype(int)
    )
    return df


def encode_bot_label(df: pd.DataFrame) -> pd.DataFrame:
    def enc(x):
        s = str(x).strip().upper()
        if s in {"1", "TRUE", "BOT"}:
            return 1
        return 0

    df["Bot Label"] = df["Bot Label"].apply(enc).astype(int)
    return df


def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    # Tweet length in characters
    df["Tweet Length"] = df["Tweet"].astype(str).str.len()

    # Hashtag count (rough: count of '#' or words in Hashtags column)
    df["Hashtag Count"] = df["Hashtags"].astype(str).apply(
        lambda x: 0 if x.strip() == "" else len(x.split())
    )
    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    Select feature columns and standardize them.
    """
    feature_cols = [
        "Retweet Count",
        "Mention Count",
        "Follower Count",
        "Verified",
        "Tweet Length",
        "Hashtag Count",
    ]

    needed = feature_cols + ["Bot Label"]
    df_clean = df.dropna(subset=needed).copy()

    X_raw = df_clean[feature_cols].values.astype(float)
    y = df_clean["Bot Label"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print("\nAfter cleaning:")
    print("  X shape:", X_scaled.shape)
    print("  y shape:", y.shape)
    print("  Class counts (0=benign,1=sybil):", np.bincount(y))

    return X_scaled, y, feature_cols


# ============================================================
# 2. 71-BLOCK SPLITTING & MATRIX GENERATION
# ============================================================

def split_into_blocks_indices(n_rows: int, n_blocks: int = N_BLOCKS):
    """
    Block i: i, i+71, i+2*71, ...
    """
    blocks = []
    for offset in range(n_blocks):
        idx = list(range(offset, n_rows, n_blocks))
        if idx:
            blocks.append(idx)
    return blocks


def build_block_matrices(X: np.ndarray, y: np.ndarray, blocks):
    """
    For each block (list of indices), build rows:

        [feature1, ..., featureK, Xi]

    Xi = number of features (same for all rows).
    Then concatenate all blocks.
    """
    X_list = []
    y_list = []

    k = X.shape[1]  # number of features

    for i, idx in enumerate(blocks):
        X_block = X[idx]
        y_block = y[idx]
        Xi = np.full((len(idx), 1), k, dtype=np.int64)
        block_matrix = np.hstack([X_block, Xi])
        X_list.append(block_matrix)
        y_list.append(y_block)
        print(f"Block {i}: X_block={block_matrix.shape}, labels={np.bincount(y_block)}")

    X_new = np.vstack(X_list)
    y_new = np.concatenate(y_list)

    print("\nFinal feature matrix (after blocks):", X_new.shape)
    print("Final labels shape:", y_new.shape)
    return X_new, y_new


# ============================================================
# 3. MODEL TRAINING & EVALUATION
# ============================================================

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n========== {name} ==========")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


def run_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    evaluate_model("K-Nearest Neighbors", knn, X_train, X_test, y_train, y_test)

    # SVM
    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    evaluate_model("Support Vector Machine (RBF)", svm, X_train, X_test, y_train, y_test)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    evaluate_model("Random Forest", rf, X_train, X_test, y_train, y_test)


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print("=== Sybil / Bot Detection – Modeling Script ===")

    # Load
    df = load_data(DATA_PATH)

    # Basic preprocessing
    df = encode_verified(df)
    df = encode_bot_label(df)
    df = add_simple_features(df)

    # Features + scaling
    X_scaled, y, feature_cols = build_feature_matrix(df)

    # 71-block splitting
    blocks = split_into_blocks_indices(len(y), N_BLOCKS)
    X, y_final = build_block_matrices(X_scaled, y, blocks)

    # Train & evaluate models
    run_models(X, y_final)


if __name__ == "__main__":
    main()
