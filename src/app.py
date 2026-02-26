import os
from flask import Flask, render_template_string
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "bot_detection_data.csv")

app = Flask(__name__)


# -------------------------------------------------
# DATA & MODEL UTILITIES
# -------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Encode Verified (TRUE/FALSE) -> 1/0
    df["Verified"] = (
        df["Verified"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0})
        .fillna(0)
        .astype(int)
    )

    # Encode Bot Label -> 1 (bot/sybil) / 0 (benign)
    def enc_bot(x):
        s = str(x).strip().upper()
        if s in {"1", "TRUE", "BOT"}:
            return 1
        return 0

    df["Bot Label"] = df["Bot Label"].apply(enc_bot).astype(int)

    # Extra simple features
    df["Tweet Length"] = df["Tweet"].astype(str).str.len()
    df["Hashtag Count"] = df["Hashtags"].astype(str).apply(
        lambda x: 0 if x.strip() == "" else len(x.split())
    )

    return df


def build_features(df: pd.DataFrame):
    feature_cols = [
        "Retweet Count",
        "Mention Count",
        "Follower Count",
        "Verified",
        "Tweet Length",
        "Hashtag Count",
    ]

    df_clean = df.dropna(subset=feature_cols + ["Bot Label"]).copy()

    X_raw = df_clean[feature_cols].values.astype(float)
    y = df_clean["Bot Label"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return df_clean, X_scaled, y, feature_cols


def train_model(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    return model, metrics


# -------------------------------------------------
# ROUTE
# -------------------------------------------------
@app.route("/")
def dashboard():
    df = load_data()
    df_clean, X, y, feature_cols = build_features(df)
    model, metrics = train_model(X, y)

    total_users = len(df_clean)
    total_bots = int(df_clean["Bot Label"].sum())
    total_humans = total_users - total_bots
    bot_ratio = (total_bots / total_users * 100) if total_users > 0 else 0.0

    # Top suspected bots
    suspected = df_clean[df_clean["Bot Label"] == 1].copy()
    cols_to_show = [
        "User ID",
        "Username",
        "Follower Count",
        "Retweet Count",
        "Mention Count",
        "Verified",
        "Tweet Length",
    ]
    cols_to_show = [c for c in cols_to_show if c in suspected.columns]
    suspected = suspected[cols_to_show].head(20)

    # Render simple HTML
    html = """
    <!doctype html>
    <html lang="en">
    <head>
        <title>Sybil / Bot Detection Dashboard</title>
        <meta charset="utf-8">
        <link
          href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
          rel="stylesheet"
        >
        <style>
          body { padding: 20px; background-color: #0f172a; color: #e5e7eb; }
          .card { border-radius: 1rem; }
          .card-title { font-size: 0.9rem; text-transform: uppercase; }
          .metric-value { font-size: 1.4rem; font-weight: 600; }
          table { font-size: 0.85rem; }
          .table thead th { background-color: #111827; color: #9ca3af; }
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <h2 class="mb-4">Sybil / Bot Detection Dashboard</h2>
            <p class="text-secondary">
              Using the Bot Detection Dataset to identify Sybil-like (bot) accounts in a Twitter-style network.
            </p>

            <!-- Top summary row -->
            <div class="row g-3 mb-4">
                <div class="col-md-3">
                    <div class="card bg-dark text-light p-3">
                        <div class="card-title text-muted">Total Users</div>
                        <div class="metric-value">{{ total_users }}</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-dark text-light p-3">
                        <div class="card-title text-muted">Sybil / Bot Accounts</div>
                        <div class="metric-value">{{ total_bots }}</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-dark text-light p-3">
                        <div class="card-title text-muted">Benign Accounts</div>
                        <div class="metric-value">{{ total_humans }}</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-dark text-light p-3">
                        <div class="card-title text-muted">Bot Ratio</div>
                        <div class="metric-value">{{ "%.2f"|format(bot_ratio) }} %</div>
                    </div>
                </div>
            </div>

            <!-- Model metrics -->
            <h4 class="mb-3">Model Performance (Random Forest)</h4>
            <div class="row g-3 mb-4">
                <div class="col-md-3">
                    <div class="card bg-dark text-light p-3">
                        <div class="card-title text-muted">Accuracy</div>
                        <div class="metric-value">{{ "%.3f"|format(metrics.accuracy) }}</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-dark text-light p-3">
                        <div class="card-title text-muted">Precision</div>
                        <div class="metric-value">{{ "%.3f"|format(metrics.precision) }}</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-dark text-light p-3">
                        <div class="card-title text-muted">Recall</div>
                        <div class="metric-value">{{ "%.3f"|format(metrics.recall) }}</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-dark text-light p-3">
                        <div class="card-title text-muted">F1 Score</div>
                        <div class="metric-value">{{ "%.3f"|format(metrics.f1) }}</div>
                    </div>
                </div>
            </div>

            <!-- Table of suspected bots -->
            <h4 class="mb-3">Top Suspected Sybil / Bot Accounts</h4>
            <div class="table-responsive">
                <table class="table table-dark table-striped table-sm align-middle">
                    <thead>
                        <tr>
                        {% for col in suspected.columns %}
                            <th>{{ col }}</th>
                        {% endfor %}
                        </tr>
                    </thead>
                    <tbody>
                    {% for _, row in suspected.iterrows() %}
                        <tr>
                        {% for col in suspected.columns %}
                            <td>{{ row[col] }}</td>
                        {% endfor %}
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """

    # Jinja expects something like a namespace for metrics
    class M:
        def __init__(self, d):
            self.__dict__.update(d)

    metrics_obj = M(metrics)

    return render_template_string(
        html,
        total_users=total_users,
        total_bots=total_bots,
        total_humans=total_humans,
        bot_ratio=bot_ratio,
        metrics=metrics_obj,
        suspected=suspected,
    )


if __name__ == "__main__":
    # Run Flask app
    app.run(debug=True)
