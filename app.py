from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template_string, request
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Ignore unpickling warnings if scikit-learn version differs slightly
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

DATA_PATH = Path("data/names_gender.csv")
MODEL_PATH = Path("models/name_gender_model.joblib")

app = Flask(__name__)

# Cache for computed statistics and feature weights
stats_cache = {}
last_modified_timestamp = 0.0


def compute_stats() -> None:
    """Computes dataset statistics and model performance metrics."""
    global stats_cache
    if not DATA_PATH.exists():
        stats_cache = {"error": "Dataset not found. Run generate_dataset.py first."}
        return

    # Load dataset
    names: list[str] = []
    genders: list[str] = []
    with DATA_PATH.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            name = (row.get("name") or "").strip()
            gender = (row.get("gender") or "").strip().lower()
            if name and gender in {"male", "female"}:
                names.append(name)
                genders.append(gender)

    total_samples = len(names)
    male_count = sum(1 for g in genders if g == "male")
    female_count = sum(1 for g in genders if g == "female")

    # Name length distribution
    length_counts: dict[int, int] = {}
    for name in names:
        length = len(name)
        length_counts[length] = length_counts.get(length, 0) + 1

    sorted_lengths = sorted(length_counts.items())
    length_labels = [str(item[0]) for item in sorted_lengths]
    length_values = [item[1] for item in sorted_lengths]

    # Model evaluation
    model_loaded = False
    accuracy = 0.0
    confusion_mat = [[0, 0], [0, 0]]
    report = {}
    top_female: list[dict[str, float | str]] = []
    top_male: list[dict[str, float | str]] = []
    intercept = 0.0

    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            model_loaded = True

            # Recreate the exact test split from train_model.py to evaluate performance
            x_train, x_test, y_train, y_test = train_test_split(
                names,
                genders,
                test_size=0.2,
                random_state=42,
                stratify=genders,
            )
            predictions = model.predict(x_test)
            accuracy = float(accuracy_score(y_test, predictions))
            confusion_mat = confusion_matrix(y_test, predictions).tolist()
            report = classification_report(y_test, predictions, output_dict=True)

            # Feature coefficients (character n-grams)
            vectorizer = model.named_steps["vectorizer"]
            classifier = model.named_steps["classifier"]
            feature_names = vectorizer.get_feature_names_out()
            coefs = classifier.coef_[0]
            intercept = float(classifier.intercept_[0])

            ngram_weights = list(zip(feature_names, coefs))
            # Sort by weight (negative = female indicator, positive = male indicator)
            ngram_weights.sort(key=lambda x: x[1])

            top_female = [
                {"ngram": k, "weight": float(v)} for k, v in ngram_weights[:20]
            ]
            # Top male: positive weights, sorted descending
            top_male = [
                {"ngram": k, "weight": float(v)} for k, v in reversed(ngram_weights[-20:])
            ]
        except Exception as e:
            print(f"Error evaluating model: {e}")

    stats_cache = {
        "total_samples": total_samples,
        "male_count": male_count,
        "female_count": female_count,
        "model_loaded": model_loaded,
        "accuracy": accuracy,
        "confusion_matrix": confusion_mat,
        "report": report,
        "top_female_ngrams": top_female,
        "top_male_ngrams": top_male,
        "length_distribution": {
            "labels": length_labels,
            "values": length_values,
        },
        "intercept": intercept,
    }


def reload_resources_if_needed() -> None:
    """Reloads model stats dynamically if data/model files changed."""
    global last_modified_timestamp
    csv_mtime = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0
    model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
    current_stamp = csv_mtime + model_mtime

    if current_stamp != last_modified_timestamp or not stats_cache:
        compute_stats()
        last_modified_timestamp = current_stamp


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Gender Classifier Dashboard</title>
    <!-- Google Fonts: Outfit (Sans-serif) & JetBrains Mono (Monospace) -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <!-- Lucide Icons CDN -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
      :root {
        --bg: #090b11;
        --bg-gradient: radial-gradient(1200px 800px at 50% -10%, #1e1b4b 0%, #090b11 70%, #030406 100%);
        --card-bg: rgba(17, 24, 39, 0.5);
        --card-border: rgba(255, 255, 255, 0.05);
        --text: #f3f4f6;
        --text-muted: #9ca3af;
        --female: #ec4899;
        --female-rgb: 236, 72, 153;
        --male: #06b6d4;
        --male-rgb: 6, 182, 212;
        --accent: #6366f1;
        --accent-rgb: 99, 102, 241;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --font: 'Outfit', system-ui, -apple-system, sans-serif;
        --mono: 'JetBrains Mono', monospace;
      }

      * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
      }

      body {
        background: var(--bg);
        background-image: var(--bg-gradient);
        background-attachment: fixed;
        color: var(--text);
        font-family: var(--font);
        min-height: 100vh;
        overflow-x: hidden;
        padding-bottom: 40px;
      }

      header {
        border-bottom: 1px solid var(--card-border);
        background: rgba(9, 11, 17, 0.8);
        backdrop-filter: blur(12px);
        position: sticky;
        top: 0;
        z-index: 100;
        padding: 16px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }

      .logo-section {
        display: flex;
        align-items: center;
        gap: 12px;
      }

      .logo-icon {
        background: linear-gradient(135deg, var(--accent), var(--male));
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
      }

      .logo-title h1 {
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(to right, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }

      .logo-title p {
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 1px;
      }

      .header-meta {
        display: flex;
        gap: 16px;
      }

      .meta-pill {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--card-border);
        border-radius: 30px;
        padding: 6px 14px;
        font-size: 13px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .meta-pill.active {
        border-color: rgba(16, 185, 129, 0.3);
        background: rgba(16, 185, 129, 0.05);
        color: var(--success);
      }

      .meta-pill.error-pill {
        border-color: rgba(239, 68, 68, 0.3);
        background: rgba(239, 68, 68, 0.05);
        color: var(--danger);
      }

      .container {
        max-width: 1280px;
        margin: 32px auto;
        padding: 0 24px;
      }

      /* Quick Overview Stats cards */
      .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px;
        margin-bottom: 28px;
      }

      .stat-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, border-color 0.2s;
      }

      .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.1);
      }

      .stat-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .stat-card:nth-child(1) .stat-icon { background: rgba(99, 102, 241, 0.1); color: var(--accent); }
      .stat-card:nth-child(2) .stat-icon { background: rgba(6, 182, 212, 0.1); color: var(--male); }
      .stat-card:nth-child(3) .stat-icon { background: rgba(236, 72, 153, 0.1); color: var(--female); }
      .stat-card:nth-child(4) .stat-icon { background: rgba(16, 185, 129, 0.1); color: var(--success); }

      .stat-info h3 {
        font-size: 13px;
        color: var(--text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .stat-info p {
        font-size: 24px;
        font-weight: 700;
        margin-top: 4px;
        color: white;
      }

      /* Tabs navigation */
      .tabs {
        display: flex;
        gap: 8px;
        border-bottom: 1px solid var(--card-border);
        padding-bottom: 1px;
        margin-bottom: 30px;
      }

      .tab-btn {
        background: transparent;
        border: none;
        color: var(--text-muted);
        padding: 12px 20px;
        font-family: var(--font);
        font-size: 15px;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 8px;
        position: relative;
        transition: color 0.2s;
      }

      .tab-btn:hover {
        color: white;
      }

      .tab-btn.active {
        color: var(--accent);
      }

      .tab-btn.active::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--accent);
        border-radius: 2px;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.8);
      }

      /* Dashboard Content */
      .tab-content {
        display: none;
        animation: fadeIn 0.4s ease-out forwards;
      }

      .tab-content.active {
        display: block;
      }

      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
      }

      /* Cards and grid systems */
      .dashboard-row {
        display: grid;
        grid-template-columns: 1fr;
        gap: 24px;
      }

      @media (min-width: 992px) {
        .dashboard-row.two-cols {
          grid-template-columns: 1.1fr 0.9fr;
        }
        .dashboard-row.two-cols-equal {
          grid-template-columns: 1fr 1fr;
        }
      }

      .card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 28px;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        display: flex;
        flex-direction: column;
      }

      .card-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        color: white;
      }

      .card-title i {
        color: var(--accent);
      }

      /* Playground Tab Styles */
      .playground-input-group {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-bottom: 24px;
      }

      .playground-input-group label {
        font-size: 14px;
        font-weight: 500;
        color: var(--text-muted);
      }

      .input-wrapper {
        position: relative;
        display: flex;
        align-items: center;
      }

      .input-wrapper i {
        position: absolute;
        left: 18px;
        color: var(--text-muted);
      }

      .playground-input {
        width: 100%;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        height: 56px;
        padding: 12px 18px 12px 52px;
        font-family: var(--font);
        font-size: 16px;
        color: white;
        transition: border-color 0.2s, box-shadow 0.2s, background-color 0.2s;
      }

      .playground-input:focus {
        outline: none;
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        background: rgba(255, 255, 255, 0.05);
      }

      /* Real-time result panel */
      .prediction-card {
        padding: 24px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.01);
        border: 1px dashed var(--card-border);
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        position: relative;
      }

      .prediction-intro {
        color: var(--text-muted);
        max-width: 300px;
      }

      .prediction-intro i {
        font-size: 32px;
        margin-bottom: 12px;
        opacity: 0.6;
      }

      .prediction-result-panel {
        width: 100%;
        animation: fadeIn 0.3s ease-out;
      }

      .result-label-wrapper {
        margin-bottom: 16px;
      }

      .result-label {
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -1px;
        text-transform: uppercase;
        display: inline-block;
        padding: 4px 16px;
        border-radius: 12px;
      }

      .result-label.male {
        color: var(--male);
        background: rgba(6, 182, 212, 0.08);
        border: 1px solid rgba(6, 182, 212, 0.2);
        text-shadow: 0 0 15px rgba(6, 182, 212, 0.3);
      }

      .result-label.female {
        color: var(--female);
        background: rgba(236, 72, 153, 0.08);
        border: 1px solid rgba(236, 72, 153, 0.2);
        text-shadow: 0 0 15px rgba(236, 72, 153, 0.3);
      }

      /* Highlighted characters panel */
      .highlighted-name-display {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: 2px;
        margin: 20px 0;
        padding: 12px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.02);
        display: flex;
        justify-content: center;
        gap: 2px;
      }

      .name-char {
        transition: background-color 0.2s, color 0.2s, transform 0.2s;
        padding: 0 4px;
        border-radius: 4px;
        display: inline-block;
      }

      .name-char.highlight-male {
        background-color: rgba(6, 182, 212, 0.25);
        color: #e0f7fa;
        transform: translateY(-2px);
      }

      .name-char.highlight-female {
        background-color: rgba(236, 72, 153, 0.25);
        color: #fce4ec;
        transform: translateY(-2px);
      }

      /* Dual Probability progress bars */
      .prob-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        width: 100%;
        margin-top: 10px;
      }

      .prob-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
      }

      .prob-bar-label.male { color: var(--male); }
      .prob-bar-label.female { color: var(--female); }

      .prob-track {
        height: 10px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        overflow: hidden;
        position: relative;
      }

      .prob-fill {
        height: 100%;
        border-radius: 5px;
        width: 0;
        transition: width 0.6s cubic-bezier(0.1, 0.8, 0.2, 1);
      }

      .prob-fill.male {
        background: linear-gradient(to right, var(--accent), var(--male));
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.5);
      }

      .prob-fill.female {
        background: linear-gradient(to right, var(--accent), var(--female));
        box-shadow: 0 0 10px rgba(236, 72, 153, 0.5);
      }

      /* Explanation breakdown list */
      .explanation-panel {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 6px;
      }

      .explanation-panel::-webkit-scrollbar {
        width: 6px;
      }

      .explanation-panel::-webkit-scrollbar-thumb {
        background: var(--card-border);
        border-radius: 3px;
      }

      .ngram-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        margin-bottom: 8px;
        transition: transform 0.2s, background-color 0.2s, border-color 0.2s;
        cursor: pointer;
      }

      .ngram-row:hover {
        background-color: rgba(255, 255, 255, 0.04);
        border-color: rgba(255, 255, 255, 0.1);
        transform: scale(1.01);
      }

      .ngram-left {
        display: flex;
        align-items: center;
        gap: 12px;
      }

      .ngram-token {
        font-family: var(--mono);
        font-size: 13px;
        font-weight: 700;
        background: rgba(0, 0, 0, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.04);
      }

      .ngram-badge {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        padding: 2px 6px;
        border-radius: 12px;
      }

      .ngram-badge.male { color: var(--male); background: rgba(6, 182, 212, 0.1); }
      .ngram-badge.female { color: var(--female); background: rgba(236, 72, 153, 0.1); }

      .ngram-math {
        font-size: 12px;
        color: var(--text-muted);
      }

      .ngram-weight {
        font-family: var(--mono);
        font-size: 13px;
        font-weight: 600;
      }

      .ngram-weight.positive { color: var(--male); }
      .ngram-weight.negative { color: var(--female); }

      .intercept-display {
        font-size: 13px;
        color: var(--text-muted);
        text-align: right;
        margin-top: 12px;
        font-family: var(--mono);
      }

      /* Dataset Table styling */
      .dataset-header-actions {
        display: grid;
        grid-template-columns: 1fr;
        gap: 12px;
        margin-bottom: 20px;
      }

      @media (min-width: 600px) {
        .dataset-header-actions {
          grid-template-columns: 2fr 1.2fr 1fr;
        }
      }

      .dataset-search-input {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        height: 44px;
        padding: 0 14px;
        color: white;
        font-family: var(--font);
        font-size: 14px;
      }

      .dataset-search-input:focus {
        outline: none;
        border-color: var(--accent);
      }

      .select-dropdown {
        background: #111827;
        border: 1px solid var(--card-border);
        border-radius: 10px;
        height: 44px;
        padding: 0 12px;
        color: white;
        font-family: var(--font);
        font-size: 14px;
        cursor: pointer;
      }

      .select-dropdown:focus {
        outline: none;
        border-color: var(--accent);
      }

      .table-wrapper {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--card-border);
        background: rgba(0, 0, 0, 0.15);
      }

      table {
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        font-size: 14px;
      }

      th {
        background: rgba(255, 255, 255, 0.02);
        color: var(--text-muted);
        font-weight: 600;
        padding: 12px 18px;
        border-bottom: 1px solid var(--card-border);
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.5px;
      }

      td {
        padding: 14px 18px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.02);
        color: var(--text);
      }

      tr:last-child td {
        border-bottom: none;
      }

      tr:hover td {
        background: rgba(255, 255, 255, 0.01);
      }

      .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .badge.badge-male {
        background: rgba(6, 182, 212, 0.1);
        color: var(--male);
        border: 1px solid rgba(6, 182, 212, 0.15);
      }

      .badge.badge-female {
        background: rgba(236, 72, 153, 0.1);
        color: var(--female);
        border: 1px solid rgba(236, 72, 153, 0.15);
      }

      /* Pagination controls */
      .pagination-controls {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 16px;
        padding-top: 10px;
      }

      .pagination-text {
        font-size: 13px;
        color: var(--text-muted);
      }

      .pagination-buttons {
        display: flex;
        gap: 8px;
      }

      .btn {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--card-border);
        color: var(--text);
        padding: 8px 14px;
        border-radius: 8px;
        cursor: pointer;
        font-family: var(--font);
        font-size: 13px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
        transition: background-color 0.2s, border-color 0.2s;
      }

      .btn:hover:not(:disabled) {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.15);
      }

      .btn:disabled {
        opacity: 0.4;
        cursor: not-allowed;
      }

      .btn.primary {
        background: var(--accent);
        color: white;
        border-color: transparent;
      }

      .btn.primary:hover {
        background: #4f46e5;
      }

      /* Model internals grid layouts */
      .metrics-grid-row {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 14px;
        margin-bottom: 24px;
      }

      .metric-small-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
      }

      .metric-small-card label {
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        font-weight: 600;
      }

      .metric-small-card p {
        font-size: 20px;
        font-weight: 700;
        margin-top: 4px;
        color: white;
        font-family: var(--mono);
      }

      /* Confusion Matrix styling */
      .confusion-matrix {
        display: grid;
        grid-template-columns: auto 1fr 1fr;
        grid-template-rows: auto 1fr 1fr;
        gap: 10px;
        margin: 20px 0;
        max-width: 320px;
        align-self: center;
      }

      .cm-corner { grid-area: 1 / 1; }
      .cm-hdr-pred-f { grid-area: 1 / 2; text-align: center; font-size: 11px; font-weight: 600; color: var(--female); }
      .cm-hdr-pred-m { grid-area: 1 / 3; text-align: center; font-size: 11px; font-weight: 600; color: var(--male); }
      .cm-hdr-act-f { grid-area: 2 / 1; display: flex; align-items: center; font-size: 11px; font-weight: 600; color: var(--female); writing-mode: vertical-lr; transform: rotate(180deg); margin-right: 6px; }
      .cm-hdr-act-m { grid-area: 3 / 1; display: flex; align-items: center; font-size: 11px; font-weight: 600; color: var(--male); writing-mode: vertical-lr; transform: rotate(180deg); margin-right: 6px; }

      .cm-cell {
        aspect-ratio: 1.2;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 12px;
      }

      .cm-cell.hit {
        background: rgba(16, 185, 129, 0.06);
        border-color: rgba(16, 185, 129, 0.15);
      }

      .cm-cell.miss {
        background: rgba(239, 68, 68, 0.04);
        border-color: rgba(239, 68, 68, 0.1);
      }

      .cm-value { font-size: 20px; font-weight: 700; font-family: var(--mono); color: white; }
      .cm-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; margin-top: 2px; }

      /* Top feature bars */
      .features-split-panel {
        display: grid;
        grid-template-columns: 1fr;
        gap: 20px;
      }

      @media (min-width: 768px) {
        .features-split-panel {
          grid-template-columns: 1fr 1fr;
        }
      }

      .feature-bar-list {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 12px;
      }

      .feature-bar-item {
        display: flex;
        align-items: center;
        gap: 12px;
      }

      .feature-bar-label {
        font-family: var(--mono);
        font-size: 13px;
        font-weight: 700;
        width: 48px;
        background: rgba(0,0,0,0.2);
        padding: 2px 6px;
        border-radius: 4px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.02);
      }

      .feature-bar-container {
        flex-grow: 1;
        height: 18px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 4px;
        overflow: hidden;
        position: relative;
      }

      .feature-bar-fill {
        height: 100%;
        border-radius: 4px;
      }

      .feature-bar-fill.female {
        background: linear-gradient(to right, rgba(236, 72, 153, 0.4), var(--female));
        float: right;
      }

      .feature-bar-fill.male {
        background: linear-gradient(to right, rgba(6, 182, 212, 0.4), var(--male));
      }

      .feature-bar-value {
        font-family: var(--mono);
        font-size: 12px;
        color: var(--text-muted);
        width: 50px;
        text-align: right;
      }

      /* Alert banners */
      .alert-banner {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.15);
        color: var(--warning);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 14px;
        font-size: 14px;
        font-weight: 500;
      }

      .alert-banner i {
        font-size: 20px;
      }
    </style>
  </head>
  <body>
    <header>
      <div class="logo-section">
        <div class="logo-icon">
          <i data-lucide="brain-circuit"></i>
        </div>
        <div class="logo-title">
          <h1>Gender Classifier</h1>
          <p>Machine Learning Dashboard</p>
        </div>
      </div>
      <div class="header-meta">
        <div id="model-status-pill" class="meta-pill">
          <i data-lucide="loader-2" class="animate-spin icon-sm"></i> Checking Model...
        </div>
        <div class="meta-pill">
          <i data-lucide="tag" class="icon-sm"></i> v1.0.0
        </div>
      </div>
    </header>

    <main class="container">
      <!-- Alert banner if model is missing -->
      <div id="missing-model-alert" class="alert-banner" style="display: none;">
        <i data-lucide="alert-triangle"></i>
        <div>
          <strong>Model Not Found!</strong> The classifier model is missing from <code>models/name_gender_model.joblib</code>. Please run <code>python train_model.py</code> to train it first.
        </div>
      </div>

      <!-- Quick statistics overview -->
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-icon"><i data-lucide="database"></i></div>
          <div class="stat-info">
            <h3>Total Samples</h3>
            <p id="stat-total">-</p>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon"><i data-lucide="mars"></i></div>
          <div class="stat-info">
            <h3>Male Split</h3>
            <p id="stat-male">-</p>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon"><i data-lucide="venus"></i></div>
          <div class="stat-info">
            <h3>Female Split</h3>
            <p id="stat-female">-</p>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon"><i data-lucide="line-chart"></i></div>
          <div class="stat-info">
            <h3>Model Accuracy</h3>
            <p id="stat-accuracy">-</p>
          </div>
        </div>
      </div>

      <!-- Tabs Navigation -->
      <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('playground')">
          <i data-lucide="sparkles"></i> Predictor Playground
        </button>
        <button class="tab-btn" onclick="switchTab('dataset')">
          <i data-lucide="bar-chart-3"></i> Dataset Insights
        </button>
        <button class="tab-btn" onclick="switchTab('analytics')">
          <i data-lucide="cpu"></i> Model Analytics
        </button>
      </div>

      <!-- TAB 1: PLAYGROUND -->
      <div id="tab-playground" class="tab-content active">
        <div class="dashboard-row two-cols">
          <!-- Prediction Form Card -->
          <div class="card">
            <h2 class="card-title"><i data-lucide="terminal"></i> Interactive Inference</h2>
            <div class="playground-input-group">
              <label for="name-input">Enter a Name</label>
              <div class="input-wrapper">
                <i data-lucide="user"></i>
                <input 
                  type="text" 
                  id="name-input" 
                  class="playground-input" 
                  placeholder="Type a name (e.g. Durga, Ramesh, Vijaya, Srinivas)..." 
                  oninput="handleNameInput(this.value)"
                  autocomplete="off"
                  autofocus
                />
              </div>
            </div>

            <!-- Prediction Result Visualizer -->
            <div class="prediction-card">
              <!-- Default state -->
              <div id="predict-default" class="prediction-intro">
                <i data-lucide="search-code"></i>
                <p>Type a name above to query the gender classification model in real time.</p>
              </div>

              <!-- Loading state -->
              <div id="predict-loading" class="prediction-intro" style="display: none;">
                <i data-lucide="loader-2" class="animate-spin text-accent"></i>
                <p>Analyzing character patterns...</p>
              </div>

              <!-- Loaded result state -->
              <div id="predict-result" class="prediction-result-panel" style="display: none;">
                <div class="result-label-wrapper">
                  <span id="predict-label" class="result-label male">Male</span>
                </div>
                
                <!-- Character highlighting display -->
                <div class="highlighted-name-display" id="char-highlight-display">
                  <!-- Generated letters spans go here -->
                </div>

                <div class="prob-container">
                  <!-- Male probability progress -->
                  <div class="prob-bar-group">
                    <div class="prob-bar-label male">
                      <span>Male Probability</span>
                      <span id="prob-val-male">0%</span>
                    </div>
                    <div class="prob-track">
                      <div id="prob-fill-male" class="prob-fill male"></div>
                    </div>
                  </div>

                  <!-- Female probability progress -->
                  <div class="prob-bar-group">
                    <div class="prob-bar-label female">
                      <span>Female Probability</span>
                      <span id="prob-val-female">0%</span>
                    </div>
                    <div class="prob-track">
                      <div id="prob-fill-female" class="prob-fill female"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Explainable AI card -->
          <div class="card">
            <h2 class="card-title">
              <i data-lucide="eye"></i> 
              Glassbox Model Explanation 
            </h2>
            <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px;">
              Hover over a character n-gram below to highlight its occurrence in the name. n-grams represent 2-to-4 character substrings used by the model to assign gender weights.
            </p>
            <div id="explanation-container" class="explanation-panel">
              <div class="prediction-intro" style="margin: 40px auto; text-align: center;">
                <i data-lucide="help-circle" style="font-size: 24px; margin-bottom: 8px;"></i>
                <p>No name currently entered to explain.</p>
              </div>
            </div>
            <div id="intercept-val" class="intercept-display" style="display: none;">
              Model Intercept: 0.0000
            </div>
          </div>
        </div>
      </div>

      <!-- TAB 2: DATASET INSIGHTS -->
      <div id="tab-dataset" class="tab-content">
        <div class="dashboard-row two-cols-equal" style="margin-bottom: 24px;">
          <!-- Gender balance chart card -->
          <div class="card" style="height: 320px; align-items: center;">
            <h2 class="card-title" style="align-self: flex-start;"><i data-lucide="pie-chart"></i> Gender Split Balance</h2>
            <div style="width: 100%; max-width: 220px; height: 200px; display: flex; align-items: center; justify-content: center;">
              <canvas id="genderPieChart"></canvas>
            </div>
          </div>

          <!-- Name length distribution chart card -->
          <div class="card" style="height: 320px;">
            <h2 class="card-title"><i data-lucide="bar-chart-horizontal"></i> Name Length Distribution</h2>
            <div style="width: 100%; height: 200px;">
              <canvas id="lengthBarChart"></canvas>
            </div>
          </div>
        </div>

        <!-- Dataset Explorer Table Card -->
        <div class="card">
          <h2 class="card-title"><i data-lucide="list"></i> Dataset Explorer</h2>
          <div class="dataset-header-actions">
            <input 
              type="text" 
              id="dataset-search" 
              class="dataset-search-input" 
              placeholder="Search names in database..." 
              oninput="handleDatasetSearch()"
            />
            <select id="dataset-gender-filter" class="select-dropdown" onchange="handleDatasetFilter()">
              <option value="all">All Genders</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
            <div></div> <!-- Spacer -->
          </div>

          <div class="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Gender Class</th>
                </tr>
              </thead>
              <tbody id="dataset-tbody">
                <!-- Table rows generated dynamically -->
              </tbody>
            </table>
          </div>

          <!-- Table Pagination -->
          <div class="pagination-controls">
            <div class="pagination-text" id="pagination-stats">
              Showing 0 to 0 of 0 names
            </div>
            <div class="pagination-buttons">
              <button id="btn-prev" class="btn" onclick="datasetPrevPage()" disabled>
                <i data-lucide="chevron-left"></i> Previous
              </button>
              <button id="btn-next" class="btn" onclick="datasetNextPage()" disabled>
                Next <i data-lucide="chevron-right"></i>
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- TAB 3: MODEL ANALYTICS -->
      <div id="tab-analytics" class="tab-content">
        <div class="dashboard-row two-cols">
          <!-- Performance Metrics Card -->
          <div class="card">
            <h2 class="card-title"><i data-lucide="gauge"></i> Evaluation Report (20% Holdout Split)</h2>
            <div class="metrics-grid-row">
              <div class="metric-small-card">
                <label>F1-Score (Male)</label>
                <p id="metric-f1-male">-</p>
              </div>
              <div class="metric-small-card">
                <label>F1-Score (Female)</label>
                <p id="metric-f1-female">-</p>
              </div>
            </div>

            <!-- Confusion Matrix Display -->
            <div style="display: flex; flex-direction: column; align-items: center; margin-top: 10px;">
              <h3 style="font-size: 13px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; margin-bottom: 10px;">
                Confusion Matrix Grid
              </h3>
              <div class="confusion-matrix">
                <div class="cm-corner"></div>
                <div class="cm-hdr-pred-f">PRED FEMALE</div>
                <div class="cm-hdr-pred-m">PRED MALE</div>
                <div class="cm-hdr-act-f">ACTUAL FEMALE</div>
                <div class="cm-hdr-act-m">ACTUAL MALE</div>

                <div class="cm-cell hit" id="cm-tn-box">
                  <span class="cm-value" id="cm-tn">0</span>
                  <span class="cm-label">True Neg (F)</span>
                </div>
                <div class="cm-cell miss" id="cm-fp-box">
                  <span class="cm-value" id="cm-fp">0</span>
                  <span class="cm-label">False Pos</span>
                </div>
                <div class="cm-cell miss" id="cm-fn-box">
                  <span class="cm-value" id="cm-fn">0</span>
                  <span class="cm-label">False Neg</span>
                </div>
                <div class="cm-cell hit" id="cm-tp-box">
                  <span class="cm-value" id="cm-tp">0</span>
                  <span class="cm-label">True Pos (M)</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Feature Importance Card -->
          <div class="card">
            <h2 class="card-title"><i data-lucide="activity"></i> Feature Coefficients (Top character n-grams)</h2>
            <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 20px;">
              These features are the character patterns that hold the highest absolute weight in the logistic regression classification.
            </p>
            
            <div class="features-split-panel">
              <!-- Female indicator list -->
              <div>
                <h3 style="font-size: 14px; font-weight: 700; color: var(--female); margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                  <i data-lucide="venus" class="icon-sm"></i> Female Indicators
                </h3>
                <div id="female-features-list" class="feature-bar-list">
                  <!-- Horizontal bars go here -->
                </div>
              </div>

              <!-- Male indicator list -->
              <div>
                <h3 style="font-size: 14px; font-weight: 700; color: var(--male); margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                  <i data-lucide="mars" class="icon-sm"></i> Male Indicators
                </h3>
                <div id="male-features-list" class="feature-bar-list">
                  <!-- Horizontal bars go here -->
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>

    <script>
      // Global stats and charts cache
      let appStats = null;
      let datasetPage = 1;
      let datasetSearchTimeout = null;
      let charts = {};
      let predictDebounceTimeout = null;

      // Initialize application
      document.addEventListener('DOMContentLoaded', () => {
        lucide.createIcons();
        fetchStats();
        loadDataset();
      });

      // Tab switcher
      function switchTab(tabId) {
        document.querySelectorAll('.tab-btn').forEach(btn => {
          btn.classList.remove('active');
        });
        document.querySelectorAll('.tab-content').forEach(content => {
          content.classList.remove('active');
        });

        // Activate button
        const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(btn => 
          btn.innerText.toLowerCase().includes(tabId)
        );
        if (activeBtn) activeBtn.classList.add('active');

        // Activate content
        const targetContent = document.getElementById(`tab-${tabId}`);
        if (targetContent) targetContent.classList.add('active');
      }

      // Fetch overview statistics and model metadata
      async function fetchStats() {
        try {
          const res = await fetch('/api/stats');
          const data = await res.json();

          if (data.error) {
            document.getElementById('missing-model-alert').style.display = 'flex';
            document.getElementById('model-status-pill').className = 'meta-pill error-pill';
            document.getElementById('model-status-pill').innerHTML = '<i data-lucide="alert-octagon" class="icon-sm"></i> Offline';
            lucide.createIcons();
            return;
          }

          appStats = data;
          
          // Populate status pills and overview statistics
          const statusPill = document.getElementById('model-status-pill');
          if (data.model_loaded) {
            statusPill.className = 'meta-pill active';
            statusPill.innerHTML = '<i data-lucide="check-circle" class="icon-sm"></i> Model Active';
          } else {
            statusPill.className = 'meta-pill error-pill';
            statusPill.innerHTML = '<i data-lucide="alert-octagon" class="icon-sm"></i> Model Missing';
            document.getElementById('missing-model-alert').style.display = 'flex';
          }

          document.getElementById('stat-total').innerText = data.total_samples.toLocaleString();
          
          const malePct = ((data.male_count / data.total_samples) * 100).toFixed(0);
          const femalePct = ((data.female_count / data.total_samples) * 100).toFixed(0);
          
          document.getElementById('stat-male').innerText = `${data.male_count.toLocaleString()} (${malePct}%)`;
          document.getElementById('stat-female').innerText = `${data.female_count.toLocaleString()} (${femalePct}%)`;
          document.getElementById('stat-accuracy').innerText = `${(data.accuracy * 100).toFixed(1)}%`;

          // Setup model analytics tab details
          setupModelAnalytics(data);

          // Render charts under Dataset Insights
          renderDatasetCharts(data);

          lucide.createIcons();
        } catch (err) {
          console.error("Failed to load statistics:", err);
        }
      }

      // Setup Model analytics metrics and n-grams lists
      function setupModelAnalytics(data) {
        if (!data.model_loaded) return;

        // F1 Scores
        const report = data.report;
        if (report) {
          if (report['male']) {
            document.getElementById('metric-f1-male').innerText = report['male']['f1-score'].toFixed(3);
          }
          if (report['female']) {
            document.getElementById('metric-f1-female').innerText = report['female']['f1-score'].toFixed(3);
          }
        }

        // Confusion Matrix
        const cm = data.confusion_matrix;
        if (cm) {
          document.getElementById('cm-tn').innerText = cm[0][0]; // TN
          document.getElementById('cm-fp').innerText = cm[0][1]; // FP
          document.getElementById('cm-fn').innerText = cm[1][0]; // FN
          document.getElementById('cm-tp').innerText = cm[1][1]; // TP
        }

        // Top features horizontal lists
        const femaleList = document.getElementById('female-features-list');
        const maleList = document.getElementById('male-features-list');

        // Render female indicators (coefficients are negative)
        // Find max absolute female weight to normalize bar scale
        const maxFemaleWeight = Math.max(...data.top_female_ngrams.map(x => Math.abs(x.weight)));
        femaleList.innerHTML = data.top_female_ngrams.slice(0, 10).map(item => {
          const pct = (Math.abs(item.weight) / maxFemaleWeight * 100).toFixed(0);
          return `
            <div class="feature-bar-item">
              <span class="feature-bar-label" style="color: var(--female);">${escapeHtml(item.ngram)}</span>
              <div class="feature-bar-container">
                <div class="feature-bar-fill female" style="width: ${pct}%;"></div>
              </div>
              <span class="feature-bar-value">${item.weight.toFixed(3)}</span>
            </div>
          `;
        }).join('');

        // Render male indicators (coefficients are positive)
        const maxMaleWeight = Math.max(...data.top_male_ngrams.map(x => x.weight));
        maleList.innerHTML = data.top_male_ngrams.slice(0, 10).map(item => {
          const pct = (item.weight / maxMaleWeight * 100).toFixed(0);
          return `
            <div class="feature-bar-item">
              <span class="feature-bar-label" style="color: var(--male);">${escapeHtml(item.ngram)}</span>
              <div class="feature-bar-container">
                <div class="feature-bar-fill male" style="width: ${pct}%;"></div>
              </div>
              <span class="feature-bar-value">+${item.weight.toFixed(3)}</span>
            </div>
          `;
        }).join('');
      }

      // Render insights graphs using Chart.js
      function renderDatasetCharts(data) {
        // Gender Pie Chart
        const pieCtx = document.getElementById('genderPieChart').getContext('2d');
        if (charts.genderPie) charts.genderPie.destroy();
        charts.genderPie = new Chart(pieCtx, {
          type: 'doughnut',
          data: {
            labels: ['Male', 'Female'],
            datasets: [{
              data: [data.male_count, data.female_count],
              backgroundColor: ['#06b6d4', '#ec4899'],
              borderColor: 'rgba(255, 255, 255, 0.05)',
              borderWidth: 1,
              hoverOffset: 4
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: 'bottom',
                labels: {
                  color: '#9ca3af',
                  font: { family: 'Outfit', size: 12 }
                }
              }
            },
            cutout: '65%'
          }
        });

        // Name length bar chart
        const barCtx = document.getElementById('lengthBarChart').getContext('2d');
        if (charts.lengthBar) charts.lengthBar.destroy();
        
        // Gradient for bars
        const gradient = barCtx.createLinearGradient(0, 0, 0, 200);
        gradient.addColorStop(0, '#6366f1');
        gradient.addColorStop(1, '#06b6d4');

        charts.lengthBar = new Chart(barCtx, {
          type: 'bar',
          data: {
            labels: data.length_distribution.labels,
            datasets: [{
              label: 'Names Count',
              data: data.length_distribution.values,
              backgroundColor: gradient,
              borderRadius: 6,
              borderWidth: 0
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false }
            },
            scales: {
              x: {
                grid: { display: false },
                ticks: { color: '#9ca3af', font: { family: 'Outfit' } }
              },
              y: {
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                ticks: { color: '#9ca3af', font: { family: 'Outfit' } }
              }
            }
          }
        });
      }

      // Load Dataset Explorer items paginated
      async function loadDataset() {
        const searchVal = document.getElementById('dataset-search').value;
        const filterVal = document.getElementById('dataset-gender-filter').value;
        
        try {
          const res = await fetch(`/api/dataset?page=${datasetPage}&search=${encodeURIComponent(searchVal)}&gender=${filterVal}`);
          const data = await res.json();
          
          // Render table rows
          const tbody = document.getElementById('dataset-tbody');
          if (data.items.length === 0) {
            tbody.innerHTML = `<tr><td colspan="2" style="text-align: center; color: var(--text-muted); padding: 28px;">No matching records found.</td></tr>`;
          } else {
            tbody.innerHTML = data.items.map(item => `
              <tr>
                <td style="font-weight: 500;">${escapeHtml(item.name)}</td>
                <td>
                  <span class="badge ${item.gender === 'male' ? 'badge-male' : 'badge-female'}">
                    <i data-lucide="${item.gender === 'male' ? 'mars' : 'venus'}" style="width: 12px; height: 12px;"></i>
                    ${item.gender}
                  </span>
                </td>
              </tr>
            `).join('');
          }
          lucide.createIcons();

          // Update pagination controls
          const startIdx = data.total > 0 ? (data.page - 1) * data.per_page + 1 : 0;
          const endIdx = Math.min(data.page * data.per_page, data.total);
          
          document.getElementById('pagination-stats').innerText = `Showing ${startIdx} to ${endIdx} of ${data.total} names`;
          document.getElementById('btn-prev').disabled = data.page <= 1;
          document.getElementById('btn-next').disabled = data.page >= data.pages;

        } catch (err) {
          console.error("Failed to load dataset items:", err);
        }
      }

      function handleDatasetSearch() {
        clearTimeout(datasetSearchTimeout);
        datasetSearchTimeout = setTimeout(() => {
          datasetPage = 1;
          loadDataset();
        }, 300);
      }

      function handleDatasetFilter() {
        datasetPage = 1;
        loadDataset();
      }

      function datasetPrevPage() {
        if (datasetPage > 1) {
          datasetPage--;
          loadDataset();
        }
      }

      function datasetNextPage() {
        datasetPage++;
        loadDataset();
      }

      // Real-time name inference prediction playground
      function handleNameInput(value) {
        clearTimeout(predictDebounceTimeout);
        const name = value.trim();

        const defPanel = document.getElementById('predict-default');
        const loadPanel = document.getElementById('predict-loading');
        const resPanel = document.getElementById('predict-result');
        const explainPanel = document.getElementById('explanation-container');
        const interceptLabel = document.getElementById('intercept-val');

        if (!name) {
          defPanel.style.display = 'block';
          loadPanel.style.display = 'none';
          resPanel.style.display = 'none';
          explainPanel.innerHTML = `
            <div class="prediction-intro" style="margin: 40px auto; text-align: center;">
              <i data-lucide="help-circle" style="font-size: 24px; margin-bottom: 8px;"></i>
              <p>No name currently entered to explain.</p>
            </div>`;
          interceptLabel.style.display = 'none';
          lucide.createIcons();
          return;
        }

        // Show loading state
        defPanel.style.display = 'none';
        loadPanel.style.display = 'block';

        predictDebounceTimeout = setTimeout(async () => {
          try {
            const response = await fetch('/api/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ name: name })
            });
            const result = await response.json();

            loadPanel.style.display = 'none';

            if (result.error) {
              explainPanel.innerHTML = `<p style="color: var(--danger); text-align: center; margin: 20px;">${escapeHtml(result.error)}</p>`;
              return;
            }

            resPanel.style.display = 'block';
            
            // Set prediction badge label
            const labelEl = document.getElementById('predict-label');
            const isMale = result.prediction.toLowerCase() === 'male';
            labelEl.innerText = result.prediction;
            labelEl.className = `result-label ${isMale ? 'male' : 'female'}`;

            // Set progress bars
            const malePct = (result.probability.male * 100).toFixed(1);
            const femalePct = (result.probability.female * 100).toFixed(1);

            document.getElementById('prob-val-male').innerText = `${malePct}%`;
            document.getElementById('prob-fill-male').style.width = `${malePct}%`;
            document.getElementById('prob-val-female').innerText = `${femalePct}%`;
            document.getElementById('prob-fill-female').style.width = `${femalePct}%`;

            // Display split characters for highlighting
            const displayContainer = document.getElementById('char-highlight-display');
            displayContainer.innerHTML = name.split('').map((char, index) => 
              `<span id="char-${index}" class="name-char">${escapeHtml(char)}</span>`
            ).join('');

            // Render n-gram explanations
            interceptLabel.innerText = `Model Intercept: ${result.intercept.toFixed(4)}`;
            interceptLabel.style.display = 'block';

            if (result.ngrams.length === 0) {
              explainPanel.innerHTML = `
                <div class="prediction-intro" style="margin: 40px auto; text-align: center;">
                  <i data-lucide="alert-circle" style="font-size: 24px; margin-bottom: 8px;"></i>
                  <p>No active feature n-grams match this name in the vectorizer vocab.</p>
                </div>`;
            } else {
              explainPanel.innerHTML = result.ngrams.map(item => {
                const itemIsMale = item.weight > 0;
                return `
                  <div class="ngram-row" 
                       onmouseenter='highlightNgram(${JSON.stringify(item.occurrences)}, ${itemIsMale})'
                       onmouseleave='clearHighlight(${name.length})'>
                    <div class="ngram-left">
                      <span class="ngram-token">${escapeHtml(item.ngram)}</span>
                      <span class="ngram-badge ${itemIsMale ? 'male' : 'female'}">
                        ${itemIsMale ? 'Male' : 'Female'}
                      </span>
                      <span class="ngram-math">
                        coef ${item.weight.toFixed(3)} × count ${item.count}
                      </span>
                    </div>
                    <span class="ngram-weight ${itemIsMale ? 'positive' : 'negative'}">
                      ${itemIsMale ? '+' : ''}${item.contribution.toFixed(3)}
                    </span>
                  </div>
                `;
              }).join('');
            }
            lucide.createIcons();

          } catch (err) {
            console.error("Error invoking prediction API:", err);
            loadPanel.style.display = 'none';
          }
        }, 250);
      }

      // Feature highlight hovers
      function highlightNgram(occurrences, isMale) {
        if (!occurrences) return;
        occurrences.forEach(occ => {
          for (let i = occ.start; i < occ.end; i++) {
            const el = document.getElementById(`char-${i}`);
            if (el) {
              el.className = `name-char ${isMale ? 'highlight-male' : 'highlight-female'}`;
            }
          }
        });
      }

      function clearHighlight(len) {
        for (let i = 0; i < len; i++) {
          const el = document.getElementById(`char-${i}`);
          if (el) {
            el.className = 'name-char';
          }
        }
      }

      // Escape helper
      function escapeHtml(text) {
        const div = document.createElement('div');
        div.innerText = text;
        return div.innerHTML;
      }
    </script>
  </body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    reload_resources_if_needed()
    return render_template_string(PAGE_TEMPLATE)


@app.route("/api/stats", methods=["GET"])
def api_stats():
    reload_resources_if_needed()
    if "error" in stats_cache:
        return jsonify(stats_cache), 404
    return jsonify(stats_cache)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not MODEL_PATH.exists():
        return (
            jsonify({"error": "Classifier model file not found. Train it first."}),
            404,
        )

    body = request.get_json() or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name query cannot be empty."}), 400

    try:
        model = joblib.load(MODEL_PATH)
        # Class probabilities
        # classes: ['female', 'male']
        proba = model.predict_proba([name])[0]
        prob_female = float(proba[0])
        prob_male = float(proba[1])
        prediction = str(model.predict([name])[0]).title()

        # Retrieve pipeline objects
        vectorizer = model.named_steps["vectorizer"]
        classifier = model.named_steps["classifier"]
        feature_names = vectorizer.get_feature_names_out()
        coefs = classifier.coef_[0]
        intercept = float(classifier.intercept_[0])

        # Vectorize the query name
        vec_x = vectorizer.transform([name])
        active_indices = vec_x.nonzero()[1]

        ngrams_found = []
        for idx in active_indices:
            ngram = feature_names[idx]
            count = int(vec_x[0, idx])
            weight = float(coefs[idx])
            contribution = float(weight * count)

            # Find starting/ending ranges of occurrences in the input name
            lower_name = name.lower()
            occurrences = []
            start = 0
            while True:
                start = lower_name.find(ngram, start)
                if start == -1:
                    break
                occurrences.append({"start": start, "end": start + len(ngram)})
                start += 1

            ngrams_found.append(
                {
                    "ngram": ngram,
                    "count": count,
                    "weight": weight,
                    "contribution": contribution,
                    "occurrences": occurrences,
                }
            )

        # Sort by contribution strength (absolute weight) descending
        ngrams_found.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return jsonify(
            {
                "name": name,
                "prediction": prediction,
                "probability": {"female": prob_female, "male": prob_male},
                "intercept": intercept,
                "ngrams": ngrams_found,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Inference engine failure: {str(e)}"}), 500


@app.route("/api/dataset", methods=["GET"])
def api_dataset():
    if not DATA_PATH.exists():
        return jsonify({"error": "Dataset not found."}), 404

    search = request.args.get("search", "").strip().lower()
    gender = request.args.get("gender", "all").strip().lower()
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 15))

    # Read names database
    rows = []
    with DATA_PATH.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            name = (row.get("name") or "").strip()
            g = (row.get("gender") or "").strip().lower()

            if gender != "all" and g != gender:
                continue
            if search and search not in name.lower():
                continue

            rows.append({"name": name, "gender": g})

    total = len(rows)
    pages = math.ceil(total / per_page) if total > 0 else 1

    # Slice list for paginated results
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_items = rows[start_idx:end_idx]

    return jsonify(
        {
            "items": paginated_items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": pages,
        }
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

