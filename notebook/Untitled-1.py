#!/usr/bin/env python3
"""
plot_model_performance.py

Creates separate Train/Test tables (CSV + Markdown) and visual charts comparing three models
(RandomForest, KNeighbors, XGB) across metrics (Accuracy, F1, Precision, Recall, ROC AUC).

Usage:
    python plot_model_performance.py

Outputs:
    - train_table.csv
    - test_table.csv
    - train_table.md
    - test_table.md
    - plots/train_vs_models.png
    - plots/test_vs_models.png
    - plots/model_train_vs_test_[model].png  (one per model)
Dependencies:
    pip install pandas matplotlib seaborn
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data (from your provided summary)
train_data = {
    "Model": [
        "Random Forest Classifier",
        "KNeighborsClassifier",
        "XGBClassifier",
    ],
    "Accuracy": [1.0000, 1.0000, 0.9995],
    "F1 Score": [1.0000, 1.0000, 0.9995],
    "Precision": [1.0000, 1.0000, 0.9993],
    "Recall": [1.0000, 1.0000, 0.9997],
    "ROC AUC": [1.0000, 1.0000, 0.9995],
}

test_data = {
    "Model": [
        "Random Forest Classifier",
        "KNeighborsClassifier",
        "XGBClassifier",
    ],
    "Accuracy": [0.9572, 0.9733, 0.9455],
    "F1 Score": [0.9606, 0.9757, 0.9498],
    "Precision": [0.9590, 0.9651, 0.9487],
    "Recall": [0.9621, 0.9865, 0.9508],
    "ROC AUC": [0.9567, 0.9721, 0.9450],
}

train_df = pd.DataFrame(train_data).set_index("Model")
test_df = pd.DataFrame(test_data).set_index("Model")

# Create output directories
os.makedirs("plots", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# Save tables as CSV and Markdown
train_df.to_csv("tables/train_table.csv")
test_df.to_csv("tables/test_table.csv")

# Save markdown versions
def df_to_markdown(df, path, title):
    md = f"## {title}\n\n" + df.reset_index().to_markdown(index=False)
    with open(path, "w") as f:
        f.write(md)

df_to_markdown(train_df, "tables/train_table.md", "Train set")
df_to_markdown(test_df, "tables/test_table.md", "Test set")

sns.set_theme(style="whitegrid")

# Helper to melt and plot
def plot_metrics(df, title, outpath):
    df_long = df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_long, x="Metric", y="Score", hue="Model")
    ax.set_ylim(0.8, 1.01)  # appropriate for these scores; change if needed
    ax.set_title(title)
    plt.xticks(rotation=15)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# Plot Train and Test charts (models compared across metrics)
plot_metrics(train_df, "Train set — model comparison across metrics", "plots/train_vs_models.png")
plot_metrics(test_df, "Test set — model comparison across metrics", "plots/test_vs_models.png")

# For each model, plot Train vs Test across metrics
for model in train_df.index:
    df_model = pd.DataFrame({
        "Dataset": ["Train"] * len(train_df.columns) + ["Test"] * len(test_df.columns),
        "Metric": list(train_df.columns) + list(test_df.columns),
        "Score": list(train_df.loc[model].values) + list(test_df.loc[model].values),
    })
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=df_model, x="Metric", y="Score", hue="Dataset")
    ax.set_ylim(0.8, 1.01)
    ax.set_title(f"{model} — Train vs Test by metric")
    plt.xticks(rotation=15)
    plt.legend(title="Dataset")
    plt.tight_layout()
    safe_name = model.replace(" ", "_").replace("/", "_")
    out = f"plots/model_train_vs_test_{safe_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()

print("Done. Files created:")
print("- tables/train_table.csv, tables/test_table.csv")
print("- tables/train_table.md, tables/test_table.md")
print("- plots/train_vs_models.png, plots/test_vs_models.png")
print("- plots/model_train_vs_test_<model>.png (one per model)")
"""
Notes / next steps:
- You can tweak y-limits, color palettes, or plot types (heatmap, radar) inside this script.
- If you'd like a single combined figure (e.g., grouped bars per metric with Train/Test side-by-side for each model),
  I can update the script to produce that layout.
"""