"""
Train gender, age, and disease models in parallel; save artifacts; parallel inference.
Prints test metrics, ASCII bar chart, and confusion matrix tables plus heatmaps (Windows-safe).
Also saves PNG graphs under plot_outputs/ so you can open them in VS Code (Explorer click).
Optional: pip install seaborn adds normalized confusion-matrix heatmaps and a feature-correlation heatmap.
"""
from __future__ import annotations

import concurrent.futures
import warnings
from pathlib import Path

import joblib
import matplotlib


def _in_ipython() -> bool:
    """True in Jupyter / Google Colab — plots should show under the cell."""
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


# CLI (`python model.py`): Agg writes PNG only, no GUI. Notebook/Colab: default backend = inline display.
if not _in_ipython():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _save_display_close(fig: plt.Figure, path: Path) -> None:
    """Write PNG to disk; in Jupyter/Colab also show the figure under the cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    if _in_ipython():
        try:
            from IPython.display import display

            display(fig)
        except Exception:
            plt.show()
    plt.close(fig)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

FEATURES = [
    "sensor1_fever",
    "sensor2_cough",
    "sensor3_sneeze",
    "sensor4_weakness",
    "migraine_pain",
]


def predict_parallel(
    sensor_values,
    features: list[str],
    model_gender: RandomForestClassifier,
    model_age: RandomForestRegressor,
    model_disease: RandomForestClassifier,
    le_gender: LabelEncoder,
) -> dict:
    """Run gender, age, and disease inference in parallel using three worker threads."""
    X_in = pd.DataFrame(sensor_values, columns=features)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        fut_g = executor.submit(model_gender.predict, X_in)
        fut_a = executor.submit(model_age.predict, X_in)
        fut_d = executor.submit(model_disease.predict_proba, X_in)

        gender = le_gender.inverse_transform(fut_g.result())[0]
        age = round(float(fut_a.result()[0]))
        probs = fut_d.result()[0]
        disease_names = model_disease.classes_
        disease_percent = {
            str(name): round(float(p) * 100, 2) for name, p in zip(disease_names, probs)
        }
        top_disease = max(disease_percent, key=disease_percent.get)

    return {
        "Gender": gender,
        "Predicted Age": age,
        "Top Disease": top_disease,
        "All Diseases %": disease_percent,
    }


def _print_confusion_matrix_text(cm: list, labels: list[str], title: str) -> None:
    """Print confusion matrix as a table (works on all consoles)."""
    w = max(len(str(x)) for row in cm for x in row)
    col_w = max(w, 6, max(len(str(x)) for x in labels) + 1)
    print("")
    print(title)
    print(" " * 14 + "Predicted ->")
    header = "".join(str(l)[: col_w - 1].ljust(col_w) for l in labels)
    print("True".ljust(14) + header)
    print("-" * (14 + col_w * len(labels)))
    for i, row_label in enumerate(labels):
        line = str(row_label)[:12].ljust(14) + "|"
        for j, v in enumerate(cm[i]):
            line += str(v).ljust(col_w)
        print(line)


def _ascii_bar_chart(title: str, items: list[tuple[str, float]], max_bar: int = 50) -> None:
    """Horizontal bar chart using only ASCII (safe on cp1252 Windows consoles)."""
    print("")
    print(title)
    if not items:
        return
    vmax = max(v for _, v in items)
    if vmax <= 0:
        vmax = 1.0
    for name, val in items:
        n = int(round((val / vmax) * max_bar))
        bar = "#" * n
        print("  {:16} |{:s} {:.2f}".format(name[:16], bar, val))


def _plot_accuracies_bar(gender_acc: float, disease_acc: float, age_r2: float) -> None:
    items = [
        ("Gender acc %", gender_acc * 100.0),
        ("Disease acc %", disease_acc * 100.0),
        ("Age R2 x 100", age_r2 * 100.0),
    ]
    _ascii_bar_chart(
        "Test set scores (ASCII bar chart). Gender and Disease = accuracy. Age = R-squared x 100.",
        items,
    )


def _save_figures_to_disk(
    out_dir: Path,
    gender_acc: float,
    disease_acc: float,
    age_r2: float,
    cm_d: np.ndarray,
    disease_labels: list,
    cm_g: np.ndarray,
    g_labels: list,
) -> list[Path]:
    """Save bar chart and confusion matrices as PNG files for VS Code / reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    names = ["Gender acc %", "Disease acc %", "Age R2 x 100"]
    vals = [gender_acc * 100.0, disease_acc * 100.0, age_r2 * 100.0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, vals, color=["#2ecc71", "#3498db", "#9b59b6"])
    ax.set_ylabel("Percent")
    ax.set_title("Test set scores (Gender & Disease = accuracy; Age = R-squared x 100)")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    p1 = out_dir / "metrics_bar_chart.png"
    _save_display_close(fig, p1)
    saved.append(p1)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_d, display_labels=disease_labels)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=True)
    ax.set_title("Disease confusion matrix (test set)")
    fig.tight_layout()
    p2 = out_dir / "disease_confusion_matrix.png"
    _save_display_close(fig, p2)
    saved.append(p2)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp_g = ConfusionMatrixDisplay(confusion_matrix=cm_g, display_labels=g_labels)
    disp_g.plot(ax=ax, cmap="Greens", colorbar=True)
    ax.set_title("Gender confusion matrix (test set)")
    fig.tight_layout()
    p3 = out_dir / "gender_confusion_matrix.png"
    _save_display_close(fig, p3)
    saved.append(p3)

    return saved


def _row_normalize_cm(cm: np.ndarray) -> np.ndarray:
    """Normalize each row to sum to 1 (recall per true class)."""
    rs = cm.sum(axis=1, keepdims=True)
    out = np.zeros_like(cm, dtype=float)
    np.divide(cm.astype(float), rs, out=out, where=rs != 0)
    return out


def _save_seaborn_figures(
    out_dir: Path,
    cm_d: np.ndarray,
    disease_labels: list,
    cm_g: np.ndarray,
    g_labels: list,
    feature_frame: pd.DataFrame,
) -> list[Path]:
    """Extra PNGs: row-normalized confusion matrices + sensor correlation (needs seaborn)."""
    try:
        import seaborn as sns
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    sns.set_theme(style="whitegrid", font_scale=0.85)

    cm_dn = _row_normalize_cm(cm_d)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_dn,
        xticklabels=disease_labels,
        yticklabels=disease_labels,
        annot=True,
        fmt=".2f",
        cmap="rocket",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Disease confusion matrix (row-normalized = recall per true class)")
    fig.tight_layout()
    p4 = out_dir / "disease_confusion_matrix_normalized.png"
    _save_display_close(fig, p4)
    saved.append(p4)

    cm_gn = _row_normalize_cm(cm_g)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_gn,
        xticklabels=g_labels,
        yticklabels=g_labels,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Gender confusion matrix (row-normalized)")
    fig.tight_layout()
    p5 = out_dir / "gender_confusion_matrix_normalized.png"
    _save_display_close(fig, p5)
    saved.append(p5)

    corr = feature_frame.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Feature correlation (sensors + migraine)")
    fig.tight_layout()
    p6 = out_dir / "feature_correlation_heatmap.png"
    _save_display_close(fig, p6)
    saved.append(p6)

    return saved


def _heatmap_ascii(cm: list[list[int]], labels: list[str], title: str) -> None:
    """Grayscale heatmap using only ASCII (darker = more samples in cell)."""
    print("")
    print(title)
    flat = [v for row in cm for v in row]
    mval = max(flat) if flat else 1
    chars = " .:-=+*#%@"
    col_head = "".join(str(x)[:7].ljust(8) for x in labels)
    print(" " * 12 + col_head)
    for i, row_label in enumerate(labels):
        line = str(row_label)[:10].ljust(11) + "|"
        for v in cm[i]:
            idx = int(round((v / mval) * (len(chars) - 1))) if mval else 0
            line += chars[idx] * 3 + " "
        print(line)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "health_dataset_20000.csv"

    print("Loading dataset:", csv_path.name)
    df = pd.read_csv(csv_path)

    X = df[FEATURES]
    y_gender = df["gender"]
    y_age = df["age"]
    y_disease = df["disease"]

    le_gender = LabelEncoder()
    y_gender_enc = le_gender.fit_transform(y_gender)

    X_train, X_test, yg_train, yg_test, ya_train, ya_test, yd_train, yd_test = train_test_split(
        X,
        y_gender_enc,
        y_age,
        y_disease,
        test_size=0.2,
        random_state=42,
    )

    model_gender = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=42
    )
    model_age = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model_disease = RandomForestClassifier(
        n_estimators=150, n_jobs=-1, random_state=42
    )

    print("")
    print("Training three models in parallel (ThreadPoolExecutor, max_workers=3).")
    print("Each model uses n_jobs=-1 inside RandomForest.")
    print("")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(model_gender.fit, X_train, yg_train)
        f2 = executor.submit(model_age.fit, X_train, ya_train)
        f3 = executor.submit(model_disease.fit, X_train, yd_train)
        f1.result()
        f2.result()
        f3.result()

    print("All models trained!")
    print("")

    yg_pred = model_gender.predict(X_test)
    yd_pred = model_disease.predict(X_test)
    ya_pred = model_age.predict(X_test)

    gender_acc = accuracy_score(yg_test, yg_pred)
    disease_acc = accuracy_score(yd_test, yd_pred)
    age_r2 = r2_score(ya_test, ya_pred)
    age_mae = mean_absolute_error(ya_test, ya_pred)

    print("=== Test set scores (same split, random_state=42) ===")
    print("Gender classification accuracy: {:.2f}%".format(gender_acc * 100))
    print("Disease classification accuracy: {:.2f}%".format(disease_acc * 100))
    print("Age regression R-squared: {:.4f} (variance explained)".format(age_r2))
    print("Age regression MAE (years): {:.2f}".format(age_mae))
    print("")

    print("Terminal graphs: ASCII bar chart and confusion matrix heatmaps.")
    _plot_accuracies_bar(gender_acc, disease_acc, age_r2)

    disease_labels = sorted(yd_test.unique())
    cm_d = confusion_matrix(yd_test, yd_pred, labels=disease_labels)
    _print_confusion_matrix_text(
        cm_d.tolist(), disease_labels, "Disease confusion matrix (counts, test set)"
    )
    _heatmap_ascii(
        cm_d.tolist(),
        disease_labels,
        "Disease confusion matrix (ASCII heatmap; darker = more counts)",
    )

    g_labels = sorted(y_gender.unique())
    cm_g = confusion_matrix(
        le_gender.inverse_transform(yg_test),
        le_gender.inverse_transform(yg_pred),
        labels=g_labels,
    )
    _print_confusion_matrix_text(
        cm_g.tolist(), g_labels, "Gender confusion matrix (counts, test set)"
    )
    _heatmap_ascii(
        cm_g.tolist(),
        g_labels,
        "Gender confusion matrix (ASCII heatmap; darker = more counts)",
    )

    plot_dir = base_dir / "plot_outputs"
    try:
        png_paths = _save_figures_to_disk(
            plot_dir,
            gender_acc,
            disease_acc,
            age_r2,
            cm_d,
            disease_labels,
            cm_g,
            g_labels,
        )
        extra = _save_seaborn_figures(
            plot_dir,
            cm_d,
            disease_labels,
            cm_g,
            g_labels,
            X,
        )
        png_paths.extend(extra)
        print("")
        print("PNG graphs saved (open in VS Code: click file in Explorer, or Preview):")
        for p in png_paths:
            print("  -", p)
        if not extra:
            print(
                "  (Install seaborn for 3 extra PNGs: normalized matrices + correlation — pip install seaborn)"
            )
    except Exception as exc:
        print("")
        print("Could not save PNG graphs (install matplotlib: pip install matplotlib).")
        print("Reason:", exc)

    paths = {
        "model_gender": base_dir / "model_gender.pkl",
        "model_age": base_dir / "model_age.pkl",
        "model_disease": base_dir / "model_disease.pkl",
        "gender_encoder": base_dir / "gender_encoder.pkl",
    }
    joblib.dump(model_gender, paths["model_gender"])
    joblib.dump(model_age, paths["model_age"])
    joblib.dump(model_disease, paths["model_disease"])
    joblib.dump(le_gender, paths["gender_encoder"])

    print("")
    print("Models saved to folder:", str(base_dir))
    for label, p in paths.items():
        print("  - {}: {}".format(label, p.name))
    print("")

    example = [[4.8, 1.1, 0.4, 4.3, 7.2]]
    print("Parallel prediction example (Typhoid-like sensor values):")
    print("  Input row:", example[0])
    print("")

    result = predict_parallel(
        example,
        FEATURES,
        model_gender,
        model_age,
        model_disease,
        le_gender,
    )

    print("Prediction result:")
    print("  Gender:", result["Gender"])
    print("  Predicted Age:", result["Predicted Age"])
    print("  Top Disease:", result["Top Disease"])
    print("  All disease probabilities (%):")
    for name in sorted(result["All Diseases %"], key=result["All Diseases %"].get, reverse=True):
        print("    - {}: {}".format(name, result["All Diseases %"][name]))


if __name__ == "__main__":
    main()
