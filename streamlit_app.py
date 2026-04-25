"""Professional Streamlit dashboard for baseline and adapted model comparison."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.image as mpimg
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.config.classes import CITYSCAPES_DETECTION_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
BASELINE_ROOT = RESULTS_ROOT / "source_baselines"
ADAPTATION_ROOT = RESULTS_ROOT / "adaptation"
COMPARISON_ROOT = RESULTS_ROOT / "comparison"
PLOTS_ROOT = RESULTS_ROOT / "plots"


def discover_models(model_root: Path, suffix: str) -> dict[str, Path]:
    """Collect available model weights from one result folder."""

    models: dict[str, Path] = {}
    for weight_path in sorted(model_root.glob(f"*{suffix}.pt")):
        label = weight_path.stem.replace(suffix, "").replace("_", " ").title()
        models[label] = weight_path
    return models


BASELINE_MODELS = discover_models(BASELINE_ROOT, "_best")
ADAPTED_MODELS = discover_models(ADAPTATION_ROOT, "_best")


@st.cache_data(show_spinner=False)
def load_baseline_metrics() -> pd.DataFrame:
    """Load stored baseline comparison metrics."""

    path = COMPARISON_ROOT / "baseline_comparison.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_adapted_metrics() -> pd.DataFrame:
    """Load stored baseline vs adapted comparison metrics."""

    path = COMPARISON_ROOT / "yolov5s_baseline_vs_adapted.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    """Load a model once per session."""

    return YOLO(model_path)


def inject_styles() -> None:
    """Apply custom dashboard styling."""

    st.markdown(
        """
        <style>
            :root {
                --panel: rgba(12, 18, 28, 0.78);
                --panel-strong: rgba(10, 16, 24, 0.92);
                --border: rgba(255,255,255,0.08);
                --muted: #aeb8c5;
                --text: #f6f8fb;
                --accent: #7dd3fc;
                --accent-2: #86efac;
                --warm: #fbbf24;
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(54, 119, 255, 0.22), transparent 26%),
                    radial-gradient(circle at top right, rgba(26, 188, 156, 0.18), transparent 24%),
                    radial-gradient(circle at bottom center, rgba(251, 191, 36, 0.08), transparent 28%),
                    linear-gradient(180deg, #08111d 0%, #0b1320 52%, #0d1522 100%);
                color: var(--text);
            }
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1400px;
            }
            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, rgba(9,16,28,0.98), rgba(11,18,31,0.98));
                border-right: 1px solid var(--border);
            }
            .hero {
                position: relative;
                overflow: hidden;
                padding: 1.5rem 1.55rem;
                border: 1px solid var(--border);
                background:
                    radial-gradient(circle at top right, rgba(125, 211, 252, 0.18), transparent 28%),
                    linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.025));
                border-radius: 24px;
                box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
                margin-bottom: 1.1rem;
            }
            .hero::after {
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(120deg, transparent 20%, rgba(255,255,255,0.03), transparent 72%);
                pointer-events: none;
            }
            .hero h1 {
                margin: 0;
                font-size: 2.35rem;
                letter-spacing: -0.03em;
            }
            .hero p {
                margin: 0.55rem 0 0 0;
                color: #d2d9e3;
                max-width: 72ch;
            }
            .hero-kicker {
                display: inline-block;
                margin-bottom: 0.8rem;
                padding: 0.3rem 0.75rem;
                border-radius: 999px;
                border: 1px solid rgba(125, 211, 252, 0.18);
                background: rgba(125, 211, 252, 0.10);
                color: #d9f5ff;
                font-size: 0.78rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                font-weight: 700;
            }
            .panel {
                border: 1px solid var(--border);
                background: var(--panel);
                border-radius: 22px;
                padding: 1rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.16);
                backdrop-filter: blur(8px);
            }
            .panel-title {
                margin: 0 0 0.9rem 0;
                font-size: 1.08rem;
                font-weight: 700;
                color: var(--text);
            }
            .image-shell {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(255,255,255,0.06);
                background: rgba(255,255,255,0.02);
            }
            .metric-card {
                border: 1px solid var(--border);
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.018));
                border-radius: 18px;
                padding: 0.95rem 1rem;
                min-height: 96px;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
            }
            .metric-label {
                font-size: 0.82rem;
                color: var(--muted);
                margin-bottom: 0.45rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .metric-value {
                font-size: 1.55rem;
                font-weight: 700;
                color: var(--text);
            }
            .section-card {
                border: 1px solid var(--border);
                background: var(--panel-strong);
                border-radius: 22px;
                padding: 1rem 1.05rem;
                box-shadow: 0 10px 28px rgba(0,0,0,0.14);
            }
            .small-note {
                color: #c7d0db;
                font-size: 0.92rem;
                line-height: 1.65;
            }
            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.75rem;
            }
            .badge {
                padding: 0.35rem 0.72rem;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 600;
                border: 1px solid rgba(255,255,255,0.08);
                background: rgba(255,255,255,0.045);
                color: #e9eef5;
            }
            .badge-accent {
                border-color: rgba(125, 211, 252, 0.2);
                background: rgba(125, 211, 252, 0.12);
            }
            .badge-success {
                border-color: rgba(134, 239, 172, 0.2);
                background: rgba(134, 239, 172, 0.12);
            }
            .badge-warm {
                border-color: rgba(251, 191, 36, 0.2);
                background: rgba(251, 191, 36, 0.12);
            }
            .eyebrow {
                color: var(--accent);
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
                margin-bottom: 0.4rem;
            }
            .summary-banner {
                border: 1px solid var(--border);
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
                padding: 0.95rem 1.05rem;
                margin: 0.5rem 0 1rem 0;
            }
            .caption-text {
                color: var(--muted);
                font-size: 0.9rem;
            }
            div[data-testid="stDataFrame"] {
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid rgba(255,255,255,0.06);
            }
            .stAlert {
                border-radius: 16px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> None:
    """Render one metric card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_open(title: str, eyebrow: str | None = None) -> None:
    """Open a reusable styled panel."""

    eyebrow_html = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(
        f"""
        <div class="panel">
            {eyebrow_html}
            <div class="panel-title">{title}</div>
        """,
        unsafe_allow_html=True,
    )


def panel_close() -> None:
    """Close a reusable styled panel."""

    st.markdown("</div>", unsafe_allow_html=True)


def render_badges(items: list[tuple[str, str]]) -> None:
    """Render small styled badge chips."""

    if not items:
        return
    badge_html = "".join(
        f'<span class="badge {variant}">{label}</span>' for label, variant in items
    )
    st.markdown(f'<div class="badge-row">{badge_html}</div>', unsafe_allow_html=True)


def predict(model_path: Path, image_path: Path, confidence: float):
    """Run one model prediction."""

    model = load_model(str(model_path))
    return model.predict(source=str(image_path), conf=confidence, verbose=False)[0]


def detection_count(result) -> int:
    """Return number of detected boxes."""

    if result.boxes is None:
        return 0
    return len(result.boxes)


def average_confidence(result) -> float:
    """Return mean prediction confidence."""

    if result.boxes is None or len(result.boxes) == 0:
        return 0.0
    return float(result.boxes.conf.mean().item())


def unique_class_count(result) -> int:
    """Return number of distinct predicted classes."""

    if result.boxes is None or len(result.boxes) == 0:
        return 0
    return len({int(class_id) for class_id in result.boxes.cls.tolist()})


def predicted_class_names(result) -> list[str]:
    """Return sorted class names predicted in one result."""

    if result.boxes is None or len(result.boxes) == 0:
        return []
    class_ids = sorted({int(class_id) for class_id in result.boxes.cls.tolist()})
    return [CITYSCAPES_DETECTION_CLASSES[class_id] for class_id in class_ids]


def summarize_result(result) -> pd.DataFrame:
    """Summarize detections by class for display."""

    if result.boxes is None or len(result.boxes) == 0:
        return pd.DataFrame(
            [{"class": "No detections", "count": 0, "avg_confidence": 0.0}]
        )

    rows = []
    grouped: dict[int, list[float]] = {}
    for class_id, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        grouped.setdefault(int(class_id), []).append(float(conf))

    for class_id, scores in sorted(grouped.items()):
        rows.append(
            {
                "class": CITYSCAPES_DETECTION_CLASSES[class_id],
                "count": len(scores),
                "avg_confidence": round(sum(scores) / len(scores), 3),
            }
        )
    return pd.DataFrame(rows)


def render_result_image(result):
    """Convert plotted result to RGB image."""

    plotted = result.plot()
    return plotted[:, :, ::-1]


def lookup_baseline_metric_row(model_name: str) -> pd.Series | None:
    """Find stored baseline metric row by model name."""

    df = load_baseline_metrics()
    if df.empty:
        return None
    match = df[df["Model"].str.lower() == model_name.lower().replace(" ", "")]
    if match.empty:
        return None
    return match.iloc[0]


def lookup_adapted_metric_row(model_name: str) -> pd.Series | None:
    """Find stored adapted metric row by selected adapted model name."""

    df = load_adapted_metrics()
    if df.empty:
        return None
    lowered = model_name.lower()
    if "yolov5s" in lowered:
        match = df[df["Setting"].str.lower() == "adapted"]
        if not match.empty:
            return match.iloc[0]
    return None


def show_stored_metrics(title: str, row: pd.Series | None) -> None:
    """Render stored evaluation metrics from CSV results."""

    st.markdown(f"**{title}**")
    if row is None:
        st.info("Stored evaluation metrics are not available for this model.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Precision", f"{row['Precision']:.3f}")
    with c2:
        metric_card("Recall", f"{row['Recall']:.3f}")
    with c3:
        metric_card("mAP50", f"{row['mAP50']:.3f}")
    with c4:
        metric_card("mAP50-95", f"{row['mAP50_95']:.3f}")


def show_prediction_panel(title: str, result, metric_row: pd.Series | None) -> None:
    """Render one prediction panel with prediction and evaluation metrics."""

    panel_open(title, "Model Output")
    st.image(render_result_image(result), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Detections", str(detection_count(result)))
    with c2:
        metric_card("Avg Confidence", f"{average_confidence(result):.3f}")
    with c3:
        metric_card("Predicted Classes", str(unique_class_count(result)))

    if predicted_class_names(result):
        render_badges([(name, "badge-accent") for name in predicted_class_names(result)])
    else:
        st.caption("Classes: none")

    show_stored_metrics("Stored Evaluation Metrics", metric_row)
    st.dataframe(summarize_result(result), use_container_width=True, hide_index=True)
    panel_close()


def render_project_dashboard() -> None:
    """Render the second page with project-wide metrics and plots."""

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Experiment Overview</div>
            <h1>Project Dashboard</h1>
            <p>Review baseline rankings, pilot adaptation outcome, and stored experiment plots.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    baseline_df = load_baseline_metrics()
    adapted_df = load_adapted_metrics()

    if not baseline_df.empty:
        leader = baseline_df.sort_values("mAP50_95", ascending=False).iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Best Baseline", str(leader["Model"]))
        with c2:
            metric_card("Best mAP50", f"{leader['mAP50']:.3f}")
        with c3:
            metric_card("Best mAP50-95", f"{leader['mAP50_95']:.3f}")

    col1, col2 = st.columns([1.15, 0.85])
    with col1:
        panel_open("Baseline Ranking", "Performance")
        if baseline_df.empty:
            st.info("Baseline comparison CSV not found.")
        else:
            st.dataframe(
                baseline_df.sort_values("mAP50_95", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        panel_close()

        panel_open("Baseline vs Adapted", "Comparison")
        if adapted_df.empty:
            st.info("Adaptation comparison CSV not found.")
        else:
            st.dataframe(adapted_df, use_container_width=True, hide_index=True)
        panel_close()

    with col2:
        panel_open("Experiment Notes", "Summary")
        st.markdown(
            """
            <div class="small-note">
                Best baseline model: <b>YOLOv5s</b><br><br>
                Compared models: YOLOv8n, YOLOv11n, YOLOv5s<br><br>
                Pilot adaptation: YOLOv5s on Frankfurt pseudo-labels<br><br>
                Adaptation outcome: no improvement over the baseline in this setup
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_badges(
            [
                ("3 Baselines", "badge-accent"),
                ("1 Adaptation Trial", "badge-success"),
                ("Foggy Evaluation", "badge-warm"),
            ]
        )
        panel_close()

    st.markdown(
        """
        <div class="summary-banner">
            <strong>Stored Plots</strong><br>
            <span class="caption-text">These charts are loaded from the exported experiment results and provide the visual record of baseline performance and the pilot adaptation outcome.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    plot_paths = [
        COMPARISON_ROOT / "baseline_comparison_combined.png",
        COMPARISON_ROOT / "yolov5s_baseline_vs_adapted.png",
        PLOTS_ROOT / "yolov8n_results.png",
        PLOTS_ROOT / "yolov11n_results.png",
        PLOTS_ROOT / "yolov5s_results.png",
        PLOTS_ROOT / "adapted_yolov5s_frankfurt_results.png",
    ]
    plot_paths = [path for path in plot_paths if path.exists()]

    if not plot_paths:
        st.info("No plots found in the results folders.")
    else:
        for left_path, right_path in zip(plot_paths[::2], plot_paths[1::2]):
            left_col, right_col = st.columns(2)
            with left_col:
                panel_open(left_path.name, "Plot")
                st.image(mpimg.imread(left_path), use_container_width=True)
                panel_close()
            with right_col:
                panel_open(right_path.name, "Plot")
                st.image(mpimg.imread(right_path), use_container_width=True)
                panel_close()
        if len(plot_paths) % 2 == 1:
            panel_open(plot_paths[-1].name, "Plot")
            st.image(mpimg.imread(plot_paths[-1]), use_container_width=True)
            panel_close()


def render_image_comparator() -> None:
    """Render the one-page baseline vs adapted comparison view."""

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Real-Time Inference</div>
            <h1>Baseline vs Adapted Image Comparator</h1>
            <p>Upload one image, run a baseline model and an adapted model, and compare predictions and stored evaluation metrics on the same page.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not BASELINE_MODELS:
        st.error("No baseline model weights were found in results/source_baselines.")
        st.stop()

    if not ADAPTED_MODELS:
        st.warning("No adapted model weights were found in results/adaptation.")

    with st.sidebar:
        st.header("Inference Controls")
        baseline_name = st.selectbox("Baseline model", list(BASELINE_MODELS))
        adapted_options = ["None"] + list(ADAPTED_MODELS)
        adapted_name = st.selectbox("Adapted model", adapted_options)
        confidence = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
        )

    if uploaded_file is None:
        st.info("Upload an image from the sidebar to start the comparison.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    suffix = Path(uploaded_file.name).suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        image.save(temp_file.name)
        temp_image_path = Path(temp_file.name)

    baseline_result = predict(BASELINE_MODELS[baseline_name], temp_image_path, confidence)
    adapted_result = (
        predict(ADAPTED_MODELS[adapted_name], temp_image_path, confidence)
        if adapted_name != "None"
        else None
    )

    st.markdown(
        """
        <div class="summary-banner">
            <strong>Comparison Mode Active</strong><br>
            <span class="caption-text">The uploaded image is being evaluated with the selected baseline and adapted weights at the same confidence threshold so the visual comparison remains fair.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    quick1, quick2, quick3, quick4 = st.columns(4)
    with quick1:
        metric_card("Baseline Model", baseline_name)
    with quick2:
        metric_card("Adapted Model", adapted_name if adapted_name != "None" else "Not selected")
    with quick3:
        metric_card("Threshold", f"{confidence:.2f}")
    with quick4:
        metric_card("Image Format", suffix.replace(".", "").upper())

    original_col, baseline_col, adapted_col = st.columns([0.9, 1.05, 1.05], gap="large")

    with original_col:
        panel_open("Original Image", "Input")
        st.image(image, use_container_width=True)
        st.caption(uploaded_file.name)
        panel_close()

    with baseline_col:
        show_prediction_panel(
            f"Baseline: {baseline_name}",
            baseline_result,
            lookup_baseline_metric_row(baseline_name),
        )

    with adapted_col:
        if adapted_result is None:
            st.markdown("### Adapted Model")
            st.info("Select an adapted model to compare.")
        else:
            show_prediction_panel(
                f"Adapted: {adapted_name}",
                adapted_result,
                lookup_adapted_metric_row(adapted_name),
            )

    st.divider()
    panel_open("Comparison Summary", "Output Review")

    comparison_rows = [
        {
            "model_type": "Baseline",
            "model_name": baseline_name,
            "detections": detection_count(baseline_result),
            "avg_confidence": round(average_confidence(baseline_result), 3),
            "predicted_classes": ", ".join(predicted_class_names(baseline_result)) or "none",
        }
    ]
    if adapted_result is not None:
        comparison_rows.append(
            {
                "model_type": "Adapted",
                "model_name": adapted_name,
                "detections": detection_count(adapted_result),
                "avg_confidence": round(average_confidence(adapted_result), 3),
                "predicted_classes": ", ".join(predicted_class_names(adapted_result)) or "none",
            }
        )

    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
    panel_close()


def main() -> None:
    """Render the Streamlit app."""

    st.set_page_config(
        page_title="Object Detection Dashboard",
        page_icon=":camera:",
        layout="wide",
    )
    inject_styles()

    with st.sidebar:
        page = st.radio("Page", ["Image Comparator", "Project Dashboard"])

    if page == "Project Dashboard":
        render_project_dashboard()
    else:
        render_image_comparator()


if __name__ == "__main__":
    main()
