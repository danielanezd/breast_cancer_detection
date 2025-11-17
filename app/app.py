"""Streamlit UI that uploads a mammogram and forwards it to MLflow endpoints."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

import streamlit as st

from api.client import predict_all_models, predict_single_model
from config import APP_TITLE, MAX_FILE_SIZE_MB, MODEL_ENDPOINTS, SUPPORTED_FORMATS


st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="centered")

st.title(APP_TITLE)
st.write(
    "Provide a PNG or DICOM mammography image and forward it to the hosted MLflow "
    "models for prediction."
)

st.markdown(
    """
### Instructions
1. Click **Browse files** and pick a PNG or DICOM scan.
2. Choose to query a single model or broadcast to every deployed model.
3. Hit **Run inference** to send the payload to the MLflow endpoints.
"""
)

mode = st.radio("Prediction mode", ["All models", "Single model"], horizontal=True)
selected_model = None
if mode == "Single model":
    selected_model = st.selectbox("Target model", sorted(MODEL_ENDPOINTS.keys()))

uploaded_file = st.file_uploader(
    "Select an image from your computer",
    type=SUPPORTED_FORMATS,
    help=f"Accepted formats: {', '.join(SUPPORTED_FORMATS)} · Max size "
    f"{MAX_FILE_SIZE_MB} MB",
)

if uploaded_file:
    st.caption(f"Selected file: `{uploaded_file.name}`")

status_placeholder = st.empty()
results_placeholder = st.empty()

def _validate_and_prepare(uploaded) -> tuple[str, str] | None:
    """Validate file size/type and return (b64_string, extension)."""
    file_size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File is {file_size_mb:.1f} MB, exceeding {MAX_FILE_SIZE_MB} MB limit.")
        return None

    suffix = Path(uploaded.name).suffix.lower()
    if suffix.startswith("."):
        suffix = suffix[1:]
    if suffix not in SUPPORTED_FORMATS:
        st.error(f"Unsupported format '{suffix}'. Allowed: {', '.join(SUPPORTED_FORMATS)}")
        return None

    encoded = base64.b64encode(uploaded.getvalue()).decode("utf-8")
    return encoded, f".{suffix}"


def _render_prediction_results(predictions: dict[str, dict]) -> None:
    """Render model responses, including Grad-CAM imagery if provided."""
    results_container = results_placeholder.container()
    for model_name, payload in predictions.items():
        with results_container.expander(model_name, expanded=True):
            error = payload.get("error")
            if error:
                st.error(error)
                continue

            st.write(
                f"**Prediction:** `{payload.get('predicted_class_name', 'N/A')}` "
                f"(class {payload.get('predicted_class')})"
            )
            confidence = payload.get("confidence")
            if confidence is not None:
                st.write(f"**Confidence:** {confidence:.2%}")

            diagnosis = payload.get("diagnosis")
            if diagnosis:
                st.info(diagnosis)

            probabilities = payload.get("probabilities")
            if probabilities:
                st.write("**Class probabilities**")
                for label, prob in probabilities.items():
                    st.write(f"- {label}: {prob:.2%}")

            for key, caption in (
                ("gradcam_overlay", "Grad-CAM Overlay"),
                ("gradcam_heatmap", "Grad-CAM Heatmap"),
            ):
                image_b64 = payload.get(key)
                if image_b64:
                    try:
                        image_bytes = base64.b64decode(image_b64)
                        st.image(image_bytes, caption=caption, use_container_width=True)
                    except binascii.Error:
                        st.warning(f"Could not decode {caption.lower()} image.")


if st.button("Run inference", disabled=uploaded_file is None):
    if not uploaded_file:
        status_placeholder.warning("Please choose an image first.")
    else:
        prepared = _validate_and_prepare(uploaded_file)
        if prepared:
            img_b64, file_type = prepared
            with status_placeholder, st.spinner("Contacting MLflow endpoints..."):
                if mode == "All models":
                    predictions = predict_all_models(img_b64, file_type)
                else:
                    predictions = {
                        selected_model: predict_single_model(
                            img_b64, file_type, selected_model
                        )
                    }
            status_placeholder.success("Inference complete.")
            _render_prediction_results(predictions)
