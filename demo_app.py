# -*- coding: utf-8 -*-
r"""Room-cleanliness PROTOTYPE demo (Streamlit).

Take or upload a room photo -> CLEAN / NOT_CLEAN + probability.

Uses the already-validated approach unchanged:
  frozen ResNet-50 (IMAGENET1K_V2) -> 384 letterbox, EXIF corrected
  -> whole + q1..q4 spatial views -> tile2x2_concat (10240-d)
  -> StandardScaler -> class-weighted LogisticRegression

Run:  streamlit run demo_app.py      (from the repository root)
"""
from __future__ import annotations

import os
import sys

import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "prep"))

from demo_inference import load_artifacts, load_backbone, predict  # noqa: E402

st.set_page_config(page_title="Room Cleanliness Prototype", page_icon="broom", layout="wide")


@st.cache_resource(show_spinner="Loading frozen ResNet-50 and demo model...")
def _load():
    bundle, meta = load_artifacts()
    return bundle, meta, load_backbone()


st.title("Room Cleanliness Classifier")

try:
    bundle, meta, backbone = _load()
except FileNotFoundError as e:
    st.exception(e)
    st.stop()

with st.sidebar:
    st.header("Demo settings")
    thr = st.slider(
        "Decision threshold (p ≥ threshold → NOT_CLEAN)",
        0.01, 0.95, float(bundle["default_threshold"]), 0.01,
        help="Default is the median of the five thresholds selected during "
             "leave-one-property-out evaluation. Lower = catches more dirty rooms "
             "and raises more false alarms.",
    )
    st.caption(f"LOPO fold thresholds: {meta['threshold']['lopo_fold_thresholds']}")

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. EXIF orientation correction\n"
        "2. 5 deterministic views: whole image + 2×2 quadrants (10% overlap)\n"
        "3. Each view → 384×384 letterbox → **frozen** ResNet-50\n"
        "4. Concatenate → 10,240-d feature\n"
        "5. StandardScaler → class-weighted logistic regression"
    )

    st.divider()
    st.subheader("Validation evidence")
    v = meta["validation_evidence"]
    st.markdown(
        f"Leave-one-property-out over 136 images / 5 properties:\n\n"
        f"- Accuracy **{v['lopo_accuracy']:.1%}**\n"
        f"- Balanced accuracy **{v['lopo_balanced_accuracy']:.1%}**\n"
        f"- NOT_CLEAN recall **{v['lopo_notclean_recall']:.1%}**\n"
        f"- NOT_CLEAN precision **{v['lopo_notclean_precision']:.1%}**"
    )
    st.warning(
        "Those figures describe the **evaluation procedure**, not this fitted demo "
        "model — it was fitted on all 136 images and has no unbiased accuracy estimate.",
        icon="ℹ️",
    )

st.subheader("Add a room photo")
mode = st.radio(
    "How would you like to provide the photo?",
    ["📷 Take a Photo", "📁 Upload Existing Photo"],
    index=1,
    horizontal=True,
    label_visibility="collapsed",
)

source = None
source_caption = "Uploaded image"

if mode == "📷 Take a Photo":
    st.caption(
        "Hold the phone **upright (portrait)** and frame the whole room including the "
        "floor — the model was trained on portrait room photos and is less reliable on "
        "sideways shots."
    )
    try:
        shot = st.camera_input("Point the camera at the room, then press the shutter")
    except Exception as exc:  # camera unsupported / blocked by the browser
        shot = None
        st.warning(
            f"Camera input is not available in this browser ({type(exc).__name__}). "
            "Switch to **📁 Upload Existing Photo** — the photo you already took "
            "with the phone's own camera app works exactly the same way.",
            icon="📷",
        )
    if shot is not None:
        source, source_caption = shot, "Captured photo"
    else:
        st.caption(
            "Waiting for a photo. Allow camera access when the browser asks. "
            "If you cancel, nothing is analysed — you can retake or switch to upload."
        )
else:
    uploaded = st.file_uploader("Upload a room photo (JPG / PNG)",
                                type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        source = uploaded

if source is None:
    st.info("Take or upload a room photo to get a prediction.")
    st.subheader("What the labels mean")
    c1, c2 = st.columns(2)
    c1.markdown(
        "**CLEAN**\n\n"
        "- Floor/tiles reasonably clean\n"
        "- No meaningful garbage, waste or residue\n"
        "- Belongings present but reasonably arranged\n"
        "- Room reasonably neat"
    )
    c2.markdown(
        "**NOT_CLEAN**\n\n"
        "- Visible garbage, waste or debris\n"
        "- Dirty floor or surfaces, kitchen dirt\n"
        "- **or** belongings scattered enough that the room looks untidy — "
        "even when the tiles themselves are clean"
    )
    st.caption(
        "Damp patches, peeling paint, wall holes and loose wires are maintenance "
        "issues, not cleanliness evidence."
    )
    st.stop()

image = Image.open(source)          # in memory only - nothing is written to disk

# --- orientation safeguard -------------------------------------------------
# Measured on the EXIF-corrected frame, i.e. the picture as a person sees it and
# as demo_inference feeds it to the model. The image itself is NOT rotated here
# and is passed to predict() untouched.
_probe = ImageOps.exif_transpose(image)
view_w, view_h = _probe.size
is_landscape = view_w > view_h

left, right = st.columns([1, 1])

with left:
    st.subheader(source_caption)
    st.image(image, use_container_width=True)

if is_landscape:
    with right:
        st.subheader("Prediction blocked")
        st.warning(
            "⚠️ Please hold the phone upright and retake the photo. "
            "The room-cleanliness model is designed for portrait room photos.",
            icon="📱",
        )
        st.caption(
            f"This photo is {view_w} × {view_h} (landscape). No prediction was run. "
            "The photo has not been rotated or altered — please retake it in portrait."
        )
    st.stop()

with right:
    with st.spinner("Analysing..."):
        out = predict(image, backbone, bundle, threshold=thr)

    st.subheader("Prediction")
    if out["label"] == "NOT_CLEAN":
        st.error(f"### NOT_CLEAN", icon="🚩")
    else:
        st.success(f"### CLEAN", icon="✅")

    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{out['confidence']:.1%}")
    m2.metric("P(CLEAN)", f"{out['probability_clean']:.3f}")
    m3.metric("P(NOT_CLEAN)", f"{out['probability_not_clean']:.3f}")
    st.progress(min(1.0, out["probability_not_clean"]),
                text=f"threshold = {out['threshold']:.2f}")

    # --- Why this prediction? -------------------------------------------------
    # The model outputs a single probability, not object-level evidence, so the
    # visible reason is chosen by the person reviewing the photo - never inferred.
    st.subheader("Why this prediction?")
    if out["label"] == "NOT_CLEAN":
        st.markdown(
            "The model scored this photo above the decision threshold and "
            "predicts **NOT_CLEAN**."
        )
    else:
        st.markdown(
            "No strong visual cleanliness or room-neatness issue was detected, "
            "so the model predicts **CLEAN**."
        )
    st.caption(
        f"Model score for this photo: **{out['probability_not_clean']:.3f}** "
        f"(decision threshold **{out['threshold']:.2f}**). The model returns one "
        "overall score and cannot identify specific objects."
    )

    # --- Visible reason: reviewer-selected, kept separate from the prediction --
    st.divider()
    st.subheader("Visible reason")
    REASON_TEXT = {
        "Items not properly arranged":
            "Items are not properly arranged, making the room look untidy.",
        "Visible dirt/waste/debris":
            "Visible dirt, waste, or debris is present and the area requires cleaning.",
        "Both":
            "Items are not properly arranged, and visible dirt, waste or debris is "
            "present; the area requires cleaning.",
        "No specific visible issue":
            "No specific visible cleanliness or arrangement issue was noted.",
    }
    choice = st.radio(
        "Visible reason",
        list(REASON_TEXT),
        index=None,
        label_visibility="collapsed",
    )
    if choice:
        st.success(f"**Visible reason (reviewed from the photo):** {REASON_TEXT[choice]}")
    else:
        st.info("Select what is visible in the photo to record the reason.")
    st.caption(
        "Chosen by the person reviewing the photo — this is separate from the "
        "model prediction above."
    )

with st.expander("Show the 5 model views (what the network actually sees)"):
    st.caption("Whole image plus 2×2 quadrants with 10% overlap, each letterboxed to 384×384.")
    names = out["view_names"]
    wanted = ["whole", "q1", "q2", "q3", "q4"]
    cols = st.columns(len(wanted))
    for col, name in zip(cols, wanted):
        col.image(out["crops"][names.index(name)], caption=name, use_container_width=True)

with st.expander("Known limitations"):
    for lim in meta["limitations"]:
        st.markdown(f"- {lim}")
    st.markdown(
        "- Fitted on 136 images from one building, photographed on one day with one camera.\n"
        "- A new property, camera or lighting condition is out of distribution.\n"
        "- Not suitable for automated decisions affecting tenants."
    )
