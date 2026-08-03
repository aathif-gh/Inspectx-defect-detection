import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(page_title="InspectX", layout="wide", page_icon="◆")

# ---------------------------------------------------------------------------
# VISUAL IDENTITY
# Subject: industrial quality-control inspection, not a generic "AI SaaS".
# Deep charcoal + steel panels, one accent (signal amber -- the color of
# inspection/hazard lighting on a real factory floor), monospace for all
# measured data (scores, thresholds) to read as instrumentation rather
# than decoration. Signature element: the verdict renders as a stamped
# PASS/FAIL tag, like a physical QC ink stamp, not a colored alert box.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #0A0D10;
    color: #D8DEE4;
}

/* Hide default Streamlit chrome for a cleaner instrument-panel feel */
#MainMenu, header, footer { visibility: hidden; }

.scan-header {
    display: flex;
    align-items: baseline;
    gap: 14px;
    padding: 28px 0 8px 0;
    border-bottom: 1px solid #232A31;
    margin-bottom: 28px;
}
.scan-header .mark {
    font-family: 'JetBrains Mono', monospace;
    color: #F2A93B;
    font-size: 22px;
}
.scan-header h1 {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #EDEFF2;
    margin: 0;
}
.scan-header .sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #5A6672;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5A6672;
    margin-bottom: 10px;
}

/* Scan-bed frame with corner brackets, like a scanner/camera viewport */
.scan-bed {
    position: relative;
    background: #10151A;
    border: 1px solid #232A31;
    border-radius: 2px;
    padding: 18px;
}
.scan-bed::before, .scan-bed::after,
.scan-bed .br1, .scan-bed .br2 {
    content: '';
    position: absolute;
    width: 18px;
    height: 18px;
    border-color: #F2A93B;
    border-style: solid;
    opacity: 0.85;
}
.scan-bed::before { top: -1px; left: -1px; border-width: 2px 0 0 2px; }
.scan-bed::after { top: -1px; right: -1px; border-width: 2px 2px 0 0; }
.scan-bed .br1 { bottom: -1px; left: -1px; border-width: 0 0 2px 2px; }
.scan-bed .br2 { bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }

/* The stamp -- signature element */
.stamp-wrap { display: flex; justify-content: center; padding: 18px 0 6px 0; }
.stamp {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 28px;
    letter-spacing: 0.15em;
    padding: 14px 34px;
    border: 3px solid currentColor;
    border-radius: 4px;
    transform: rotate(-3deg);
    display: inline-block;
}
.stamp.fail { color: #F2A93B; box-shadow: 0 0 0 1px rgba(242,169,59,0.15) inset; }
.stamp.pass { color: #5FBF9F; box-shadow: 0 0 0 1px rgba(95,191,159,0.15) inset; }

.metric-row {
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid #1B2126;
    font-size: 13px;
}
.metric-row .k { color: #5A6672; letter-spacing: 0.05em; text-transform: uppercase; font-size: 11px; }
.metric-row .v { color: #D8DEE4; font-weight: 500; }

.stButton>button {
    background: #F2A93B;
    color: #0A0D10;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: none;
    border-radius: 2px;
    padding: 10px 22px;
    width: 100%;
}
.stButton>button:hover { background: #FFC266; }

[data-testid="stFileUploader"] {
    background: #10151A;
    border: 1px dashed #2A323A;
    border-radius: 2px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="scan-header">
    <span class="mark">◆</span>
    <h1>InspectX</h1>
    <span class="sub">— Bottle Category / Patchcore Model</span>
</div>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="panel-label">01 / Sample Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.markdown('<div class="scan-bed"><div class="br1"></div><div class="br2"></div>', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        run = st.button("Run Inspection")
    else:
        run = False

with col_result:
    st.markdown('<div class="panel-label">02 / Result</div>', unsafe_allow_html=True)

    if uploaded_file is not None and run:
        with st.spinner("Scanning..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_URL}/predict", files=files).json()

        label = response["pred_label"]
        score = response["pred_score"]
        stamp_class = "fail" if label == "defective" else "pass"
        stamp_text = "DEFECT" if label == "defective" else "PASS"

        st.markdown(f"""
        <div class="stamp-wrap"><span class="stamp {stamp_class}">{stamp_text}</span></div>
        <div class="metric-row"><span class="k">Anomaly Score</span><span class="v">{score:.4f}</span></div>
        <div class="metric-row"><span class="k">Threshold</span><span class="v">0.5000</span></div>
        <div class="metric-row"><span class="k">Model</span><span class="v">PatchCore / wide_resnet50_2</span></div>
        """, unsafe_allow_html=True)

        heatmap_bytes = base64.b64decode(response["heatmap_base64"])
        heatmap_image = Image.open(io.BytesIO(heatmap_bytes))

        st.markdown('<div class="panel-label" style="margin-top:22px;">03 / Anomaly Localization</div>', unsafe_allow_html=True)
        st.markdown('<div class="scan-bed"><div class="br1"></div><div class="br2"></div>', unsafe_allow_html=True)
        st.image(heatmap_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#3A434C; font-family:JetBrains Mono, monospace; font-size:13px; padding:30px 0;">Awaiting input — upload an image and run inspection.</div>', unsafe_allow_html=True)