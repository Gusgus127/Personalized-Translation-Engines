"""
iDISC · Personalized Translation Engines
Medical Domain Demo — Streamlit Interface
"""

import os
import json
import time
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="iDISC · MedTranslate",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,300&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 300;
}
.stApp { background: #0d1117; }

section[data-testid="stSidebar"] {
    background: #090c10;
    border-right: 1px solid #1e2530;
}
section[data-testid="stSidebar"] * { color: #8b949e !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #c9d1d9 !important; }

.wordmark {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.15rem;
}
.wordmark-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #3d444d;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}

.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 500;
    color: #c9d1d9;
    letter-spacing: -0.01em;
    line-height: 1.2;
}
.page-title span { color: #58a6ff; }
.page-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #3d444d;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.3rem;
    margin-bottom: 2rem;
}

.status-bar {
    display: flex;
    gap: 20px;
    align-items: center;
    padding: 8px 14px;
    background: #090c10;
    border: 1px solid #1e2530;
    border-radius: 4px;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.status-item {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #3d444d;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.status-item b { color: #8b949e; font-weight: 400; }
.status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    display: inline-block; margin-right: 6px; vertical-align: middle;
}
.dot-green { background: #3fb950; }
.dot-amber { background: #d29922; }
.dot-red   { background: #f85149; }
.dot-blue  { background: #58a6ff; }

.panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3d444d;
    margin-bottom: 6px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2530;
}

.stTextArea textarea {
    background: #090c10 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 4px !important;
    color: #c9d1d9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.8 !important;
    caret-color: #58a6ff;
}
.stTextArea textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.08) !important;
}
.stTextArea textarea::placeholder { color: #3d444d !important; }

.output-panel {
    background: #090c10;
    border: 1px solid #1e2530;
    border-radius: 4px;
    padding: 14px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
    line-height: 1.8;
    color: #c9d1d9;
    min-height: 200px;
    white-space: pre-wrap;
    word-break: break-word;
}
.output-panel.empty { color: #3d444d; font-style: italic; }
.output-panel.active { animation: fadeIn 0.3s ease; }
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}

.conf-strip {
    margin-top: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.conf-bar-bg {
    flex: 1;
    height: 3px;
    background: #1e2530;
    border-radius: 2px;
    overflow: hidden;
}
.conf-bar-fill { height: 3px; border-radius: 2px; transition: width 0.5s ease; }
.conf-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    white-space: nowrap;
    min-width: 110px;
    text-align: right;
}

.chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border-radius: 2px;
    border: 1px solid #1e2530;
    color: #8b949e;
    background: #090c10;
    white-space: nowrap;
}
.chip b { color: #58a6ff; font-weight: 400; }

.stButton > button {
    background: #090c10 !important;
    border: 1px solid #1e2530 !important;
    color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 3px !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: #58a6ff !important;
    color: #58a6ff !important;
    background: #090c10 !important;
}
.stButton > button[kind="primary"] {
    background: #1f6feb !important;
    border-color: #1f6feb !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover {
    background: #388bfd !important;
    border-color: #388bfd !important;
    color: #ffffff !important;
}

.hist-row {
    display: grid;
    grid-template-columns: 1fr 1fr auto;
    gap: 12px;
    align-items: start;
    padding: 10px 0;
    border-bottom: 1px solid #1e2530;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
}
.hist-src { color: #8b949e; line-height: 1.5; }
.hist-tgt { color: #c9d1d9; line-height: 1.5; }
.hist-badge {
    font-size: 0.62rem;
    padding: 2px 8px;
    border-radius: 2px;
    white-space: nowrap;
    border: 1px solid;
}

.stSelectbox > div > div {
    background: #0d1117 !important;
    border-color: #1e2530 !important;
    color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}

.sidebar-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3d444d;
    margin-bottom: 4px;
    margin-top: 16px;
}

hr { border-color: #1e2530 !important; margin: 1.5rem 0 !important; }

.streamlit-expanderHeader {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #8b949e !important;
    background: #090c10 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 3px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_MODEL  = "Helsinki-NLP/opus-mt-es-en"
FINETUNED_MODEL = "./mespen_medical_model"

SAMPLE_SENTENCES = [
    "El paciente presenta fiebre alta, tos seca y dificultad para respirar.",
    "Se recomienda administrar amoxicilina 500 mg cada 8 horas durante 7 días.",
    "La resonancia magnética reveló una lesión hipointensa en el lóbulo temporal derecho.",
    "El análisis de sangre mostró niveles elevados de glucosa en ayunas.",
    "Se diagnosticó hipertensión arterial y se inició tratamiento con enalapril.",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def finetuned_available():
    return os.path.isdir(FINETUNED_MODEL) and os.path.isfile(
        os.path.join(FINETUNED_MODEL, "model.safetensors")
    )


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model     = MarianMTModel.from_pretrained(model_path)
    device    = 0 if torch.cuda.is_available() else -1

    # Try pipeline task names across transformers versions
    pipe = None
    for task in ("translation", "text2text-generation"):
        try:
            pipe = pipeline(task, model=model, tokenizer=tokenizer, device=device)
            break
        except Exception:
            continue

    # Final fallback: use model.generate directly
    if pipe is None:
        def _pipe(text, max_length=256, **kwargs):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            translated = model.generate(**inputs, max_length=max_length)
            decoded = tokenizer.decode(translated[0], skip_special_tokens=True)
            return [{"generated_text": decoded}]
        pipe = _pipe

    return pipe, tokenizer, device


def translate_text(pipe, tokenizer, text: str):
    t0     = time.time()
    result = pipe(text.strip(), max_length=256)
    elapsed = round(time.time() - t0, 2)

    # transformers ≥4.40 uses "generated_text"; older builds use "translation_text"
    out = result[0]
    translation = out.get("generated_text") or out.get("translation_text") or ""

    src_len    = len(tokenizer(text)["input_ids"])
    tgt_len    = len(tokenizer(translation)["input_ids"]) if translation else 1
    ratio      = min(src_len, tgt_len) / max(src_len, tgt_len)
    confidence = round(min(ratio * 1.12, 1.0), 3)

    return translation, confidence, elapsed


def conf_color(c: float) -> str:
    if c >= 0.80: return "#3fb950"
    if c >= 0.60: return "#d29922"
    return "#f85149"

def conf_label(c: float) -> str:
    if c >= 0.80: return "AUTO-ACCEPT"
    if c >= 0.60: return "REVIEW"
    return "FLAG"

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("history",           []),
    ("translation",       ""),
    ("confidence",        None),
    ("elapsed",           None),
    ("source_text",       ""),
    ("_pending_example",  None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Apply any pending example BEFORE widgets are instantiated.
# This is the only safe window to write to a widget key.
if st.session_state["_pending_example"] is not None:
    st.session_state["src_area"]         = st.session_state["_pending_example"]
    st.session_state["_pending_example"] = None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='wordmark'>iDISC</div>", unsafe_allow_html=True)
    st.markdown("<div class='wordmark-sub'>Translation Engines · v0.1</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-label'>Domain</div>", unsafe_allow_html=True)
    domain = st.radio(
        "Domain", ["🧬  Medical", "⚖️  Legal (soon)", "🔧  Automotive (soon)"],
        label_visibility="collapsed",
    )
    if domain != "🧬  Medical":
        st.info("Only the Medical domain is available in this demo.")

    st.markdown("<div class='sidebar-label'>Engine</div>", unsafe_allow_html=True)
    ft_ready = finetuned_available()
    engine_options = ["Baseline (opus-mt-es-en)"]
    if ft_ready:
        engine_options.append("Fine-tuned (MeSpEn Medical)")
    engine_choice = st.radio("Engine", engine_options, label_visibility="collapsed")

    if not ft_ready:
        st.caption("Fine-tuned model not found at `./mespen_medical_model`.")

    selected_model = (
        FINETUNED_MODEL if ("Fine-tuned" in engine_choice and ft_ready)
        else BASELINE_MODEL
    )

    st.markdown("<div class='sidebar-label'>Direction</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;"
        "color:#58a6ff;padding:4px 0'>ES → EN</div>",
        unsafe_allow_html=True,
    )
    st.caption("EN → ES coming in next iteration.")

    st.markdown("<div class='sidebar-label'>Device</div>", unsafe_allow_html=True)
    device_str = "CUDA · GPU" if torch.cuda.is_available() else "CPU"
    device_dot = "dot-green" if torch.cuda.is_available() else "dot-amber"
    st.markdown(
        f"<span class='status-dot {device_dot}'></span>"
        f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;"
        f"color:#8b949e'>{device_str}</span>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("<div class='sidebar-label'>Session</div>", unsafe_allow_html=True)
    n = len(st.session_state.history)
    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;"
        f"color:#8b949e'>{n} translation{'s' if n != 1 else ''} this session</div>",
        unsafe_allow_html=True,
    )
    if st.button("Clear history", use_container_width=True):
        for k in ("history", "translation", "confidence", "elapsed", "source_text"):
            st.session_state[k] = [] if k == "history" else ("" if k in ("translation","source_text") else None)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading `{selected_model}`…"):
    try:
        pipe, tokenizer, device_id = load_model(selected_model)
        model_ok = True
    except Exception as e:
        st.error(f"**Model load failed:** `{selected_model}`\n\n```\n{e}\n```")
        model_ok = False

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
engine_label = "Fine-tuned · MeSpEn" if "Fine-tuned" in engine_choice else "Baseline · opus-mt"

st.markdown(
    "<div class='page-title'>Med<span>Translate</span></div>"
    "<div class='page-subtitle'>Spanish → English · Medical Domain · iDISC Demo v0.1</div>",
    unsafe_allow_html=True,
)
st.markdown(f"""
<div class='status-bar'>
    <div class='status-item'><span class='status-dot dot-blue'></span>ENGINE &nbsp;<b>{engine_label}</b></div>
    <div class='status-item'>DOMAIN &nbsp;<b>Medical</b></div>
    <div class='status-item'>DIRECTION &nbsp;<b>ES → EN</b></div>
    <div class='status-item'>DEVICE &nbsp;<b>{device_str}</b></div>
    <div class='status-item'>QA &nbsp;<b>Confidence Threshold</b></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SPLIT SCREEN EDITOR
# ─────────────────────────────────────────────────────────────────────────────
col_src, col_tgt = st.columns(2, gap="medium")

with col_src:
    st.markdown("<div class='panel-label'>Source · Español</div>", unsafe_allow_html=True)
    source_input = st.text_area(
        "source",
        height=220,
        placeholder="Escribe o pega el texto médico en español aquí…",
        label_visibility="collapsed",
        key="src_area",
    )

with col_tgt:
    st.markdown("<div class='panel-label'>Translation · English</div>", unsafe_allow_html=True)
    if st.session_state.translation:
        c   = st.session_state.confidence
        el  = st.session_state.elapsed
        pct = int(c * 100)
        st.markdown(
            f"<div class='output-panel active'>{st.session_state.translation}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"""
<div class='conf-strip'>
    <div class='conf-bar-bg'>
        <div class='conf-bar-fill' style='width:{pct}%;background:{conf_color(c)}'></div>
    </div>
    <div class='conf-text' style='color:{conf_color(c)}'>{pct}% · {conf_label(c)}</div>
</div>
<div class='chip-row'>
    <div class='chip'>Time &nbsp;<b>{el}s</b></div>
    <div class='chip'>Engine &nbsp;<b>{engine_label}</b></div>
    <div class='chip'>Chars &nbsp;<b>{len(st.session_state.translation)}</b></div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='output-panel empty'>Translation will appear here…</div>",
            unsafe_allow_html=True,
        )

# ── Translate button ──
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    translate_btn = st.button(
        "Translate →", use_container_width=True,
        disabled=not model_ok, type="primary",
    )

# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
if translate_btn:
    text = source_input.strip()
    if not text:
        st.warning("Please enter some Spanish text to translate.")
    else:
        with st.spinner("Translating…"):
            try:
                translation, confidence, elapsed = translate_text(pipe, tokenizer, text)
                st.session_state.translation = translation
                st.session_state.confidence  = confidence
                st.session_state.elapsed     = elapsed
                st.session_state.source_text = text
                st.session_state.history.insert(0, {
                    "src":    text[:140],
                    "tgt":    translation[:140],
                    "conf":   confidence,
                    "label":  conf_label(confidence),
                    "color":  conf_color(confidence),
                    "time":   elapsed,
                    "engine": engine_label,
                })
                st.rerun()
            except Exception as e:
                st.error(f"Translation error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE SENTENCES
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("<div class='panel-label'>Quick Examples</div>", unsafe_allow_html=True)
ex_cols = st.columns(len(SAMPLE_SENTENCES))
for i, (col, sentence) in enumerate(zip(ex_cols, SAMPLE_SENTENCES)):
    with col:
        if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
            st.session_state["_pending_example"] = sentence
            st.session_state.translation          = ""
            st.session_state.confidence           = None
            st.session_state.elapsed              = None
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE VS FINE-TUNED COMPARATOR
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("⚖️  Baseline vs Fine-tuned — side-by-side"):
    if not ft_ready:
        st.info("Fine-tuned model not found at `./mespen_medical_model`. Run `mespen-medical-finetune.ipynb` first.")
    else:
        cmp_text = st.text_area(
            "Input for comparison", height=100,
            placeholder="Paste a Spanish medical sentence to compare both models…",
            key="cmp_input",
        )
        if st.button("Run comparison →", key="cmp_btn") and cmp_text.strip():
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.markdown("<div class='panel-label'>Baseline · opus-mt-es-en</div>", unsafe_allow_html=True)
                with st.spinner("Running baseline…"):
                    pb, tb, _ = load_model(BASELINE_MODEL)
                    ob, cb, timeb = translate_text(pb, tb, cmp_text.strip())
                pct_b = int(cb * 100)
                st.markdown(f"<div class='output-panel active'>{ob}</div>", unsafe_allow_html=True)
                st.markdown(f"""
<div class='conf-strip'><div class='conf-bar-bg'>
    <div class='conf-bar-fill' style='width:{pct_b}%;background:{conf_color(cb)}'></div>
</div><div class='conf-text' style='color:{conf_color(cb)}'>{pct_b}% · {conf_label(cb)}</div></div>
<div class='chip-row'><div class='chip'>Time &nbsp;<b>{timeb}s</b></div></div>
""", unsafe_allow_html=True)

            with c2:
                st.markdown("<div class='panel-label'>Fine-tuned · MeSpEn Medical</div>", unsafe_allow_html=True)
                with st.spinner("Running fine-tuned…"):
                    pf, tf, _ = load_model(FINETUNED_MODEL)
                    of, cf, timef = translate_text(pf, tf, cmp_text.strip())
                pct_f = int(cf * 100)
                delta = pct_f - pct_b
                st.markdown(f"<div class='output-panel active'>{of}</div>", unsafe_allow_html=True)
                st.markdown(f"""
<div class='conf-strip'><div class='conf-bar-bg'>
    <div class='conf-bar-fill' style='width:{pct_f}%;background:{conf_color(cf)}'></div>
</div><div class='conf-text' style='color:{conf_color(cf)}'>{pct_f}% · {conf_label(cf)}</div></div>
<div class='chip-row'>
    <div class='chip'>Time &nbsp;<b>{timef}s</b></div>
    <div class='chip'>Δ Confidence &nbsp;<b>{'+' if delta>=0 else ''}{delta}%</b></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    with st.expander(f"📋  Translation History ({len(st.session_state.history)} entries)"):
        st.markdown("""
<div class='hist-row' style='color:#3d444d;font-size:0.62rem;letter-spacing:0.15em'>
    <div>SOURCE</div><div>TRANSLATION</div><div>QA</div>
</div>""", unsafe_allow_html=True)

        for item in st.session_state.history[:20]:
            badge_style = f"color:{item['color']};border-color:{item['color']};background:transparent"
            st.markdown(f"""
<div class='hist-row'>
    <div class='hist-src'>{item['src']}{'…' if len(item['src'])>=140 else ''}</div>
    <div class='hist-tgt'>{item['tgt']}{'…' if len(item['tgt'])>=140 else ''}</div>
    <div>
        <span class='hist-badge' style='{badge_style}'>{item['label']}</span>
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;color:#3d444d;
                    margin-top:4px'>{item['engine']}<br>{item['time']}s</div>
    </div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                "Export JSON",
                data=json.dumps(st.session_state.history, indent=2, ensure_ascii=False),
                file_name="translation_history.json",
                mime="application/json",
                use_container_width=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#3d444d;
            letter-spacing:0.1em;text-align:center;border-top:1px solid #1e2530;padding-top:1rem'>
    iDISC · PERSONALIZED TRANSLATION ENGINES · DEMO v0.1 · MEDICAL DOMAIN
    &nbsp;·&nbsp; QA MODULE / AUTOMATED REPAIR / LEGAL / AUTOMOTIVE — COMING SOON
</div>
""", unsafe_allow_html=True)