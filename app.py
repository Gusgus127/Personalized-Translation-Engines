"""
iDISC · Personalized Translation Engines
Medical Domain Demo — Streamlit Interface (with Real COMET QA Thresholds)
"""

import os
import json
import time
import re
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch
import pdfplumber  

st.set_page_config(
    page_title="iDISC · MedTranslate",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,300&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; font-weight: 300; }
.stApp { background: #0d1117; }

section[data-testid="stSidebar"] { background: #090c10; border-right: 1px solid #1e2530; }
section[data-testid="stSidebar"] * { color: #8b949e !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #c9d1d9 !important; }

.wordmark { font-family:'IBM Plex Mono',monospace;font-size:1.05rem;font-weight:600;color:#58a6ff;letter-spacing:.18em;text-transform:uppercase;margin-bottom:.15rem; }
.wordmark-sub { font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#3d444d;letter-spacing:.2em;text-transform:uppercase;margin-bottom:1.8rem; }

.page-title { font-family:'IBM Plex Mono',monospace;font-size:1.7rem;font-weight:500;color:#c9d1d9;letter-spacing:-.01em;line-height:1.2; }
.page-title span { color:#58a6ff; }
.page-subtitle { font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#3d444d;letter-spacing:.15em;text-transform:uppercase;margin-top:.3rem;margin-bottom:2rem; }

.status-bar { display:flex;gap:20px;align-items:center;padding:8px 14px;background:#090c10;border:1px solid #1e2530;border-radius:4px;margin-bottom:1.5rem;flex-wrap:wrap; }
.status-item { font-family:'IBM Plex Mono',monospace;font-size:.68rem;color:#3d444d;letter-spacing:.08em;text-transform:uppercase; }
.status-item b { color:#8b949e;font-weight:400; }
.status-dot { width:6px;height:6px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle; }
.dot-green { background:#3fb950; } .dot-amber { background:#d29922; } .dot-red { background:#f85149; } .dot-blue { background:#58a6ff; }

.panel-label { font-family:'IBM Plex Mono',monospace;font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;color:#3d444d;margin-bottom:6px;padding-bottom:6px;border-bottom:1px solid #1e2530; }

.stTextArea textarea { background:#090c10 !important;border:1px solid #1e2530 !important;border-radius:4px !important;color:#c9d1d9 !important;font-family:'IBM Plex Mono',monospace !important;font-size:.88rem !important;line-height:1.8 !important;caret-color:#58a6ff; }
.stTextArea textarea:focus { border-color:#58a6ff !important;box-shadow:0 0 0 2px rgba(88,166,255,.08) !important; }
.stTextArea textarea::placeholder { color:#3d444d !important; }

.output-panel { background:#090c10;border:1px solid #1e2530;border-radius:4px;padding:14px 16px;font-family:'IBM Plex Mono',monospace;font-size:.88rem;line-height:1.8;color:#c9d1d9;min-height:200px;white-space:pre-wrap;word-break:break-word; }
.output-panel.empty { color:#3d444d;font-style:italic; }
.output-panel.active { animation:fadeIn .3s ease; }
@keyframes fadeIn { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:translateY(0)} }

.comet-segment { display:inline;padding:2px 4px;margin:0 1px;border-radius:3px;position:relative;cursor:help;transition:background .2s ease; }
.comet-segment:hover { filter:brightness(1.2); }

.conf-strip { margin-top:10px;display:flex;align-items:center;gap:10px; }
.conf-bar-bg { flex:1;height:3px;background:#1e2530;border-radius:2px;overflow:hidden; }
.conf-bar-fill { height:3px;border-radius:2px;transition:width .5s ease; }
.conf-text { font-family:'IBM Plex Mono',monospace;font-size:.68rem;letter-spacing:.1em;white-space:nowrap;min-width:110px;text-align:right; }

.chip-row { display:flex;gap:8px;flex-wrap:wrap;margin-top:10px; }
.chip { font-family:'IBM Plex Mono',monospace;font-size:.65rem;letter-spacing:.08em;padding:3px 10px;border-radius:2px;border:1px solid #1e2530;color:#8b949e;background:#090c10;white-space:nowrap; }
.chip b { color:#58a6ff;font-weight:400; }

.stButton > button { background:#090c10 !important;border:1px solid #1e2530 !important;color:#8b949e !important;font-family:'IBM Plex Mono',monospace !important;font-size:.72rem !important;letter-spacing:.05em !important;border-radius:3px !important;padding:6px 14px !important;transition:all .15s ease !important; }
.stButton > button:hover { border-color:#58a6ff !important;color:#58a6ff !important;background:#090c10 !important; }
.stButton > button[kind="primary"] { background:#1f6feb !important;border-color:#1f6feb !important;color:#ffffff !important; }
.stButton > button[kind="primary"]:hover { background:#388bfd !important;border-color:#388bfd !important;color:#ffffff !important; }

.hist-row { display:grid;grid-template-columns:1fr 1fr auto;gap:12px;align-items:start;padding:10px 0;border-bottom:1px solid #1e2530;font-family:'IBM Plex Mono',monospace;font-size:.76rem; }
.hist-src { color:#8b949e;line-height:1.5; } .hist-tgt { color:#c9d1d9;line-height:1.5; }
.hist-badge { font-size:.62rem;padding:2px 8px;border-radius:2px;white-space:nowrap;border:1px solid; }

.sidebar-label { font-family:'IBM Plex Mono',monospace;font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#3d444d;margin-bottom:4px;margin-top:16px; }
hr { border-color:#1e2530 !important;margin:1.5rem 0 !important; }
.streamlit-expanderHeader { font-family:'IBM Plex Mono',monospace !important;font-size:.78rem !important;color:#8b949e !important;background:#090c10 !important;border:1px solid #1e2530 !important;border-radius:3px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_ES_EN  = "Helsinki-NLP/opus-mt-es-en"
BASELINE_EN_ES  = "Helsinki-NLP/opus-mt-en-es"
FINETUNED_ES_EN = "./mespen_medical_model"
FINETUNED_EN_ES = "./mespen_medical_model_en_es"

COMET_THRESHOLD_HI = 0.85
COMET_THRESHOLD_LO = 0.70

SAMPLE_SENTENCES_ES = [
    "El paciente presenta fiebre alta, tos seca y dificultad para respirar.",
    "Se recomienda administrar amoxicilina 500 mg cada 8 horas durante 7 días.",
    "La resonancia magnética reveló una lesión hipointensa en el lóbulo temporal derecho.",
    "El análisis de sangre mostró niveles elevados de glucosa en ayunas.",
    "Se diagnosticó hipertensión arterial y se inició tratamiento con enalapril.",
]

SAMPLE_SENTENCES_EN = [
    "The patient presents with high fever, dry cough, and difficulty breathing.",
    "It is recommended to administer amoxicillin 500 mg every 8 hours for 7 days.",
    "The MRI revealed a hypointense lesion in the right temporal lobe.",
    "The blood test showed elevated fasting glucose levels.",
    "Arterial hypertension was diagnosed and treatment with enalapril was initiated.",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def finetuned_available(direction: str = "ES → EN") -> bool:
    path = FINETUNED_EN_ES if direction == "EN → ES" else FINETUNED_ES_EN
    return os.path.isdir(path) and (
        os.path.isfile(os.path.join(path, "model.safetensors")) or
        os.path.isfile(os.path.join(path, "pytorch_model.bin"))
    )

def resolve_model(direction: str, engine_choice: str) -> str:
    ft = "Fine-tuned" in engine_choice
    if direction == "EN → ES":
        return FINETUNED_EN_ES if (ft and finetuned_available("EN → ES")) else BASELINE_EN_ES
    else:
        return FINETUNED_ES_EN if (ft and finetuned_available("ES → EN")) else BASELINE_ES_EN

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model     = MarianMTModel.from_pretrained(model_path)
    device    = 0 if torch.cuda.is_available() else -1

    pipe = None
    for task in ("translation", "text2text-generation"):
        try:
            pipe = pipeline(task, model=model, tokenizer=tokenizer, device=device)
            break
        except Exception:
            continue

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

@st.cache_resource(show_spinner=False)
def load_comet():
    """
    Try COMET models in order of preference.
    Returns (model, variant, model_name) or (None, None, None).

    Variants:
      "kiwi"  — wmt22/wmt23-cometkiwi-da: outputs [0,1], higher=better, clamp only
      "qe_da" — wmt20-comet-qe-da:        outputs [-1,1], map with (s+1)/2
    """
    candidates = [
        ("Unbabel/wmt22-cometkiwi-da",    "kiwi"),
        ("Unbabel/wmt23-cometkiwi-da-xl", "kiwi"),
        ("Unbabel/wmt20-comet-qe-da",     "qe_da"),
    ]
    for model_name, variant in candidates:
        try:
            from comet import download_model, load_from_checkpoint
            model_path = download_model(model_name)
            model = load_from_checkpoint(model_path)
            st.sidebar.success(f"✅ COMET loaded: {model_name}")
            return model, variant, model_name
        except Exception:
            continue
    st.sidebar.error("⚠️ No COMET model could be loaded. Falling back to heuristic.")
    return None, None, None

def _normalise_comet(raw_scores: list, variant: str) -> list:
    """
    Normalise raw COMET scores to [0, 1] depending on which model is loaded:

    "kiwi"  — wmt22/wmt23-cometkiwi-da outputs scores already in [0, 1].
               Higher = better. Just clamp.
    "qe_da" — wmt20-comet-qe-da outputs scores in roughly [-1, 1].
               Map to [0, 1] with (s + 1) / 2.
    """
    processed = []
    for s in raw_scores:
        if variant == "kiwi":
            val = max(0.0, min(1.0, float(s)))
        else:  # qe_da
            val = max(0.0, min(1.0, (float(s) + 1.0) / 2.0))
        processed.append(round(val, 3))
    return processed


def clean_extracted_text(text: str) -> str:
    """
    Clean up pdfplumber output:
    - Remove table-of-contents dot leaders (3+ dots or mixed dots/spaces)
    - Collapse runs of 3+ repeated identical lines (loop artifacts)
    - Remove lines that are pure page numbers or whitespace
    - Collapse excessive blank lines
    """
    lines = text.splitlines()
    cleaned = []
    prev_line = None
    repeat_count = 0
    for line in lines:
        # Strip trailing whitespace
        line = line.rstrip()
        # Drop TOC dot-leader lines (line is mostly dots/spaces after some text)
        if re.search(r'[.·\s]{6,}', line) and re.search(r'\d+\s*$', line):
            continue
        # Drop lines that are only a number (page numbers)
        if re.fullmatch(r'\s*\d{1,3}\s*', line):
            continue
        # Collapse repeated identical lines (loop artifacts from PDF layout)
        if line == prev_line:
            repeat_count += 1
            if repeat_count >= 2:   # allow at most 2 identical consecutive lines
                continue
        else:
            repeat_count = 0
        prev_line = line
        cleaned.append(line)
    # Collapse 3+ consecutive blank lines into one
    result = re.sub(r'(\n\s*){3,}', '\n\n', '\n'.join(cleaned))
    return result.strip()


def split_into_sentences(text: str) -> list:
    """Split text into sentence-level segments."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def calculate_comet_scores(comet_model, comet_variant: str, sources: list, hypotheses: list) -> tuple:
    """
    Score a batch of (src, mt) sentence pairs in one COMET call.
    Returns (mean_score, [per_sentence_scores]) both normalised to [0, 1].
    """
    if comet_model is None or not sources:
        scores = []
        for s_text, h_text in zip(sources, hypotheses):
            s = len(s_text.split())
            h = len(h_text.split()) if h_text else 1
            scores.append(round(min(min(s, h) / max(s, h, 1) * 0.95, 0.95), 3))
        mean = round(sum(scores) / len(scores), 3) if scores else 0.75
        return mean, scores

    gpus = 1 if torch.cuda.is_available() else 0
    data = [{"src": s, "mt": h} for s, h in zip(sources, hypotheses)]

    try:
        result = comet_model.predict(data, batch_size=16, gpus=gpus, progress_bar=False)
        if hasattr(result, "scores"):
            raw_scores = [float(x) for x in result.scores]
        elif isinstance(result, dict):
            raw_scores = [float(x) for x in result.get("scores", [0.5] * len(sources))]
        else:
            return 0.75, [0.75] * len(sources)

        # Log raw scores to sidebar so we can see what COMET actually outputs
        st.sidebar.write(f"🔬 Raw COMET scores: {[round(x,4) for x in raw_scores]}")

        norm_scores = _normalise_comet(raw_scores, comet_variant)
        mean = round(sum(norm_scores) / len(norm_scores), 3)
        return mean, norm_scores
    except Exception as e:
        st.sidebar.warning(f"⚠️ Scoring error: {str(e)}")
        return 0.75, [0.75] * len(sources)

MAX_CHUNK_TOKENS = 400  # safe budget below MarianMT's 512-token limit

def _chunk_sentences(sentences: list, tokenizer, max_tokens: int = MAX_CHUNK_TOKENS) -> list:
    """
    Group sentences into chunks that fit within `max_tokens` tokens.
    Each chunk is a list of sentences.
    """
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        sent_len = len(tokenizer.encode(sent, add_special_tokens=False))
        # If a single sentence is already over the limit, split it by words
        if sent_len > max_tokens:
            words = sent.split()
            sub, sub_len = [], 0
            for word in words:
                w_len = len(tokenizer.encode(word, add_special_tokens=False))
                if sub_len + w_len > max_tokens and sub:
                    chunks.append([" ".join(sub)])
                    sub, sub_len = [word], w_len
                else:
                    sub.append(word)
                    sub_len += w_len
            if sub:
                chunks.append([" ".join(sub)])
            continue
        if current_len + sent_len > max_tokens and current:
            chunks.append(current)
            current, current_len = [sent], sent_len
        else:
            current.append(sent)
            current_len += sent_len
    if current:
        chunks.append(current)
    return chunks


def translate_text(pipe, tokenizer, comet_model, comet_variant: str, text: str):
    t0 = time.time()

    # Split into sentences then group into token-safe chunks
    src_segs   = split_into_sentences(text)
    chunks     = _chunk_sentences(src_segs, tokenizer)

    tgt_segs   = []
    used_src   = []   # flattened sentences actually translated
    for chunk_sents in chunks:
        chunk_text = " ".join(chunk_sents)
        try:
            result = pipe(chunk_text, max_length=512)
            out    = result[0]
            translated_chunk = out.get("generated_text") or out.get("translation_text") or ""
        except Exception:
            translated_chunk = chunk_text   # fallback: keep original on error
        chunk_tgt_segs = split_into_sentences(translated_chunk)
        # Align segment count to source sentences in this chunk
        while len(chunk_tgt_segs) < len(chunk_sents):
            chunk_tgt_segs.append(chunk_tgt_segs[-1] if chunk_tgt_segs else translated_chunk)
        chunk_tgt_segs = chunk_tgt_segs[:len(chunk_sents)]
        tgt_segs.extend(chunk_tgt_segs)
        used_src.extend(chunk_sents)

    translation = " ".join(tgt_segs)
    elapsed     = round(time.time() - t0, 2)

    # Score all sentence pairs in a single batched COMET call
    global_score, seg_scores = calculate_comet_scores(
        comet_model, comet_variant, used_src, tgt_segs
    )

    segments_data = [
        {"src_seg": s, "tgt_seg": t, "score": score}
        for s, t, score in zip(used_src, tgt_segs, seg_scores)
    ]

    return translation, round(global_score, 3), elapsed, segments_data

def conf_color(c: float, hi: float = COMET_THRESHOLD_HI, lo: float = COMET_THRESHOLD_LO) -> str:
    if c >= hi: return "#3fb950"
    if c >= lo: return "#d29922"
    return "#f85149"

def conf_label(c: float, hi: float = COMET_THRESHOLD_HI, lo: float = COMET_THRESHOLD_LO) -> str:
    if c >= hi: return "AUTO-ACCEPT"
    if c >= lo: return "REVIEW"
    return "FLAG"

def render_segment_heatmap(segments: list, hi: float, lo: float) -> str:
    html_out = "<div class='output-panel active'>"
    for seg in segments:
        score  = seg["score"]
        label  = conf_label(score, hi, lo)
        color  = conf_color(score, hi, lo)

        safe   = seg["tgt_seg"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Only show badge and colour if NOT auto‑accepted
        if label == "AUTO-ACCEPT":
            # No background, no underline, no badge
            span_style = ""
            badge = ""
        else:
            alpha = color + "25"
            span_style = f"background-color:{alpha};border-bottom:2px solid {color};border-radius:3px;padding:3px 5px"
            badge = (
                f"<span style='font-size:0.6rem;font-family:IBM Plex Mono,monospace;"
                f"color:{color};border:1px solid {color};border-radius:2px;"
                f"padding:1px 6px;margin-left:8px;vertical-align:middle;"
                f"white-space:nowrap;letter-spacing:0.06em;opacity:0.9'>"
                f"{score:.2f} · {label}"
                f"</span>"
            )

        html_out += (
            f"<div style='margin-bottom:10px;line-height:1.7'>"
            f"<span class='comet-segment' style='{span_style}' "
            f"title='COMET {score:.3f} · {label}'>{safe}</span>"
            f"{badge}"
            f"</div>"
        )
    html_out += "</div>"
    return html_out

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("history",           []),
    ("translation",       ""),
    ("confidence",        None),
    ("elapsed",           None),
    ("segments",          []),
    ("source_text",       ""),
    ("_pending_example",  None),
    ("_pending_pdf_text", None),
    ("_last_extracted_pdf",    None),
    ("_editing_translation",   False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Resolve any pending pre-fill before widgets are instantiated
if st.session_state["_pending_example"] is not None:
    st.session_state["src_area"]         = st.session_state["_pending_example"]
    st.session_state["_pending_example"] = None

if st.session_state["_pending_pdf_text"] is not None:
    st.session_state["src_area"]          = st.session_state["_pending_pdf_text"]
    st.session_state["_pending_pdf_text"] = None

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

    st.markdown("<div class='sidebar-label'>Direction</div>", unsafe_allow_html=True)
    direction = st.radio("Direction", ["ES → EN", "EN → ES"], label_visibility="collapsed")

    st.markdown("<div class='sidebar-label'>Engine</div>", unsafe_allow_html=True)
    ft_ready = finetuned_available(direction)
    engine_options = [f"Baseline (opus-mt-{'es-en' if direction == 'ES → EN' else 'en-es'})"]
    if ft_ready:
        engine_options.append("Fine-tuned (MeSpEn Medical)")
    engine_choice = st.radio("Engine", engine_options, label_visibility="collapsed")

    if not ft_ready:
        ft_path = FINETUNED_EN_ES if direction == "EN → ES" else FINETUNED_ES_EN
        st.caption(f"Fine-tuned model not found at `{ft_path}`.")

    selected_model = resolve_model(direction, engine_choice)

    st.markdown("<div class='sidebar-label'>Device</div>", unsafe_allow_html=True)
    device_str = "CUDA · GPU" if torch.cuda.is_available() else "CPU"
    device_dot = "dot-green" if torch.cuda.is_available() else "dot-amber"
    st.markdown(
        f"<span class='status-dot {device_dot}'></span>"
        f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#8b949e'>{device_str}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-label'>QA · COMET Thresholds</div>", unsafe_allow_html=True)
    comet_hi = st.slider("Auto-accept  ≥", min_value=0.50, max_value=1.00, value=COMET_THRESHOLD_HI, step=0.01, format="%.2f", key="comet_hi")
    comet_lo = st.slider("Review  ≥", min_value=0.40, max_value=float(comet_hi), value=min(COMET_THRESHOLD_LO, comet_hi), step=0.01, format="%.2f", key="comet_lo")
    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#3d444d;margin-top:2px;line-height:1.8'>"
        f"<span style='color:#3fb950'>■</span> ≥{comet_hi:.2f} AUTO-ACCEPT<br>"
        f"<span style='color:#d29922'>■</span> ≥{comet_lo:.2f} REVIEW<br>"
        f"<span style='color:#f85149'>■</span> &lt;{comet_lo:.2f} FLAG</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("<div class='sidebar-label'>Session</div>", unsafe_allow_html=True)
    n = len(st.session_state.history)
    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#8b949e'>"
        f"{n} translation{'s' if n != 1 else ''} this session</div>",
        unsafe_allow_html=True,
    )
    if st.button("Clear history", use_container_width=True):
        for k in ("history", "translation", "confidence", "elapsed", "segments", "source_text"):
            st.session_state[k] = [] if k == "history" else ("" if k in ("translation", "source_text") else None)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL + COMET
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading `{selected_model}`…"):
    try:
        pipe, tokenizer, device_id = load_model(selected_model)
        model_ok = True
    except Exception as e:
        st.error(f"**Model load failed:** `{selected_model}`\n\n```\n{e}\n```")
        model_ok = False

with st.spinner("Loading COMET QA model…"):
    _comet_result = load_comet()
    comet_model, comet_variant, comet_model_name = (
        _comet_result if _comet_result[0] is not None else (None, None, None)
    )
    comet_ready = comet_model is not None

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
engine_label = "Fine-tuned · MeSpEn" if "Fine-tuned" in engine_choice else "Baseline · opus-mt"
dir_label    = direction

st.markdown(
    "<div class='page-title'>Med<span>Translate</span></div>"
    f"<div class='page-subtitle'>{dir_label} · Medical Domain · iDISC Demo v0.1</div>",
    unsafe_allow_html=True,
)
st.markdown(f"""
<div class='status-bar'>
    <div class='status-item'><span class='status-dot dot-blue'></span>ENGINE &nbsp;<b>{engine_label}</b></div>
    <div class='status-item'>DOMAIN &nbsp;<b>Medical</b></div>
    <div class='status-item'>DIRECTION &nbsp;<b>{dir_label}</b></div>
    <div class='status-item'>DEVICE &nbsp;<b>{device_str}</b></div>
    <div class='status-item'>QA &nbsp;<b>{comet_model_name.split('/')[-1] if comet_ready else 'Heuristic'}</b></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SPLIT SCREEN EDITOR
# ─────────────────────────────────────────────────────────────────────────────
col_src, col_tgt = st.columns(2, gap="medium")

src_lang, tgt_lang = ("Español", "English") if direction == "ES → EN" else ("English", "Español")
src_placeholder = (
    "Escribe o pega el texto médico en español aquí…"
    if direction == "ES → EN"
    else "Type or paste medical text in English here…"
)

with col_src:
    st.markdown(f"<div class='panel-label'>Source · {src_lang}</div>", unsafe_allow_html=True)

    # tabs for input method
    tab1, tab2 = st.tabs(["✏️  Text input", "📄  PDF upload"])

    with tab1:
        source_input = st.text_area(
            "source_text",
            height=220,
            placeholder=src_placeholder,
            label_visibility="collapsed",
            key="src_area",
        )

    with tab2:
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            key="pdf_uploader",
        )
        if uploaded_file is not None:
            # Only extract if this is a newly uploaded file (guard against rerun loops)
            if st.session_state["_last_extracted_pdf"] != uploaded_file.name:
                with st.spinner("Extracting text from PDF…"):
                    try:
                        with pdfplumber.open(uploaded_file) as pdf:
                            pages_text = []
                            for page in pdf.pages:
                                t = page.extract_text()
                                if t:
                                    pages_text.append(t.strip())
                            extracted_text = "\n\n".join(pages_text)
                        # Clean layout artifacts then stage
                        extracted_text = clean_extracted_text(extracted_text)
                        st.session_state["_last_extracted_pdf"] = uploaded_file.name
                        st.session_state["_pending_pdf_text"]   = extracted_text
                        st.rerun()
                    except Exception as e:
                        st.error(f"PDF extraction failed: {e}")
            else:
                char_count = len(st.session_state.get("src_area", ""))
                st.success(f"✅ {uploaded_file.name} — {char_count:,} characters extracted.")
                st.text_area(
                    "Extracted text (read-only preview)",
                    value=st.session_state.get("src_area", ""),
                    height=200,
                    disabled=True,
                    label_visibility="collapsed",
                    key="pdf_preview",
                )
                st.caption("Switch to the ✏️ Text input tab to edit before translating.")
        else:
            st.session_state["_last_extracted_pdf"] = None
            st.info("Upload a PDF file and its text will appear in the source editor.")

with col_tgt:
    # Header row: label + edit/done toggle button
    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.markdown(f"<div class='panel-label'>Translation · {tgt_lang}</div>", unsafe_allow_html=True)
    with hdr_r:
        if st.session_state.translation:
            editing = st.session_state["_editing_translation"]
            btn_txt = "✅ Done" if editing else "✏️ Edit"
            if st.button(btn_txt, key="toggle_edit", use_container_width=True):
                st.session_state["_editing_translation"] = not editing
                st.rerun()

    if st.session_state.translation:
        c  = st.session_state.confidence
        el = st.session_state.elapsed

        if st.session_state["_editing_translation"]:
            # ── EDIT MODE: plain editable text area ──────────────────────
            edited_translation = st.text_area(
                "editable_translation",
                value=st.session_state.translation,
                height=340,
                label_visibility="collapsed",
                key="editable_tgt",
            )
            if edited_translation != st.session_state.translation:
                st.session_state.translation = edited_translation
                st.session_state.segments    = []   # scores no longer valid
                st.session_state.confidence  = None
            st.caption("Editing mode — COMET scores cleared. Click ✅ Done to return to score view.")
        else:
            # ── SCORE VIEW: per-sentence COMET heatmap ───────────────────
            if st.session_state.segments:
                heatmap_html = render_segment_heatmap(
                    st.session_state.segments, comet_hi, comet_lo
                )
                st.markdown(heatmap_html, unsafe_allow_html=True)
            else:
                # Edited text with no scores — show plain panel
                safe_txt = st.session_state.translation.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                st.markdown(f"<div class='output-panel active'>{safe_txt}</div>", unsafe_allow_html=True)

            # Global confidence bar + chips
            if c is not None:
                color = conf_color(c, comet_hi, comet_lo)
                label = conf_label(c, comet_hi, comet_lo)
                pct   = min(100, int(c * 100))
                st.markdown(f"""
                <div class='conf-strip'>
                    <div class='conf-bar-bg'><div class='conf-bar-fill' style='width:{pct}%;background:{color}'></div></div>
                    <div class='conf-text' style='color:{color}'>{label} &nbsp;{c:.3f}</div>
                </div>
                <div class='chip-row'>
                    <div class='chip'>COMET &nbsp;<b>{c:.3f}</b></div>
                    <div class='chip'>Time &nbsp;<b>{el}s</b></div>
                    <div class='chip'>Segments &nbsp;<b>{len(st.session_state.segments)}</b></div>
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
    btn_label = "Translate →" if direction == "ES → EN" else "Traducir →"
    translate_btn = st.button(btn_label, use_container_width=True, disabled=not model_ok, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
if translate_btn:
    text = source_input.strip()
    if not text:
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating & scoring sentences with COMET…"):
            try:
                translation, confidence, elapsed, segments = translate_text(
                    pipe, tokenizer, comet_model, comet_variant, text
                )
                st.session_state.translation = translation
                st.session_state.confidence  = confidence
                st.session_state.elapsed     = elapsed
                st.session_state.segments    = segments
                st.session_state.source_text = text
                st.session_state.history.insert(0, {
                    "src":      text[:140],
                    "tgt":      translation[:140],
                    "conf":     confidence,
                    "label":    conf_label(confidence, comet_hi, comet_lo),
                    "color":    conf_color(confidence, comet_hi, comet_lo),
                    "time":     elapsed,
                    "engine":   engine_label,
                    "segments": len(segments),
                })
                st.rerun()
            except Exception as e:
                st.error(f"Translation error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE SENTENCES
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("<div class='panel-label'>Quick Examples</div>", unsafe_allow_html=True)
SAMPLE_SENTENCES = SAMPLE_SENTENCES_ES if direction == "ES → EN" else SAMPLE_SENTENCES_EN
ex_cols = st.columns(len(SAMPLE_SENTENCES))
for i, (col, sentence) in enumerate(zip(ex_cols, SAMPLE_SENTENCES)):
    with col:
        if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
            st.session_state["_pending_example"] = sentence
            st.session_state.translation          = ""
            st.session_state.confidence           = None
            st.session_state.elapsed              = None
            st.session_state.segments             = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE VS FINE-TUNED COMPARATOR
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("⚖️  Baseline vs Fine-tuned — side-by-side"):
    ft_ready_cmp  = finetuned_available(direction)
    ft_path_cmp   = FINETUNED_EN_ES if direction == "EN → ES" else FINETUNED_ES_EN
    base_path_cmp = BASELINE_EN_ES  if direction == "EN → ES" else BASELINE_ES_EN
    base_label_cmp = f"opus-mt-{'en-es' if direction == 'EN → ES' else 'es-en'}"

    if not ft_ready_cmp:
        st.info(f"Fine-tuned model not found at `{ft_path_cmp}`. Run the fine-tuning notebook first.")
    else:
        cmp_text = st.text_area(
            "Input for comparison", height=100,
            placeholder="Paste one or more sentences to compare models side by side…",
            key="cmp_input",
        )
        if st.button("Run comparison →", key="cmp_btn") and cmp_text.strip():
            c1, c2 = st.columns(2, gap="medium")

            with c1:
                st.markdown(f"<div class='panel-label'>Baseline · {base_label_cmp}</div>", unsafe_allow_html=True)
                with st.spinner("Running baseline…"):
                    pb, tb, _ = load_model(base_path_cmp)
                    ob, cb, timeb, segs_b = translate_text(pb, tb, comet_model, comet_variant, cmp_text.strip())
                st.markdown(render_segment_heatmap(segs_b, comet_hi, comet_lo), unsafe_allow_html=True)
                st.markdown(f"<div class='chip-row'><div class='chip'>Global COMET &nbsp;<b>{cb:.3f}</b></div><div class='chip'>Time &nbsp;<b>{timeb}s</b></div></div>", unsafe_allow_html=True)

            with c2:
                st.markdown("<div class='panel-label'>Fine-tuned · MeSpEn Medical</div>", unsafe_allow_html=True)
                with st.spinner("Running fine-tuned…"):
                    pf, tf, _ = load_model(ft_path_cmp)
                    of, cf, timef, segs_f = translate_text(pf, tf, comet_model, comet_variant, cmp_text.strip())
                st.markdown(render_segment_heatmap(segs_f, comet_hi, comet_lo), unsafe_allow_html=True)
                delta = round(cf - cb, 3)
                st.markdown(f"<div class='chip-row'><div class='chip'>Global COMET &nbsp;<b>{cf:.3f}</b></div><div class='chip'>Δ &nbsp;<b>{'+' if delta>=0 else ''}{delta}</b></div><div class='chip'>Time &nbsp;<b>{timef}s</b></div></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    with st.expander(f"📋  Translation History ({len(st.session_state.history)} entries)"):
        st.markdown("""
<div class='hist-row' style='color:#3d444d;font-size:0.62rem;letter-spacing:0.15em'>
    <div>SOURCE</div><div>TRANSLATION</div><div>QA COMET STATUS</div>
</div>""", unsafe_allow_html=True)

        for item in st.session_state.history[:20]:
            badge_style = f"color:{item['color']};border-color:{item['color']};background:transparent"
            st.markdown(f"""
<div class='hist-row'>
    <div class='hist-src'>{item['src']}{'…' if len(item['src'])>=140 else ''}</div>
    <div class='hist-tgt'>{item['tgt']}{'…' if len(item['tgt'])>=140 else ''}</div>
    <div>
        <span class='hist-badge' style='{badge_style}'>{item['label']} ({item['conf']:.3f})</span>
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;color:#3d444d;margin-top:4px'>
            {item['engine']} · {item.get('segments',1)} sentence{'s' if item.get('segments',1)!=1 else ''}<br>{item['time']}s
        </div>
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
    &nbsp;·&nbsp; COMET-Kiwi wmt22 · REFERENCE-FREE · PER-SENTENCE SCORING
</div>
""", unsafe_allow_html=True)