# Personalized Translation Engines
> A modular Spanish–English Machine Translation system with domain-specific fine-tuning for Medical, Legal, and Automotive sectors. Includes a Human-in-the-Loop QA module with COMET-based confidence scoring, per-segment heatmaps, and a glossary correction pipeline.

---

## Project Overview

The objective of this project is to develop a modular AI translation system for **iDISC**. The core strategy is to use one robust **General Translation Engine** (Spanish–English) and fine-tune it into domain-specific versions: **Automotive**, **Legal/Regulatory**, and **Medical**.

The system delivers high-quality, trustworthy translations that maintain terminology consistency, adhere to client "Brand Voice," and preserve formatting integrity. The prototype features a fully functional Streamlit interface that balances automated efficiency with the precision required for high-stakes language.

---

## Repository Structure

```
.
├── app.py                          # Streamlit demo interface
├── glossary.py                     # Modular glossary, terminology rules, and text utilities
├── evaluate_pdfs.py                # PDF translation evaluation pipeline (BLEU, ChrF, COMET)
├── mespen-medical-finetune.ipynb   # Fine-tuning pipeline (MeSpEn / PubMed dataset)
├── mespen_data/                    # Raw MeSpEn dataset files
│   └── medlineplus_extracted/      # Extracted XML health topic files
├── mespen_medical_model/           # Fine-tuned MarianMT model (ES→EN)
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── source.spm / target.spm
│   └── checkpoint-96/              # Intermediate training checkpoint
└── results/
    └── Baseline_Report.csv         # Baseline evaluation results (NLLB-200 3.3B)
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip
- (Recommended) A virtual environment

### 1. Clone the repository

```bash
git clone https://github.com/Gusgus127/Personalized-Translation-Engines
cd personalized-translation-engines
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contents:

```
streamlit>=1.32.0
transformers>=4.40.0
torch>=2.2.0
sentencepiece>=0.2.0
sacrebleu>=2.4.0
evaluate>=0.4.0
unbabel-comet>=2.2.0
pdfplumber>=0.11.0
pandas>=2.0.0
```

---

## Running the Web App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Features

| Feature | Description |
|---|---|
| **Split-screen editor** | Source (ES or EN) on the left, translation on the right |
| **Direction toggle** | Switch between ES → EN and EN → ES in the sidebar |
| **Engine selector** | Switch between Baseline (`opus-mt-es-en`) and Fine-tuned model |
| **COMET confidence bar** | Per-translation score with Green / Amber / Red classification |
| **Segment heatmap** | Per-sentence confidence coloring directly on the output text |
| **Segment editing mode** | Inline editor for flagged segments; auto-accept segments shown as text inputs |
| **Glossary correction** | One-click application of brand, clinical, technical, and style rules with a detailed report |
| **Quick examples** | 5 pre-loaded medical sentences in each direction |
| **Baseline vs Fine-tuned comparator** | Side-by-side comparison on any input text |
| **Translation history** | Full session log with QA labels, exportable as JSON |

### Engine paths expected by `app.py`

| Engine | Path |
|---|---|
| Baseline ES→EN | Downloaded automatically from HuggingFace (`Helsinki-NLP/opus-mt-es-en`) |
| Baseline EN→ES | Downloaded automatically from HuggingFace (`Helsinki-NLP/opus-mt-en-es`) |
| Fine-tuned ES→EN | `./mespen_medical_model/` (must exist locally) |
| Fine-tuned EN→ES | `./mespen_medical_model_en_es/` (must exist locally) |

If a fine-tuned model directory is missing, the app falls back gracefully to the baseline only.

### COMET QA Thresholds

| Color | Score | Action |
|---|---|---|
|  Transparent | ≥ 0.85 | AUTO-ACCEPT |
| 🟡 Amber | 0.70 – 0.85 | Review recommended |
| 🔴 Red | < 0.70 | Mandatory human intervention |

---

## Running the Fine-tuning Notebook

Open `mespen-medical-finetune.ipynb` in Jupyter or upload it to Google Colab / Kaggle.

The notebook covers:

1. Downloading the MeSpEn dataset (PubMed subset from Zenodo)
2. XML extraction and sentence-pair parsing
3. Data filtering (length, deduplication) and train/eval/test splits
4. Tokenization for MarianMT (`max_length=128`)
5. Fine-tuning with `Seq2SeqTrainer` (1 epoch, batch size 32, lr 2e-5)
6. Training loss curve visualization
7. BLEU comparison: baseline vs fine-tuned on the held-out test set
8. COMET evaluation using `wmt22-comet-da`

After training, the model is saved to `./mespen_medical_model/` and is immediately available in `app.py`.

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | `Helsinki-NLP/opus-mt-es-en` |
| Dataset | MeSpEn PubMed subset (~500 pairs after filtering) |
| Max sequence length | 128 tokens |
| Batch size | 32 (train) / 8 (eval) |
| Learning rate | 2e-5 |
| Epochs | 1 |
| Mixed precision | fp16 (GPU only) |

---

## Evaluating on PDF Documents

`evaluate_pdfs.py` translates Spanish medical PDFs and scores them against reference English PDFs using BLEU, ChrF, and COMET.

```bash
# All matched pairs in a folder (files must match *_es.pdf / *_en.pdf)
python evaluate_pdfs.py --input_dir ./pdfs --model ./mespen_medical_model

# Using a CSV mapping source files to references
python evaluate_pdfs.py --csv pairs.csv --model ./mespen_medical_model --output_csv results.csv
```

The CSV mapping file (`pairs.csv`) must have columns: `doc_id`, `src_pdf`, `ref_pdf`.

Output `results.csv` includes per-document BLEU, ChrF, COMET, character counts, and processing time, plus an aggregated summary printed to stdout.

---

## Glossary Module (`glossary.py`)

`glossary.py` is a standalone module imported by both `app.py` and `evaluate_pdfs.py`. It provides:

- **`GLOSSARY`** — preferred and forbidden terms for ES→EN and EN→ES directions, organised by category: Brand, Clinical, Technical, Style.
- **`apply_glossary(text, direction, enabled_categories)`** — applies substitutions with regex word-boundary matching and returns `(corrected_text, list_of_changes)`.
- **`clean_extracted_text(text)`** — normalises raw pdfplumber output by removing table-of-contents dot leaders, repeated lines, and excessive blank lines.

Glossary categories and their scope:

| Category | Examples |
|---|---|
| Brand | "hearing device" → "hearing aid"; full Unitron product names |
| Clinical | "eardrum" → "tympanic membrane"; person-first language |
| Technical | "turn on" → "activate"; Bluetooth pairing qualifiers |
| Style | "make sure" → "ensure"; removing informal filler |

---

> Note: NLLB-200 was excluded from the final system due to its CC-BY-NC licence (non-commercial only). The production system uses MarianMT (Apache 2.0).

---

## Model Architecture

### Base Model
**MarianMT** (`Helsinki-NLP/opus-mt-es-en` / `opus-mt-en-es`) — Apache 2.0 licensed, fast inference, runs fully locally for data privacy.

### Domain Adaptation

| Domain | Dataset | Direction | Status |
|---|---|---|---|
| Medical | MeSpEn (PubMed + MedlinePlus), EMEA | ES→EN | ✅ Fine-tuned |
| Medical | MeSpEn (PubMed + MedlinePlus), EMEA | EN→ES | ✅ Fine-tuned |
| Legal | Europarl, JRC-Acquis, LEX-GLUE | ES↔EN | 🔜 Planned |
| Automotive | EuroPat, Technical Manuals | ES↔EN | 🔜 Planned |

---

## Roadmap

- [x] NLLB-200 baseline evaluation
- [x] MeSpEn dataset pipeline (PubMed subset)
- [x] Medical domain fine-tuning (ES→EN)
- [x] EN→ES fine-tuned model
- [x] Streamlit demo interface — split-screen, heatmap, history
- [x] COMET QA module with per-segment confidence scoring
- [x] Segment editing mode for flagged translations
- [x] Glossary correction pipeline with application report
- [x] Baseline vs fine-tuned side-by-side comparator
- [x] PDF evaluation script (`evaluate_pdfs.py`)
- [x] Modular glossary and text utilities (`glossary.py`)
- [x] File upload (.pdf) in the Streamlit UI (text extraction only)
- [ ] Legal domain fine-tuning
- [ ] Automotive domain fine-tuning
- [ ] Automated repair module (Mistral-7B + T5)
- [ ] Export with formatting preservation

---

## Requirements from iDISC

- **Test files** — to validate the engine on real daily content.
- **Style guides & glossaries** — to refine and tune the engine in later stages.