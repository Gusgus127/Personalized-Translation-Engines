# Personalized Translation Engines
> A modular Spanish–English Machine Translation system with domain-specific fine-tuning for Medical, Legal, and Automotive sectors. Includes a Human-in-the-Loop QA module and automated tag repair.

---

## Project Proposal

The objective of this project is to develop a modular AI Translation system for **iDISC**. The core strategy is to use one robust **General Translation Engine** (Spanish–English) and subsequently fine-tune it into three domain-specific versions: **Automotive**, **Legal/Regulatory**, and **Medical**.

The goal is to deliver high-quality, trustworthy translations that maintain terminology consistency, adhere to client "Brand Voice," and preserve formatting/tag integrity. The project will culminate in a functional software prototype with a user-friendly interface that balances automated efficiency with the precision required for high-stakes language.

---

## Repository Structure

```
.
├── app.py                          # Streamlit demo interface
├── mespen-medical-finetune.ipynb   # Fine-tuning pipeline (MeSpEn dataset)
├── nllb-baseline.ipynb             # NLLB-200 baseline experiments
├── sp-test.ipynb                   # Sanity-check / scratchpad notebook
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
    └── Baseline_Report.csv         # Baseline evaluation results
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip
- (Recommended) A virtual environment

### 1. Clone the repository

```bash
git clone https://github.com/your-org/personalized-translation-engines.git
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
```

> **GPU note:** If you have a CUDA-capable GPU, install the matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) before running pip install. The app auto-detects CUDA at runtime.

---

## Running the Web App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### What you'll see

| Feature | Description |
|---|---|
| **Split-screen editor** | Source (ES) on the left, translation (EN) on the right |
| **Confidence bar** | Green / Amber / Red heatmap per segment (auto-accept / review / flag) |
| **Engine selector** | Switch between Baseline and Fine-tuned model in the sidebar |
| **Domain selector** | Medical active; Legal and Automotive stubs (coming soon) |
| **Quick examples** | 5 pre-loaded medical sentences to test immediately |
| **Comparator** | Run baseline and fine-tuned side-by-side on the same input |
| **History panel** | Full session log with QA labels, exportable as JSON |

### Engine paths expected by `app.py`

| Engine | Path |
|---|---|
| Baseline | Downloaded automatically from HuggingFace (`Helsinki-NLP/opus-mt-es-en`) |
| Fine-tuned | `./mespen_medical_model/` (must exist locally) |

If the fine-tuned model directory is missing, the app falls back gracefully to the baseline only.

---

## Running the Fine-tuning Notebook

Open `mespen-medical-finetune.ipynb` in Jupyter or upload it to Google Colab / Kaggle.

The notebook covers:

1. Downloading and loading the MeSpEn dataset (PubMed + SciELO subsets)
2. Data exploration and quality filtering
3. Building train / val / test splits
4. Tokenization for MarianMT
5. Fine-tuning with `Seq2SeqTrainer` (3 epochs, ~6–7 hrs on a T4 GPU)
6. BLEU + ChrF + COMET evaluation vs baseline

After training, save the model output to `./mespen_medical_model/` to make it available in the app.

---

## Development Plan

### 1. Model Architecture

#### General Base Model
We use **MarianMT** (`Helsinki-NLP/opus-mt-es-en`) as the base — commercially licensed (Apache 2.0), fast, and well-suited for domain fine-tuning. Models run locally or on private servers to guarantee data privacy.

> NLLB-200 was evaluated but excluded due to its CC-BY-NC license (non-commercial only).

#### Domain Adaptation

| Domain | Dataset | Status |
|---|---|---|
| Medical | MeSpEn (PubMed + SciELO), EMEA, WMT Biomedical | ✅ Fine-tuned |
| Legal | Europarl, JRC-Acquis, LEX-GLUE | 🔜 Planned |
| Automotive | EuroPat (patents), Technical Manuals | 🔜 Planned |

**Advantages:** Shared linguistic knowledge, faster convergence, lower compute cost, easier maintenance, extensible to new domains (finance, marketing, cybersecurity).

#### Client-Specific Customization
Engines will be refined using client-provided **glossaries** (preferred and forbidden terms) and **style guides** to capture the "Brand Voice."

> **Open Question — Data Scarcity & Quality:** If public datasets for a specific domain (like niche Medical or Legal sub-sectors) are limited or noisy, how will that impact the initial engine's baseline quality?

---

### 2. Quality Assurance (Human-in-the-Loop System)

A sophisticated module will determine whether a translation can be auto-accepted or requires human review.

- **Semantic Check:** RoBERTa-based BERTScore to ensure meaning is preserved.
- **Industry Standard:** COMET (Cross-lingual Optimized Metric for Evaluation of Translation).
  - **Threshold:** Segments scoring below 0.85 are flagged for human review.
- **Metrics:** BLEU, ChrF, METEOR for benchmarking.

> **Open Question — Human-in-the-Loop Thresholds:** What are the specific threshold scores for metrics that should trigger human review vs auto-acceptance?

---

### 3. Automated Repair

Instead of just detecting errors, the system will attempt to fix them:

- **Glossary-anchored corrections** — force preferred terms.
- **Targeted re-translation** of flagged segments using different prompts or models.
- **Automated tag repair** — fix broken XML/HTML structures.

Models for this module:
- **Mistral-7B / Instruct** — glossary-guided correction (input: translated sentence + glossary + error)
- **T5** — tag repair tasks

---

### 4. End-Product Interface (The Prototype)

A web-based interface built in **Streamlit**, featuring:

#### File Upload Zone
Drag-and-drop supporting `.docx`, `.pdf`, and `.html`. Back-end uses **Python-Docx** / **BeautifulSoup** to extract text while masking tags.

#### Domain Selector
Toggle between **Automotive**, **Legal**, and **Medical** fine-tuned engines.

#### Split-Screen Editor

| Panel | Content |
|---|---|
| Left | Source text |
| Right | MT output with dynamic confidence heatmaps |

Confidence legend:

| Color | Level | Action |
|---|---|---|
| 🟢 Green | ≥ 80% | Auto-accepted |
| 🟡 Amber | 60–80% | Automated repair / careful review |
| 🔴 Red | < 60% | Mandatory human intervention |

#### Glossary Dashboard
Shows which client terms were applied or corrected during the Automated Repair phase.

#### Export Button
Download the final translated file with original formatting preserved.

---

## Requirements from iDISC

- **Test files** — to validate the engine on real daily content.
- **Style guides & glossaries** — to refine and tune the engine in later stages.

---

## Roadmap

- [x] Baseline evaluation (MarianMT ES→EN)
- [x] MeSpEn dataset pipeline
- [x] Medical domain fine-tuning
- [x] Streamlit demo interface (v0.1)
- [ ] EN→ES direction fine-tuning
- [ ] COMET-based QA module integration
- [ ] File upload (.docx / .pdf / .html)
- [ ] Glossary dashboard
- [ ] Legal domain fine-tuning
- [ ] Automotive domain fine-tuning
- [ ] Automated repair module (Mistral-7B + T5)
- [ ] Export with formatting preservation