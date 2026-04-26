# Personalized-Translation-Engines
A modular Spanish-English Machine Translation system using a high-performance general baseline with domain-specific fine-tuning for Medical, Legal, and Automotive sectors. Includes a Human-in-the-Loop QA module and automated tag repair.

# Personalized Translation Engines
## Project Proposal

The objective of this project is to develop a modular AI Translation system for **iDISC**. The core strategy is to use one robust **General Translation Engine** (Spanish–English) and subsequently fine-tune it into three domain-specific versions: **Automotive**, **Legal/Regulatory**, and **Medical**.

The goal is to deliver high-quality, trustworthy translations that maintain terminology consistency, adhere to client "Brand Voice," and preserve formatting/tag integrity. The project will culminate in a functional software prototype with a user-friendly interface that balances automated efficiency with the precision required for high-stakes language.

---

## Development Plan

### 1. Model Architecture

#### General Base Model
We first pick a strong general-domain bilingual (Spanish–English) model. This General Base Model serves as the Standard Benchmark for all future improvements and ensures the project remains resource-efficient. These models can be run locally or on private servers to guarantee data privacy ("no leaking client content").

- Meta AI's NLLB-200
- Hugging Face MarianMT models

#### Domain Adaptation
We will fine-tune the base model using specialized public datasets.

| Domain | Datasets |
|---|---|
| Medical | EMEA (European Medicines Agency), PubMed (OPUS) |
| Legal | Europarl, JRC-Acquis, LEX-GLUE |
| Automotive | EuroPat (patents), Technical Manuals |

**Advantages:** Shared linguistic knowledge, faster convergence, lower compute cost, easier maintenance, and the possibility of adding new domains later (finance, marketing, cybersecurity).

#### Client-Specific Customization
Engines will be refined using client-provided **glossaries** (preferred and forbidden terms) and **style guides** to ensure the "Brand Voice" is captured.

> **Open Question — Data Scarcity & Quality:** If public datasets for a specific domain (like niche Medical or Legal sub-sectors) are limited or noisy, how will that impact the initial engine's baseline quality?

---

### 2. Quality Assurance (Human-in-the-Loop System)

A sophisticated module will be developed to determine whether a translation can be auto-accepted or requires human review.

- **Semantic Check:** Using RoBERTa-based BERTScore to ensure meaning is preserved even if the words vary.
- **Industry Standard:** Implementation of **COMET** (Cross-lingual Optimized Metric for Evaluation of Translation).
  - **Threshold:** If the COMET score falls below a specific threshold (e.g., 0.85), the segment is flagged for human review.
- **Metrics:** Results will be benchmarked using **BLEU**, **CHRF**, and **METEOR**.

> **Open Question — Human-in-the-Loop Thresholds:** What are the specific threshold scores for metrics that should trigger a human review versus an auto-acceptance?

---

### 3. Automated Repair

Instead of just detecting errors, the system will attempt to fix them. This includes:

- **Glossary-anchored corrections** to force preferred terms.
- **Targeted re-translation** of flagged segments using different prompts or models.
- **Automated passes** to repair broken tags or structures.

For this module, models must be able to edit text rather than just translate it:

- **Mistral-7B / Instruct models** — suitable for glossary-guided correction (input: translated sentence + glossary + error found by QA module).
- **T5** — well-suited for tag repair tasks.

---

### 4. End-Product Interface (The Prototype)

A web-based interface will be developed using **Streamlit** or **Gradio**, featuring:

#### File Upload Zone
A drag-and-drop area supporting `.docx`, `.pdf`, and `.html`. The back-end will use **Python-Docx** or **BeautifulSoup** to extract text while masking tags.

#### Domain Selector
A toggle switch to choose between the **Automotive**, **Legal**, or **Medical** fine-tuned engines.

#### Split-Screen Editor
| Panel | Content |
|---|---|
| Left | Source text |
| Right | Machine Translation output with dynamic confidence heatmaps |

Confidence heatmap legend:

| Color | Confidence Level | Action |
|---|---|---|
| Transparent | High confidence | Auto-accepted |
| Yellow / Orange | Mid-tier (~85%) | Automated repair or careful review |
| Red | Low-tier (50–60%) | Mandatory human intervention |

> Thresholds to be further calibrated during the testing phase.

#### Glossary Dashboard
A view showing which specific client terms were successfully applied or corrected during the Automated Repair phase.

#### Export Button
Allows the user to download the final translated file with the original formatting preserved.

---

## Requirements from iDISC

- **Test files** — to validate the engine and confirm it can handle the actual content translated daily.
- **Style guides & glossaries** — to refine and tune the engine in later stages.