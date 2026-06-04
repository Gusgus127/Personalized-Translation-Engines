#!/usr/bin/env python3
"""
evaluate_pdfs.py - Translate Spanish medical PDFs and score against reference English PDFs.

Usage examples:
    # All pairs in a folder (files must match *_es.pdf / *_en.pdf)
    python evaluate_pdfs.py --input_dir ./pdfs --model ./mespen_medical_model

    # Using a CSV that maps each source file to its reference:
    python evaluate_pdfs.py --csv pairs.csv --model ./mespen_medical_model

The script outputs results.csv with per-document BLEU, ChrF, and COMET scores,
and prints an aggregated summary.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import pdfplumber
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch

def split_text_into_token_chunks(text: str, tokenizer, max_tokens: int = 512) -> list:
    """
    Split text into chunks whose token length (after adding special tokens) is ≤ max_tokens.
    We reserve 2 tokens for the start/end markers that the pipeline will add.
    """
    # Tokenize without special tokens to get a list of IDs
    input_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    chunks = []
    # Reserve 2 slots for special tokens (e.g., <s> and </s> in Marian)
    chunk_size = max_tokens - 2
    for i in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[i:i + chunk_size]
        # Decode back to text without special tokens – the pipeline will re‑add them
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks if chunks else [text]

# ----------------------------------------------------------------------
# PDF text extraction
# ----------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF, joining pages with double newlines."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                # Normalise whitespace
                page_text = re.sub(r'\s+', ' ', page_text).strip()
                text_parts.append(page_text)
    return "\n\n".join(text_parts)

# ----------------------------------------------------------------------
# Model loading (similar to your app)
# ----------------------------------------------------------------------
def load_translation_model(model_path: str):
    """Load MarianMT model and tokenizer."""
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline("translation", model=model, tokenizer=tokenizer, device=device)
    return pipe, tokenizer, device

def translate_text(pipe, tokenizer, text: str) -> str:
    chunks = split_text_into_token_chunks(text, tokenizer, max_tokens=512)
    translated_chunks = []
    for chunk in chunks:
        out = pipe(chunk, max_length=512)
        translated_chunks.append(out[0]["translation_text"])
    return " ".join(translated_chunks)

# ----------------------------------------------------------------------
# COMET (reference-based) loading – different from Kiwi
# ----------------------------------------------------------------------
def load_comet_ref_based():
    """Load a reference-based COMET model (wmt22-comet-da)."""
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-comet-da")
    return load_from_checkpoint(model_path)

def comet_score(ref_model, sources: List[str], hypotheses: List[str], references: List[str]) -> float:
    """Compute corpus-level COMET score (higher = better, range ~0-1)."""
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    out = ref_model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
    return out.system_score  # already in [0,1]

# ----------------------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------------------
def evaluate_pairs(
    pairs: List[Tuple[str, str, str]],   # (doc_id, src_pdf, ref_pdf)
    model_path: str,
    output_csv: str,
):
    # Load translation pipeline
    print(f"Loading translation model from {model_path} ...")
    pipe, tokenizer, device = load_translation_model(model_path)
    print(f"Model loaded. Device: {'GPU' if device==0 else 'CPU'}")

    # Load reference-based COMET
    print("Loading COMET (reference-based) ...")
    comet_ref = load_comet_ref_based()
    print("COMET loaded.")

    results = []
    for doc_id, src_pdf, ref_pdf in pairs:
        print(f"\nProcessing: {doc_id}  (ES: {src_pdf}, EN: {ref_pdf})")
        t_start = time.time()

        # Extract text
        src_text = extract_text_from_pdf(src_pdf)
        ref_text = extract_text_from_pdf(ref_pdf)
        if not src_text.strip() or not ref_text.strip():
            print("  WARNING: empty text, skipping.")
            continue

        # Translate (full text as one segment, or split sentences for better scoring)
        hyp_text = translate_text(pipe, tokenizer, src_text)

        # Compute BLEU and ChrF (sentence-level is more robust; we'll use sacrebleu corpus)
        import sacrebleu
        # sacrebleu expects tokenized inputs; we'll let its internal tokenizer handle it
        bleu = sacrebleu.corpus_bleu([hyp_text], [[ref_text]]).score
        chrf = sacrebleu.corpus_chrf([hyp_text], [[ref_text]]).score

        # COMET score (reference-based) – using full texts
        comet_val = comet_score(comet_ref, [src_text], [hyp_text], [ref_text])

        elapsed = time.time() - t_start
        print(f"  BLEU: {bleu:.1f}  ChrF: {chrf:.1f}  COMET: {comet_val:.3f}  ({elapsed:.1f}s)")

        results.append({
            "doc_id": doc_id,
            "src_pdf": src_pdf,
            "ref_pdf": ref_pdf,
            "src_chars": len(src_text),
            "ref_chars": len(ref_text),
            "hyp_chars": len(hyp_text),
            "BLEU": round(bleu, 1),
            "ChrF": round(chrf, 1),
            "COMET": round(comet_val, 3),
            "time_s": round(elapsed, 1),
        })

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # Summary
    if results:
        avg_bleu = df["BLEU"].mean()
        avg_chrf = df["ChrF"].mean()
        avg_comet = df["COMET"].mean()
        print(f"=== AVERAGE OVER {len(results)} DOCUMENTS ===")
        print(f"  BLEU:  {avg_bleu:.1f}")
        print(f"  ChrF:  {avg_chrf:.1f}")
        print(f"  COMET: {avg_comet:.3f}")

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate translation model on paired PDFs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", help="Folder containing *_es.pdf and *_en.pdf pairs")
    group.add_argument("--csv", help="CSV with columns 'doc_id,src_pdf,ref_pdf'")
    parser.add_argument("--model", required=True, help="Path to fine-tuned MarianMT model")
    parser.add_argument("--output_csv", default="results.csv", help="Output CSV file name")
    args = parser.parse_args()

    pairs = []
    if args.input_dir:
        folder = Path(args.input_dir)
        # Find all Spanish files
        es_files = list(folder.glob("*_es.pdf")) + list(folder.glob("*_ES.pdf"))
        for src_path in es_files:
            stem = src_path.stem
            # Determine English counterpart
            ref_path = None
            for suffix in ["_en.pdf", "_EN.pdf"]:
                candidate = folder / stem.replace("_es", suffix).replace("_ES", suffix)
                if candidate.exists():
                    ref_path = candidate
                    break
            if ref_path is None:
                print(f"WARNING: No English pair found for {src_path}, skipping.")
                continue
            doc_id = stem.replace("_es", "").replace("_ES", "")
            pairs.append((doc_id, str(src_path), str(ref_path)))
    else:
        # CSV mode
        df = pd.read_csv(args.csv)
        for _, row in df.iterrows():
            pairs.append((row["doc_id"], row["src_pdf"], row["ref_pdf"]))

    if not pairs:
        print("No valid PDF pairs found. Exiting.")
        sys.exit(1)

    evaluate_pairs(pairs, args.model, args.output_csv)

if __name__ == "__main__":
    main()