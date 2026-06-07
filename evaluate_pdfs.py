#!/usr/bin/env python3
"""
evaluate_pdfs.py - Evaluate medical translation models in BOTH directions (ES->EN and EN->ES).

Usage examples:
    # Point to your primary ES->EN local folder. It will look for its EN->ES sibling automatically:
    python evaluate_pdfs.py --input_dir ./pdfs --model ./mespen_medical_model

    # Or let it search your root path automatically:
    python evaluate_pdfs.py --input_dir ./pdfs
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

try:
    from glossary import clean_extracted_text
except ImportError:
    def clean_extracted_text(text: str) -> str:
        """
        Fallback implementation matching the app's clean_extracted_text logic.
        Removes alignment dots, cleans up spacing, and filters extraction artifacts.
        """
        if not text:
            return ""
        text = re.sub(r'\.{2,}', ' ', text)
        cleaned_lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line or re.match(r'^[._\-\s•]+$', line):
                continue
            line = re.sub(r'\s+', ' ', line)
            cleaned_lines.append(line)
        return "\n\n".join(cleaned_lines)

def split_into_sentences(text: str) -> list:
    """Split clean text into individual sentences using punctuation boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# ----------------------------------------------------------------------
# PDF text extraction
# ----------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract, filter, and normalize plain text from a PDF."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
                
    raw_combined = "\n".join(text_parts)
    return clean_extracted_text(raw_combined)

# ----------------------------------------------------------------------
# Model loading & Translation Strategy
# ----------------------------------------------------------------------
def load_translation_model(model_path: str):
    """Load MarianMT model and tokenizer dynamically (handles local path or HF repo)."""
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline("translation", model=model, tokenizer=tokenizer, device=device)
    return pipe, tokenizer, device

def translate_text(pipe, tokenizer, text: str) -> str:
    """Translate text sentence-by-sentence to avoid sequence length truncation."""
    sentences = split_into_sentences(text)
    translated_chunks = []
    max_chunk_tokens = 400
    
    for sent in sentences:
        tokens = tokenizer.encode(sent, add_special_tokens=False)
        if len(tokens) > max_chunk_tokens:
            sent = tokenizer.decode(tokens[:max_chunk_tokens], skip_special_tokens=True)
            
        if not sent.strip():
            continue
            
        try:
            out = pipe(sent, max_length=512)
            translated = out[0].get("translation_text") or out[0].get("generated_text") or sent
            translated_chunks.append(translated.strip())
        except Exception:
            translated_chunks.append(sent.strip())
            
    return " ".join(translated_chunks)

# ----------------------------------------------------------------------
# COMET (reference-based) loading
# ----------------------------------------------------------------------
def load_comet_ref_based():
    """Load a reference-based COMET model (wmt22-comet-da)."""
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-comet-da")
    return load_from_checkpoint(model_path)

def comet_score(ref_model, sources: List[str], hypotheses: List[str], references: List[str]) -> float:
    """Compute corpus-level reference-based COMET score."""
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    out = ref_model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
    return out.system_score

# ----------------------------------------------------------------------
# Main bidirectional runner
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate translation model bidirectionally on paired PDFs.")
    parser.add_argument("--input_dir", required=True, help="Folder containing matching *_es.pdf and *_en.pdf pairs")
    parser.add_argument("--model", default="./mespen_medical_model", help="Path to your custom fine-tuned model directory")
    parser.add_argument("--output_csv", default="results.csv", help="Output CSV file name")
    args = parser.parse_args()

    folder = Path(args.input_dir)
    es_files = list(folder.glob("*_es.pdf")) + list(folder.glob("*_ES.pdf"))
    
    # 1. Discover all paired documents
    pairs = []
    for src_path in es_files:
        stem = src_path.name
        ref_name = re.sub(r'_(es|ES)\.pdf$', '_en.pdf', stem)
        ref_path = folder / ref_name
        
        if not ref_path.exists():
            ref_name = re.sub(r'_(es|ES)\.pdf$', '_EN.pdf', stem)
            ref_path = folder / ref_name
            
        if not ref_path.exists():
            print(f"WARNING: No English pair discovered for {src_path.name}, skipping.")
            continue
            
        doc_id = re.sub(r'_(es|ES)\.pdf$', '', src_path.name)
        pairs.append((doc_id, src_path, ref_path))

    if not pairs:
        print("No matched PDF pairs found. Check your file names inside the folder path.")
        sys.exit(1)

    # 2. Derive path destinations based on inputs
    given_path = Path(args.model).resolve()
    
    # Intelligently assume root parent directory locations
    if given_path.name in ["mespen_medical_model", "mespen_medical_model_en_es"]:
        base_dir = given_path.parent
    else:
        base_dir = given_path

    es_en_local = base_dir / "mespen_medical_model"
    en_es_local = base_dir / "mespen_medical_model_en_es"

    # Set paths dynamically, fallback to standard repositories if local directory doesn't exist
    es_en_checkpoint = str(es_en_local) if es_en_local.is_dir() else "Helsinki-NLP/opus-mt-es-en"
    en_es_checkpoint = str(en_es_local) if en_es_local.is_dir() else "Helsinki-NLP/opus-mt-en-es"

    directions_to_run = [
        {"name": "ES → EN", "checkpoint": es_en_checkpoint},
        {"name": "EN → ES", "checkpoint": en_es_checkpoint}
    ]

    print("\nLoading reference-based COMET model...")
    comet_ref = load_comet_ref_based()
    print("COMET framework ready.")

    import sacrebleu
    all_results = []

    # 3. Process execution loop
    for run in directions_to_run:
        print(f"\n========================================================")
        print(f" STARTING EVALUATION FLOW: {run['name']}")
        print(f"========================================================")
        print(f"Resolving engine checkpoint: using `{run['checkpoint']}`")
        
        try:
            pipe, tokenizer, device = load_translation_model(run['checkpoint'])
        except Exception as e:
            print(f"ERROR: Could not load engine pipeline for {run['name']}: {e}. Skipping flow.")
            continue

        for doc_id, es_path, en_path in pairs:
            if run["name"] == "ES → EN":
                input_pdf, reference_pdf = es_path, en_path
            else:
                input_pdf, reference_pdf = en_path, es_path

            print(f" -> Doc: {doc_id} [{run['name']}]")
            t_start = time.time()

            src_text = extract_text_from_pdf(str(input_pdf))
            ref_text = extract_text_from_pdf(str(reference_pdf))

            if not src_text.strip() or not ref_text.strip():
                print("    WARNING: Found empty text blocks post-cleaning extraction, skipping.")
                continue

            hyp_text = translate_text(pipe, tokenizer, src_text)

            # Performance scoring evaluation
            bleu = sacrebleu.corpus_bleu([hyp_text], [[ref_text]]).score
            chrf = sacrebleu.corpus_chrf([hyp_text], [[ref_text]]).score
            comet_val = comet_score(comet_ref, [src_text], [hyp_text], [ref_text])
            elapsed = time.time() - t_start

            print(f"    BLEU: {bleu:.1f} | ChrF: {chrf:.1f} | COMET: {comet_val:.3f} ({elapsed:.1f}s)")

            all_results.append({
                "doc_id": doc_id,
                "direction": run["name"],
                "src_pdf": str(input_pdf),
                "ref_pdf": str(reference_pdf),
                "src_chars": len(src_text),
                "ref_chars": len(ref_text),
                "hyp_chars": len(hyp_text),
                "BLEU": round(bleu, 1),
                "ChrF": round(chrf, 1),
                "COMET": round(comet_val, 3),
                "time_s": round(elapsed, 1),
            })
            
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Export consolidated metrics report
    if not all_results:
        print("\nNo translation workflows were processed successfully. Nothing written to file.")
        sys.exit(1)

    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nConsolidated bidirectional data exported to: {args.output_csv}")

    # 5. Output macro summary splits
    print("\n" + "="*40 + "\n FINAL METRICS DISPATCH BREAKDOWN\n" + "="*40)
    for direction_name in ["ES → EN", "EN → ES"]:
        sub_df = df[df["direction"] == direction_name]
        if not sub_df.empty:
            print(f"\n📊 FLOW DIRECTION SUMMARY: {direction_name} ({len(sub_df)} Docs)")
            print(f"  • Mean BLEU:  {sub_df['BLEU'].mean():.1f}")
            print(f"  • Mean ChrF:  {sub_df['ChrF'].mean():.1f}")
            print(f"  • Mean COMET: {sub_df['COMET'].mean():.3f}")
            print(f"  • Total Time: {sub_df['time_s'].sum():.1f}s")
        else:
            print(f"\n📊 NO LOGS CAPTURED FOR FLOW: {direction_name}")

if __name__ == "__main__":
    main()