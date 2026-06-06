"""
glossary.py
Modular Data and Helper Utilities for iDISC MedTranslate
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# DICTIONARIES — Preferred terms, brand voice, style rules
# ─────────────────────────────────────────────────────────────────────────────
GLOSSARY = {
    # ── ES → EN entries ───────────────────────────────────────────────────────
    "es_en": [
        {"wrong": "hearing device",        "right": "hearing aid",           "category": "Brand",      "note": "Unitron preferred term"},
        {"wrong": "hearing instrument",    "right": "hearing aid",           "category": "Brand",      "note": "Unitron preferred term"},
        {"wrong": "auditory device",       "right": "hearing aid",           "category": "Brand",      "note": "Unitron preferred term"},
        {"wrong": "Remote Plus app",       "right": "Unitron Remote Plus app","category": "Brand",     "note": "Always use full product name"},
        {"wrong": "the app",               "right": "the application",       "category": "Style",      "note": "Formal register in user guides"},
        {"wrong": "ear drum",              "right": "tympanic membrane",     "category": "Clinical",   "note": "Anatomical standard"},
        {"wrong": "eardrum",               "right": "tympanic membrane",     "category": "Clinical",   "note": "Anatomical standard"},
        {"wrong": "ringing in the ears",   "right": "tinnitus",              "category": "Clinical",   "note": "Use medical term"},
        {"wrong": "hearing loss",          "right": "hearing impairment",    "category": "Clinical",   "note": "Person-first preferred"},
        {"wrong": "deaf",                  "right": "person with hearing impairment", "category": "Clinical", "note": "Person-first language"},
        {"wrong": "battery",               "right": "cell",                  "category": "Clinical",   "note": "Unitron technical manual term"},
        {"wrong": "please",                "right": "",                       "category": "Style",      "note": "Remove filler in technical docs"},
        {"wrong": "make sure",             "right": "ensure",                "category": "Style",      "note": "Formal register"},
        {"wrong": "get in touch",          "right": "contact",               "category": "Style",      "note": "Formal register"},
        {"wrong": "find out",              "right": "determine",             "category": "Style",      "note": "Formal register"},
        {"wrong": "a lot of",              "right": "numerous",              "category": "Style",      "note": "Formal register"},
        {"wrong": "shows up",              "right": "appears",               "category": "Style",      "note": "Formal register"},
        {"wrong": "pairing",               "right": "Bluetooth pairing",     "category": "Technical",  "note": "Always qualify pairing type"},
        {"wrong": "connecting",            "right": "pairing",               "category": "Technical",  "note": "Use pairing for BT connections"},
        {"wrong": "turn on",               "right": "activate",              "category": "Technical",  "note": "UI action wording"},
        {"wrong": "turn off",              "right": "deactivate",            "category": "Technical",  "note": "UI action wording"},
        {"wrong": "mute",                  "right": "silence",               "category": "Technical",  "note": "Matches UI button label"},
        {"wrong": "settings",              "right": "adjustments",           "category": "Technical",  "note": "Unitron UI terminology"},
    ],
    # ── EN → ES entries ───────────────────────────────────────────────────────
    "en_es": [
        {"wrong": "hearing device",        "right": "audífono",              "category": "Brand",      "note": "Término preferido de Unitron"},
        {"wrong": "hearing instrument",    "right": "audífono",              "category": "Brand",      "note": "Término preferido de Unitron"},
        {"wrong": "the app",               "right": "la aplicación",         "category": "Style",      "note": "Registro formal en guías de usuario"},
        {"wrong": "Remote Plus app",       "right": "aplicación Unitron Remote Plus", "category": "Brand", "note": "Nombre completo del producto"},
        {"wrong": "ringing in the ears",   "right": "tinnitus",              "category": "Clinical",   "note": "Usar término médico"},
        {"wrong": "hearing loss",          "right": "pérdida auditiva",      "category": "Clinical",   "note": "Término estándar"},
        {"wrong": "battery",               "right": "pila",                  "category": "Technical",  "note": "Terminología manual técnico Unitron"},
        {"wrong": "make sure",             "right": "asegúrese de",          "category": "Style",      "note": "Registro formal"},
        {"wrong": "please",                "right": "",                       "category": "Style",      "note": "Eliminar en documentos técnicos"},
        {"wrong": "turn on",               "right": "activar",               "category": "Technical",  "note": "Vocabulario de UI"},
        {"wrong": "turn off",              "right": "desactivar",            "category": "Technical",  "note": "Vocabulario de UI"},
        {"wrong": "settings",              "right": "ajustes",               "category": "Technical",  "note": "Terminología UI Unitron"},
        {"wrong": "pairing",               "right": "emparejamiento Bluetooth", "category": "Technical", "note": "Siempre calificar tipo de conexión"},
    ],
}

GLOSSARY_CATEGORIES = ["Brand", "Clinical", "Technical", "Style"]
CATEGORY_COLORS = {
    "Brand":    "#58a6ff",
    "Clinical": "#3fb950",
    "Technical":"#d29922",
    "Style":    "#bc8cff",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER ACTIONS
# ─────────────────────────────────────────────────────────────────────────────
def apply_glossary(text: str, direction: str, enabled_categories: list) -> tuple[str, list]:
    """
    Apply glossary substitutions to `text`.
    Returns (corrected_text, list_of_applied_changes).
    """
    key     = "es_en" if direction == "ES → EN" else "en_es"
    entries = [e for e in GLOSSARY[key] if e["category"] in enabled_categories]
    applied = []

    for entry in entries:
        wrong = entry["wrong"]
        right = entry["right"]
        pattern = re.compile(r'(?<![\w])' + re.escape(wrong) + r'(?![\w])', re.IGNORECASE)

        matches = pattern.findall(text)
        if not matches:
            continue

        if right == "":
            new_text = pattern.sub("", text)
            new_text = re.sub(r'  +', ' ', new_text).strip()
        else:
            new_text = pattern.sub(right, text)

        if new_text != text:
            applied.append({
                "wrong":    wrong,
                "right":    right,
                "category": entry["category"],
                "note":     entry["note"],
                "count":    len(matches),
            })
            text = new_text

    return text, applied


def clean_extracted_text(text: str) -> str:
    """Clean up raw text extracted via pdfplumber."""
    lines = text.splitlines()
    cleaned = []
    prev_line = None
    repeat_count = 0
    for line in lines:
        line = line.rstrip()
        if re.search(r'[.·\s]{6,}', line) and re.search(r'\d+\s*$', line):
            continue
        if re.fullmatch(r'\s*\d{1,3}\s*', line):
            continue
        if line == prev_line:
            repeat_count += 1
            if repeat_count >= 2:
                continue
        else:
            repeat_count = 0
        prev_line = line
        cleaned.append(line)
        
    result = re.sub(r'(\n\s*){3,}', '\n\n', '\n'.join(cleaned))
    return result.strip()