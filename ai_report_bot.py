#!/usr/bin/env python3
"""
SB-ITM AI Report Bot
Reads the latest Morning Scan CSV, Bridge Bot log, and Regime Bot log,
calls OpenAI, and generates institutional EN + FR PDF reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from datetime import date, datetime
from pathlib import Path

import yaml
from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


# ── Brand colors ──────────────────────────────────────────────────────────────

DARK_BLUE  = colors.HexColor("#193F56")
ORANGE     = colors.HexColor("#F26022")
WHITE      = colors.white
LIGHT_GRAY = colors.HexColor("#F2F2F2")
MID_GRAY   = colors.HexColor("#CCCCCC")
TEXT_DARK  = colors.HexColor("#222222")
TEXT_MID   = colors.HexColor("#555555")


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_folder: str) -> logging.Logger:
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_folder) / f"{date.today().strftime('%Y_%m_%d')}_AI_Report_Bot.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── File discovery ────────────────────────────────────────────────────────────

def find_latest_file(folder: str, pattern: str) -> Path:
    files = sorted(Path(folder).glob(pattern), reverse=True)
    if not files:
        raise FileNotFoundError(f"No file matching '{pattern}' in {folder}")
    return files[0]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_scan_csv(path: Path, score_threshold: int, max_candidates: int) -> tuple[list[dict], int, int]:
    all_count  = 0
    candidates = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_count += 1
            if row.get("CandidateFlag") != "TRUE":
                continue
            try:
                score = int(row["Score"])
            except (ValueError, KeyError):
                continue
            if score >= score_threshold:
                candidates.append(row)

    candidates.sort(key=lambda r: int(r.get("Score", 0)), reverse=True)
    return candidates[:max_candidates], len(candidates), all_count


def load_log_text(path: Path, max_lines: int = 150) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])


def extract_regime(regime_log: str) -> str:
    for line in reversed(regime_log.splitlines()):
        m = re.search(r"market_mode updated:.*?->\s*(\w+)", line)
        if m:
            return m.group(1).upper()
        m = re.search(r"Regime:\s*(\w+)", line)
        if m:
            return m.group(1).upper()
    return "UNKNOWN"


# ── OpenAI ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a professional financial analyst writing institutional morning reports for SB-ITM clients.
Generate a structured JSON response with exactly two top-level keys: "en" (English) and "fr" (French).
Each must contain exactly these five keys:
  regime_summary       — 2-3 sentences on the detected market regime and what it means for today
  market_context       — 2-3 sentences on current market conditions based on the indicator data
  candidates_analysis  — 3-5 sentences analyzing the top scan candidates collectively
  bridge_summary       — 2-3 sentences on what Bridge Bot selected and why it is relevant
  key_levels           — 1-2 sentences on key price levels to monitor today

Rules:
- Write in a concise, professional, institutional tone.
- Do NOT make price predictions or give buy/sell advice.
- Do NOT include markdown, bullet points, or formatting inside the text values.
- Respond with valid JSON only — no preamble, no explanation outside the JSON.
"""


def build_prompt(candidates: list[dict], regime_log: str, bridge_log: str, report_date: str) -> str:
    lines = []
    for r in candidates:
        dist = r.get("SupportDistance", "")
        try:
            dist_str = f"{float(dist)*100:.1f}%"
        except (ValueError, TypeError):
            dist_str = str(dist)
        lines.append(
            f"  {r.get('Symbol','?')} | Score: {r.get('Score','?')} "
            f"| Close: {r.get('Close','?')} | Support: {r.get('Support','?')} "
            f"| Resistance: {r.get('Resistance','?')} | Dist to Support: {dist_str} "
            f"| Comment: {r.get('Comment','')}"
        )
    cand_text = "\n".join(lines) if lines else "  No candidates passed the scan filters today."

    return (
        f"SB-ITM Morning Report Data — {report_date}\n\n"
        f"=== REGIME BOT LOG ===\n{regime_log.strip()}\n\n"
        f"=== MORNING SCAN CANDIDATES ({len(candidates)} selected) ===\n{cand_text}\n\n"
        f"=== BRIDGE BOT LOG ===\n{bridge_log.strip()}\n\n"
        "Generate the institutional morning report JSON as instructed."
    )


def call_openai(client: OpenAI, prompt: str, cfg: dict) -> dict:
    oa = cfg["openai"]
    resp = client.chat.completions.create(
        model=str(oa["model"]),
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
        timeout=int(oa.get("timeout", 60)),
        max_tokens=int(oa.get("max_tokens", 2000)),
    )
    return json.loads(resp.choices[0].message.content)


# ── PDF styles ────────────────────────────────────────────────────────────────

def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "hdr_brand": ParagraphStyle(
            "hdr_brand", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=18,
            textColor=ORANGE,
        ),
        "hdr_title": ParagraphStyle(
            "hdr_title", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=12,
            textColor=WHITE, alignment=TA_RIGHT,
        ),
        "hdr_date": ParagraphStyle(
            "hdr_date", parent=base["Normal"],
            fontName="Helvetica", fontSize=9,
            textColor=colors.HexColor("#AACCDD"), alignment=TA_RIGHT,
        ),
        "section": ParagraphStyle(
            "section", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=10,
            textColor=DARK_BLUE, spaceBefore=10, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontName="Helvetica", fontSize=9,
            textColor=TEXT_DARK, leading=14,
        ),
        "th": ParagraphStyle(
            "th", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=8,
            textColor=WHITE,
        ),
        "td": ParagraphStyle(
            "td", parent=base["Normal"],
            fontName="Helvetica", fontSize=8,
            textColor=TEXT_DARK,
        ),
        "footer": ParagraphStyle(
            "footer", parent=base["Normal"],
            fontName="Helvetica", fontSize=7,
            textColor=TEXT_MID, alignment=TA_CENTER,
        ),
    }


# ── PDF building blocks ───────────────────────────────────────────────────────

def _build_header(title: str, report_date: str, st: dict) -> Table:
    left  = Paragraph("SB<font color='#F26022'>-</font>ITM", st["hdr_brand"])
    right = [
        Paragraph(title, st["hdr_title"]),
        Paragraph(report_date, st["hdr_date"]),
    ]
    data = [[left, right]]
    t = Table(data, colWidths=[6*cm, 12*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), DARK_BLUE),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING",   (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 12),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _build_regime_badge(regime: str) -> Table:
    regime_colors = {
        "BULL":      colors.HexColor("#1A7F3C"),
        "DEFENSIVE": colors.HexColor("#C0392B"),
        "RANGE":     colors.HexColor("#D68910"),
    }
    badge_color = regime_colors.get(regime, DARK_BLUE)
    st = _styles()
    label = Paragraph(
        "<font color='white'><b>MARKET REGIME</b></font>",
        ParagraphStyle("rl", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)
    )
    value = Paragraph(
        f"<font color='white'><b>{regime}</b></font>",
        ParagraphStyle("rv", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE, alignment=TA_CENTER)
    )
    data = [[label, value]]
    t = Table(data, colWidths=[4*cm, 14*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (0, 0), DARK_BLUE),
        ("BACKGROUND",   (1, 0), (1, 0), badge_color),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _build_candidates_table(candidates: list[dict], st: dict) -> Table:
    headers = ["Symbol", "Score", "Close", "Support", "Resistance", "Dist %", "Comment"]
    rows    = [[Paragraph(h, st["th"]) for h in headers]]

    for i, r in enumerate(candidates):
        dist = r.get("SupportDistance", "")
        try:
            dist_str = f"{float(dist)*100:.1f}%"
        except (ValueError, TypeError):
            dist_str = ""

        rows.append([
            Paragraph(str(r.get("Symbol", "")),     st["td"]),
            Paragraph(str(r.get("Score", "")),       st["td"]),
            Paragraph(str(r.get("Close", "")),       st["td"]),
            Paragraph(str(r.get("Support", "")),     st["td"]),
            Paragraph(str(r.get("Resistance", "")),  st["td"]),
            Paragraph(dist_str,                       st["td"]),
            Paragraph(str(r.get("Comment", "")),     st["td"]),
        ])

    col_widths = [2*cm, 1.3*cm, 1.8*cm, 1.8*cm, 2.2*cm, 1.5*cm, 7.4*cm]
    style = [
        ("BACKGROUND",   (0, 0), (-1, 0), DARK_BLUE),
        ("GRID",         (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))

    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


# ── PDF generation ────────────────────────────────────────────────────────────

_SECTION_LABELS = {
    "en": {
        "regime_summary":      "Market Regime Summary",
        "market_context":      "Market Context",
        "candidates_analysis": "Candidate Analysis",
        "bridge_summary":      "Bridge Bot Selection",
        "key_levels":          "Key Levels to Watch",
        "candidates_table":    "Top Scan Candidates",
        "no_candidates":       "No candidates passed the scan filters today.",
        "footer":              "Generated by SB-ITM AI Report Bot",
        "confidential":        "Confidential",
    },
    "fr": {
        "regime_summary":      "Résumé du Régime de Marché",
        "market_context":      "Contexte de Marché",
        "candidates_analysis": "Analyse des Candidats",
        "bridge_summary":      "Sélection Bridge Bot",
        "key_levels":          "Niveaux Clés à Surveiller",
        "candidates_table":    "Meilleurs Candidats du Scan",
        "no_candidates":       "Aucun candidat n'a passé les filtres du scan aujourd'hui.",
        "footer":              "Généré par SB-ITM AI Report Bot",
        "confidential":        "Confidentiel",
    },
}


def generate_pdf(
    output_path: Path,
    lang: str,
    title: str,
    report_date: str,
    candidates: list[dict],
    ai_content: dict,
    regime: str,
):
    st     = _styles()
    labels = _SECTION_LABELS.get(lang, _SECTION_LABELS["en"])
    story  = []

    # Header
    story.append(_build_header(title, report_date, st))
    story.append(Spacer(1, 0.4*cm))

    # Regime badge
    story.append(_build_regime_badge(regime))
    story.append(Spacer(1, 0.5*cm))

    # AI text sections
    for key in ("regime_summary", "market_context", "candidates_analysis", "bridge_summary", "key_levels"):
        text = ai_content.get(key, "").strip()
        if not text:
            continue
        story.append(Paragraph(labels[key], st["section"]))
        story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=5))
        story.append(Paragraph(text, st["body"]))
        story.append(Spacer(1, 0.25*cm))

    # Candidates table
    story.append(Paragraph(labels["candidates_table"], st["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=5))
    if candidates:
        story.append(_build_candidates_table(candidates, st))
    else:
        story.append(Paragraph(labels["no_candidates"], st["body"]))

    # Footer
    story.append(Spacer(1, 0.8*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GRAY, spaceAfter=4))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph(
        f"{labels['footer']} — {ts} — {labels['confidential']}",
        st["footer"]
    ))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm,  bottomMargin=1.5*cm,
    )
    doc.build(story)


# ── JSON archive ──────────────────────────────────────────────────────────────

def save_archive(data: dict, folder: str, date_str: str) -> Path:
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    out = p / f"{date_str}_AI_Report_Bot.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SB-ITM AI Report Bot")
    parser.add_argument("--config",  default="config/ai_report_bot.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Detect files and log without calling OpenAI")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging(cfg["outputs"]["log_folder"])
    log.info("AI Report Bot started")

    dry_run = args.dry_run or cfg.get("dry_run", False)
    if dry_run:
        log.info("DRY RUN mode — OpenAI will not be called, no PDFs will be generated")

    today_str    = date.today().strftime("%Y%m%d")
    display_date = date.today().strftime("%B %d, %Y")
    inp          = cfg["inputs"]

    # ── Detect input files ────────────────────────────────────────────────────
    try:
        scan_path = find_latest_file(inp["scan_csv_folder"], inp.get("scan_csv_pattern", "scan_*.csv"))
        log.info("Scan CSV:        %s", scan_path.name)
    except FileNotFoundError as e:
        log.error("Missing input: %s", e)
        sys.exit(1)

    try:
        bridge_path = find_latest_file(inp["bridge_bot_log_folder"], inp.get("bridge_bot_log_pattern", "*_Bridge_Bot.log"))
        log.info("Bridge Bot log:  %s", bridge_path.name)
    except FileNotFoundError as e:
        log.error("Missing input: %s", e)
        sys.exit(1)

    try:
        regime_path = find_latest_file(inp["regime_bot_log_folder"], inp.get("regime_bot_log_pattern", "regime_*.log"))
        log.info("Regime Bot log:  %s", regime_path.name)
    except FileNotFoundError as e:
        log.error("Missing input: %s", e)
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    score_threshold = int(cfg["report"].get("score_threshold", 3))
    max_candidates  = int(cfg["report"].get("max_candidates", 10))

    candidates, n_cand, n_total = load_scan_csv(scan_path, score_threshold, max_candidates)
    log.info("Scan: %d rows total, %d candidates, showing top %d", n_total, n_cand, len(candidates))

    bridge_log = load_log_text(bridge_path)
    regime_log = load_log_text(regime_path)
    log.info("Logs loaded — Bridge Bot: %d chars, Regime Bot: %d chars", len(bridge_log), len(regime_log))

    regime = extract_regime(regime_log)
    log.info("Market regime detected: %s", regime)

    if not candidates:
        log.warning("No candidates passed filters — reports will show empty candidate table")

    if dry_run:
        log.info("DRY RUN complete — file detection and data loading successful, all inputs valid")
        return

    # ── OpenAI call ───────────────────────────────────────────────────────────
    api_key = cfg["openai"].get("api_key", "")
    if not api_key or api_key.startswith("YOUR_"):
        log.error("OpenAI API key not configured — set openai.api_key in config/ai_report_bot.yaml")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    prompt = build_prompt(candidates, regime_log, bridge_log, display_date)
    log.info("Calling OpenAI (%s)...", cfg["openai"]["model"])

    try:
        ai_response = call_openai(client, prompt, cfg)
        log.info("OpenAI response received successfully")
    except Exception as exc:
        log.error("OpenAI call failed: %s", exc)
        sys.exit(1)

    for lang in ("en", "fr"):
        if lang not in ai_response or not isinstance(ai_response[lang], dict):
            log.error("OpenAI response missing or invalid '%s' section — cannot generate PDFs", lang)
            sys.exit(1)

    # ── JSON archive ──────────────────────────────────────────────────────────
    archive_data = {
        "date":             today_str,
        "scan_file":        str(scan_path),
        "bridge_log_file":  str(bridge_path),
        "regime_log_file":  str(regime_path),
        "market_regime":    regime,
        "candidates_count": len(candidates),
        "ai_response":      ai_response,
    }
    archive_path = save_archive(archive_data, cfg["outputs"]["archive_folder"], today_str)
    log.info("JSON archive saved: %s", archive_path)

    # ── PDF generation ────────────────────────────────────────────────────────
    reports_dir = Path(cfg["outputs"]["reports_folder"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    for lang, title_key, suffix in (("en", "title_en", "EN"), ("fr", "title_fr", "FR")):
        title    = cfg["report"].get(title_key, "SB-ITM Morning Report")
        out_path = reports_dir / f"{today_str}_SB-ITM_Morning_Candidate_Review_{suffix}.pdf"
        log.info("Generating %s PDF...", suffix)
        generate_pdf(
            output_path=out_path,
            lang=lang,
            title=title,
            report_date=display_date,
            candidates=candidates,
            ai_content=ai_response[lang],
            regime=regime,
        )
        log.info("PDF saved: %s", out_path)

    log.info("AI Report Bot completed successfully")


if __name__ == "__main__":
    main()
