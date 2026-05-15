#!/usr/bin/env python3
"""SB-ITM AI Report Bot"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
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
    BaseDocTemplate, Frame, HRFlowable, KeepTogether,
    NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

# ── Brand ─────────────────────────────────────────────────────────────────────
DARK_BLUE  = colors.HexColor("#193F56")
ORANGE     = colors.HexColor("#F26022")
WHITE      = colors.white
LIGHT_GRAY = colors.HexColor("#F5F5F5")
BLUE_LIGHT = colors.HexColor("#EBF2F7")
MID_GRAY   = colors.HexColor("#CCCCCC")
TEXT_DARK  = colors.HexColor("#1A1A1A")
TEXT_MID   = colors.HexColor("#666666")
TEXT_LIGHT = colors.HexColor("#999999")

PW, PH = A4
LM = RM = 2.0 * cm
TM = BM = 2.0 * cm
CW = PW - LM - RM          # usable content width


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
def load_scan_csv(path: Path):
    all_count, candidates = 0, []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_count += 1
            if row.get("CandidateFlag") != "TRUE":
                continue
            candidates.append(row)
    candidates.sort(key=lambda r: _safe_score(r), reverse=True)
    return candidates, len(candidates), all_count


def _safe_score(r: dict) -> int:
    try:    return int(r.get("Score", 0))
    except: return 0


def load_log_text(path: Path, max_lines: int = 200) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])


def extract_regime_data(regime_log: str) -> dict:
    data = {"regime": "UNKNOWN", "spy": "N/A", "qqq": "N/A",
            "vix": "N/A", "breadth": "N/A"}
    for line in regime_log.splitlines():
        m = re.search(r"SPY [\d.]+ \| MA200: [\d.]+ \| dist: ([+\-\d.]+%)", line)
        if m:
            data["spy"] = f"Above MA200 ({m.group(1)})"
            continue
        m = re.search(r"QQQ [\d.]+ \| MA200.*?(above|below) MA", line)
        if m:
            data["qqq"] = f"{'Above' if m.group(1)=='above' else 'Below'} MA200"
            continue
        m = re.search(r"\bVIX ([\d.]+)\s*$", line.strip())
        if m:
            data["vix"] = m.group(1)
            continue
        m = re.search(r"breadth: (above|below) MA", line)
        if m:
            data["breadth"] = f"RSP {'above' if m.group(1)=='above' else 'below'} MA50"
            continue
        m = re.search(r"market_mode updated:.*?-> (\w+)", line)
        if m:
            data["regime"] = m.group(1).upper()
            continue
        m = re.search(r"Regime: (\w+)", line)
        if m and data["regime"] == "UNKNOWN":
            data["regime"] = m.group(1).upper()
    return data


def extract_bridge_symbols(bridge_log: str) -> list[str]:
    for line in reversed(bridge_log.splitlines()):
        m = re.search(r"symbols\s*=\s*\[([^\]]*)\]", line)
        if m:
            return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
    return []


def split_candidates(candidates: list[dict], selected_symbols: list[str], watchlist_min_score: int = 0):
    sel_set   = {s.upper() for s in selected_symbols}
    selected  = [r for r in candidates if r.get("Symbol", "").upper() in sel_set]
    watchlist = [r for r in candidates if r.get("Symbol", "").upper() not in sel_set
                 and _safe_score(r) >= watchlist_min_score]
    return selected, watchlist


# ── OpenAI ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a professional financial analyst writing institutional morning research reports for SB-ITM.

Each candidate row includes scanner flags: VolumeConfirmed, VolatilityFlag, ReboundPattern,
TwoStrongGreenCandle, LiquidityFlag, MA200Slope, WeeksAboveMA200, VeryCloseToSupport.
Ground every analysis in these actual flag values — do not invent data.

STRICT LANGUAGE RULES — apply to both English and French:
- NEVER use: "potential upside", "potential upward", "signals potential", "potential for upward moves",
  "upward corrections", "could rise", "may rise", "bullish potential", "upside potential", or any
  phrasing that implies a price prediction or directional recommendation.
- ALWAYS use neutral, rules-based wording such as:
  EN: "resistance distance remains available", "technical rebound pattern detected",
      "volume condition confirmed", "support proximity remains favorable under the scoring model",
      "market breadth remains constructive", "volatility moderate", "participation remains favorable"
  FR: "la distance de résistance reste disponible", "configuration de rebond technique détectée",
      "condition de volume confirmée", "la proximité du support reste favorable selon le modèle de scoring",
      "la participation de marché reste favorable", "volatilité modérée",
      "environnement de marché constructif", "configuration de rebond technique"
- French must be natural professional financial French — avoid literal machine translations.
  BAD: "environnement proactif", "corrections haussières potentielles", "largeur saine", "réflexion modérée"
  GOOD: "environnement de marché constructif", "configuration de rebond technique",
        "la participation de marché reste favorable", "volatilité modérée"

Generate a JSON with two top-level keys: "en" and "fr". Each contains:
  executive_summary    — 2-3 sentences summarising today's picture, referencing actual scan totals and selected symbols
  key_observations     — list of 4-6 concise bullet strings grounded in actual flag data
  regime_analysis      — 2-3 sentences interpreting regime indicators with exact SPY/QQQ/VIX values
  selected_candidates  — array, one per Bridge-Bot symbol:
    { symbol, profile (one of: "High Volatility Momentum" | "Momentum Pullback" |
      "Balanced Trend" | "Support Rebound" | "Extended Momentum"),
      structure_analysis (reference MA200Slope, WeeksAboveMA200, long-term trend),
      technical_position (reference exact SupportDist and ResDist percentages),
      momentum_volatility (reference RSI value and ATR% value explicitly),
      strengths (array of 3-4 short strings based on TRUE flags: VolumeConfirmed, ReboundPattern, etc.),
      points_of_attention (array of 1-3 short strings, always mention VolatilityFlag if High),
      neutral_reading (1-2 sentences, name the symbol, state its key characteristic) }
  strong_watchlist    — array of {symbol, company, observation} — near-support candidates not selected by Bridge Bot
  secondary_watchlist — array of {symbol, company, observation} — candidates with support too far or resistance too close
  scan_assessment     — 2 sentences: first on scan selectivity (X candidates from Y symbols), second on Bridge Bot coherence
  final_conclusion    — 2-3 sentences institutional wrap-up referencing regime and volatility theme

Tone: professional, neutral, factual, institutional. No trading advice. No markdown inside text. Valid JSON only.
"""


def _fmt_row(r: dict) -> str:
    def pct(v):
        try: return f"{float(v)*100:.2f}%"
        except: return str(v)
    return (
        f"  {r.get('Symbol','?')} | {r.get('CompanyName','')} | Score:{r.get('Score','?')} "
        f"| Close:{r.get('Close','?')} | SupportDist:{pct(r.get('SupportDistance',''))} "
        f"| ResDist:{pct(r.get('ResistanceDistance',''))} | RSI:{r.get('RSI14','?')} "
        f"| ATR%:{r.get('ATR_Pct','?')} | VolatilityFlag:{r.get('VolatilityFlag','?')} "
        f"| VolumeConfirmed:{r.get('VolumeConfirmed','?')} | ReboundPattern:{r.get('ReboundPattern','?')} "
        f"| TwoStrongGreenCandle:{r.get('TwoStrongGreenCandle','?')} | LiquidityFlag:{r.get('LiquidityFlag','?')} "
        f"| MA200Slope:{r.get('MA200Slope','?')} | WeeksAboveMA200:{r.get('WeeksAboveMA200','?')} "
        f"| VeryCloseToSupport:{r.get('VeryCloseToSupport','?')} | NearResistance:{r.get('NearResistance','?')}"
    )


def build_prompt(selected, watchlist, selected_symbols, regime_data, bridge_log, scan_date, n_total):
    sel_lines   = "\n".join(_fmt_row(r) for r in selected)   or "  None"
    watch_lines = "\n".join(_fmt_row(r) for r in watchlist)  or "  None"
    return (
        f"SB-ITM Morning Report — {scan_date} — {n_total} symbols scanned\n\n"
        f"REGIME: {regime_data['regime']} | SPY: {regime_data['spy']} | QQQ: {regime_data['qqq']} "
        f"| VIX: {regime_data['vix']} | Breadth: {regime_data['breadth']}\n\n"
        f"BRIDGE BOT SELECTED: {', '.join(selected_symbols) or 'None'}\n{sel_lines}\n\n"
        f"OTHER SCAN CANDIDATES:\n{watch_lines}\n\n"
        f"BRIDGE BOT LOG:\n{bridge_log.strip()}\n\n"
        "Generate the full institutional morning report JSON."
    )


def call_openai(client: OpenAI, prompt: str, cfg: dict) -> dict:
    oa = cfg["openai"]
    resp = client.chat.completions.create(
        model=str(oa["model"]),
        messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                  {"role": "user",   "content": prompt}],
        response_format={"type": "json_object"},
        timeout=int(oa.get("timeout", 90)),
        max_tokens=int(oa.get("max_tokens", 4000)),
    )
    return json.loads(resp.choices[0].message.content)


# ── PDF styles ────────────────────────────────────────────────────────────────
def _styles() -> dict:
    b = getSampleStyleSheet()
    def S(name, **kw):
        return ParagraphStyle(name, parent=b["Normal"], **kw)
    return {
        "cover_title":    S("ct",  fontName="Helvetica-Bold",  fontSize=26, textColor=DARK_BLUE, alignment=TA_CENTER, leading=32, spaceAfter=6),
        "cover_sub":      S("cs",  fontName="Helvetica-Bold",  fontSize=12, textColor=ORANGE,    alignment=TA_CENTER),
        "cover_label":    S("cl",  fontName="Helvetica-Bold",  fontSize=10, textColor=DARK_BLUE),
        "cover_val":      S("cv",  fontName="Helvetica",       fontSize=10, textColor=TEXT_DARK, alignment=TA_RIGHT),
        "cover_disc":     S("cd",  fontName="Helvetica",       fontSize=8,  textColor=TEXT_LIGHT, alignment=TA_CENTER),
        "section":        S("sec", fontName="Helvetica-Bold",  fontSize=14, textColor=ORANGE,    spaceBefore=4, spaceAfter=4),
        "subsection":     S("ss",  fontName="Helvetica-Bold",  fontSize=10, textColor=ORANGE,    spaceBefore=6, spaceAfter=2),
        "body":           S("bd",  fontName="Helvetica",       fontSize=10, textColor=TEXT_DARK, leading=16, spaceAfter=4),
        "body_sm":        S("bs",  fontName="Helvetica",       fontSize=9,  textColor=TEXT_DARK, leading=14),
        "th":             S("th",  fontName="Helvetica-Bold",  fontSize=8,  textColor=WHITE),
        "th_sm":          S("ths", fontName="Helvetica-Bold",  fontSize=7,  textColor=WHITE),
        "td":             S("td",  fontName="Helvetica",       fontSize=8,  textColor=TEXT_DARK, leading=11),
        "td_bold":        S("tdb", fontName="Helvetica-Bold",  fontSize=8,  textColor=TEXT_DARK),
        "td_lg":          S("tdl", fontName="Helvetica-Bold",  fontSize=11, textColor=TEXT_DARK, alignment=TA_CENTER, leading=14),
        "td_orange":      S("tdo", fontName="Helvetica-Bold",  fontSize=9,  textColor=ORANGE),
        "card_title":     S("crt", fontName="Helvetica-Bold",  fontSize=12, textColor=DARK_BLUE, leading=16),
        "footer":         S("ft",  fontName="Helvetica",       fontSize=7,  textColor=TEXT_LIGHT, alignment=TA_CENTER),
    }


# ── Page callbacks ────────────────────────────────────────────────────────────
def _on_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(DARK_BLUE)
    canvas.rect(0, PH - 0.5*cm, PW, 0.5*cm, fill=1, stroke=0)
    canvas.setFillColor(ORANGE)
    canvas.rect(0, 0, PW, 0.35*cm, fill=1, stroke=0)
    canvas.restoreState()


def _on_content(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(MID_GRAY)
    canvas.setLineWidth(0.5)
    canvas.line(LM, BM * 0.65, PW - RM, BM * 0.65)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(TEXT_LIGHT)
    ts = datetime.now().strftime("%Y-%m-%d")
    canvas.drawCentredString(PW / 2, BM * 0.38,
        f"SB-ITM AI Report Bot — {ts} — Confidential — Page {doc.page - 1}")
    canvas.restoreState()


# ── Cover page ────────────────────────────────────────────────────────────────
def _cover(title: str, report_date: str, selected_symbols: list[str],
           regime: str, labels: dict, st: dict) -> list:
    syms = ", ".join(selected_symbols) if selected_symbols else labels.get("no_symbols", "None")
    scope = "Morning Scan CSV · Bridge Bot LOG · Regime Bot LOG"

    info_rows = [
        [Paragraph(labels["exec_date"],    st["cover_label"]), Paragraph(report_date,            st["cover_val"])],
        [Paragraph(labels["sel_symbols"],  st["cover_label"]), Paragraph(syms,                   st["cover_val"])],
        [Paragraph(labels["det_regime"],   st["cover_label"]), Paragraph(regime.capitalize(),    st["cover_val"])],
        [Paragraph(labels["scope"],        st["cover_label"]), Paragraph(scope,                  st["cover_val"])],
    ]
    info_t = Table(info_rows, colWidths=[5*cm, CW - 5*cm])
    info_t.setStyle(TableStyle([
        ("BOX",           (0, 0), (-1, -1), 0.5, MID_GRAY),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.3, MID_GRAY),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))

    return [
        Spacer(1, 2.5*cm),
        Paragraph("SB-ITM", ParagraphStyle("ctl", fontName="Helvetica-Bold", fontSize=42,
                                            textColor=DARK_BLUE, alignment=TA_CENTER)),
        Paragraph(labels.get("logo_sub", ""), ParagraphStyle("ls", fontName="Helvetica", fontSize=9,
                                                               textColor=TEXT_LIGHT, alignment=TA_CENTER)),
        Spacer(1, 2.0*cm),
        Paragraph(title,                    st["cover_title"]),
        Paragraph(labels["subtitle"],       st["cover_sub"]),
        Spacer(1, 1.5*cm),
        info_t,
        Spacer(1, 2.5*cm),
        Paragraph(labels["disclaimer_short"], st["cover_disc"]),
        PageBreak(),
    ]


# ── Section header ────────────────────────────────────────────────────────────
def _sec(n: int, title: str, st: dict) -> list:
    return [
        Paragraph(f"{n}. {title}", st["section"]),
        HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=8),
    ]


# ── Callout box ───────────────────────────────────────────────────────────────
def _callout(text: str, st: dict) -> Table:
    t = Table([["", Paragraph(text, st["body"])]], colWidths=[0.22*cm, CW - 0.22*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), DARK_BLUE),
        ("BACKGROUND",    (1, 0), (1, -1), BLUE_LIGHT),
        ("LEFTPADDING",   (0, 0), (0, -1), 0),
        ("RIGHTPADDING",  (0, 0), (0, -1), 0),
        ("TOPPADDING",    (0, 0), (0, -1), 0),
        ("BOTTOMPADDING", (0, 0), (0, -1), 0),
        ("LEFTPADDING",   (1, 0), (1, -1), 12),
        ("RIGHTPADDING",  (1, 0), (1, -1), 12),
        ("TOPPADDING",    (1, 0), (1, -1), 10),
        ("BOTTOMPADDING", (1, 0), (1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return t


# ── Regime table ──────────────────────────────────────────────────────────────
def _regime_table(regime_data: dict, labels: dict, st: dict) -> Table:
    cw = CW / 5
    headers = [Paragraph(h, st["th_sm"]) for h in
               [labels["r_regime"], "SPY", "QQQ", "VIX", labels["r_breadth"]]]
    vals = [
        Paragraph(f"<b>{regime_data['regime'].capitalize()}</b>", st["td_lg"]),
        Paragraph(regime_data["spy"],     st["td_lg"]),
        Paragraph(regime_data["qqq"],     st["td_lg"]),
        Paragraph(regime_data["vix"],     st["td_lg"]),
        Paragraph(regime_data["breadth"], ParagraphStyle("rbb", fontName="Helvetica-Bold",
                  fontSize=10, textColor=TEXT_DARK, alignment=TA_CENTER, leading=13)),
    ]
    t = Table([headers, vals], colWidths=[cw]*5)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  DARK_BLUE),
        ("BOX",           (0, 0), (-1, -1), 0.5, MID_GRAY),
        ("LINEBEFORE",    (1, 0), (-1, -1), 0.3, MID_GRAY),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


# ── Candidate card ────────────────────────────────────────────────────────────
def _pct(v):
    try:    return f"{float(v)*100:.2f}%"
    except: return str(v)


def _candidate_card(csv_row: dict, ai: dict, st: dict, labels: dict) -> list:
    sym     = csv_row.get("Symbol", "?")
    company = csv_row.get("CompanyName", "")
    profile = ai.get("profile", "")
    score   = csv_row.get("Score", "")
    rsi     = csv_row.get("RSI14", "")
    atr     = csv_row.get("ATR_Pct", "")
    sup     = csv_row.get("SupportDistance", "")
    res     = csv_row.get("ResistanceDistance", "")

    def safe_num(v, decimals=2):
        try:    return f"{round(float(v), decimals)}"
        except: return str(v)

    # Header — inline: "SYMBOL — COMPANY | Profile" with profile in orange
    hdr_text = (
        f'<b><font color="#193F56">{sym} — {company.upper()}</font></b>'
        f' | <b><font color="#F26022">{profile}</font></b>'
    )
    hdr = Table(
        [[Paragraph(hdr_text, st["card_title"])]],
        colWidths=[CW],
    )
    hdr.setStyle(TableStyle([
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    # Orange rule under header
    rule = Table([[""]],  colWidths=[CW])
    rule.setStyle(TableStyle([
        ("LINEBELOW",     (0, 0), (-1, -1), 1.5, ORANGE),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    # Metric table
    m_headers = [labels["m_score"], labels["m_profile"], "RSI", "ATR %", labels["m_sup"], labels["m_res"]]
    m_vals    = [str(score), profile, safe_num(rsi), f"{safe_num(atr)}%", _pct(sup), _pct(res)]
    mcw = CW / 6
    metric_t = Table(
        [[Paragraph(h, st["th_sm"]) for h in m_headers],
         [Paragraph(v, st["td_lg"]) for v in m_vals]],
        colWidths=[mcw]*6,
    )
    metric_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  DARK_BLUE),
        ("LINEBEFORE",    (1, 0), (-1, -1), 0.3, MID_GRAY),
        ("BOX",           (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
    ]))

    def _sub(label_key: str, text: str) -> list:
        if not text.strip():
            return []
        return [
            Table([[Paragraph(labels[label_key], st["subsection"])]],
                  colWidths=[CW],
                  style=TableStyle([("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),
                                    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),2)])),
            Table([[Paragraph(text, st["body_sm"])]],
                  colWidths=[CW],
                  style=TableStyle([("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),
                                    ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),6)])),
        ]

    # Strengths / Points two-column
    s_items = ai.get("strengths", [])
    p_items = ai.get("points_of_attention", [])
    s_text  = "<br/>".join(f"• {x}" for x in s_items) or "N/A"
    p_text  = "<br/>".join(f"• {x}" for x in p_items) or "N/A"
    two_col = Table(
        [[Paragraph(labels["strengths"],  st["subsection"]), Paragraph(labels["points_attn"], st["subsection"])],
         [Paragraph(s_text, st["body_sm"]),                 Paragraph(p_text, st["body_sm"])]],
        colWidths=[CW/2, CW/2],
    )
    two_col.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LINEBEFORE",    (1, 0), (1, -1),  0.3, MID_GRAY),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    # Assemble rows inside outer bordered box
    inner_rows = (
        [[hdr], [rule], [metric_t]]
        + [[t] for t in _sub("struct_analysis", ai.get("structure_analysis", ""))]
        + [[t] for t in _sub("tech_position",   ai.get("technical_position", ""))]
        + [[t] for t in _sub("momentum_vol",    ai.get("momentum_volatility", ""))]
        + [[two_col]]
        + [[t] for t in _sub("neutral_reading", ai.get("neutral_reading", ""))]
    )
    outer = Table(inner_rows, colWidths=[CW])
    outer.setStyle(TableStyle([
        ("BOX",           (0, 0), (-1, -1), 0.8, MID_GRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))
    return [KeepTogether([outer]), Spacer(1, 0.5*cm)]


# ── Watchlist table ───────────────────────────────────────────────────────────
def _watchlist_table(rows: list[dict], csv_lookup: dict, st: dict, labels: dict) -> Table:
    headers = [Paragraph(h, st["th"]) for h in
               [labels["w_symbol"], labels["w_company"], labels["w_score"],
                "RSI", "ATR %", labels["w_sup"], labels["w_res"], labels["w_obs"]]]
    data = [headers]
    for r in rows:
        sym = r.get("symbol", r.get("Symbol", ""))
        obs = r.get("observation", r.get("Observation", ""))
        csv = csv_lookup.get(sym.upper(), {})
        data.append([
            Paragraph(sym,                          st["td_bold"]),
            Paragraph(r.get("company", csv.get("CompanyName", "")), st["td"]),
            Paragraph(str(csv.get("Score", "")),    st["td"]),
            Paragraph(str(csv.get("RSI14", "")),    st["td"]),
            Paragraph(f"{csv.get('ATR_Pct','')}%",  st["td"]),
            Paragraph(_pct(csv.get("SupportDistance","")),    st["td"]),
            Paragraph(_pct(csv.get("ResistanceDistance","")), st["td"]),
            Paragraph(obs,                          st["td"]),
        ])
    cws = [1.7*cm, 2.7*cm, 1.2*cm, 1.2*cm, 1.2*cm, 1.5*cm, 1.9*cm, CW-11.4*cm]
    t = Table(data, colWidths=cws, repeatRows=1)
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  DARK_BLUE),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))
    t.setStyle(TableStyle(style))
    return t


# ── Scan comment (programmatic from CSV flags) ────────────────────────────────
def _scan_comment(r: dict) -> str:
    parts = []
    try:
        sup = float(r.get("SupportDistance") or 1)
        if sup < 0.02:
            parts.append("Very close to support")
        elif sup < 0.06:
            parts.append("Near support")
        else:
            parts.append(f"Support too far ({sup*100:.1f}%)")
    except (ValueError, TypeError):
        pass
    if str(r.get("ReboundPattern", "")).upper() in ("TRUE", "YES", "1"):
        parts.append("Rebound pattern")
    if str(r.get("TwoStrongGreenCandle", "")).upper() in ("TRUE", "YES", "1"):
        parts.append("Two-green-candle")
    if str(r.get("VolumeConfirmed", "")).upper() in ("TRUE", "YES", "1"):
        parts.append("Volume confirmed")
    if str(r.get("LiquidityFlag", "")).upper() in ("TRUE", "YES", "1"):
        parts.append("Liquidity positive")
    try:
        res = float(r.get("ResistanceDistance") or 1)
        if res < 0.04:
            parts.append(f"Resistance too close ({res*100:.1f}%)")
    except (ValueError, TypeError):
        pass
    if str(r.get("VolatilityFlag", "")).upper() in ("HIGH", "TRUE", "YES", "1"):
        parts.append("High volatility")
    return " / ".join(parts)


# ── Comparative table ─────────────────────────────────────────────────────────
def _comparative_table(all_rows: list[dict], selected_symbols: list[str], st: dict, labels: dict) -> Table:
    sel_set = {s.upper() for s in selected_symbols}
    headers = [Paragraph(h, st["th"]) for h in
               ["Symbol", labels["role"], labels["w_score"],
                "RSI", "ATR %", labels["w_sup"], labels["w_res"], labels["scan_comment"]]]
    data = [headers]
    for r in all_rows:
        sym  = r.get("Symbol", "")
        role = labels["role_sel"] if sym.upper() in sel_set else labels["role_watch"]
        data.append([
            Paragraph(sym,                                          st["td_bold"]),
            Paragraph(role,                                         st["td_orange"] if sym.upper() in sel_set else st["td"]),
            Paragraph(str(r.get("Score", "")),                      st["td"]),
            Paragraph(str(r.get("RSI14", "")),                      st["td"]),
            Paragraph(f"{r.get('ATR_Pct', '')}%",                   st["td"]),
            Paragraph(_pct(r.get("SupportDistance","")),            st["td"]),
            Paragraph(_pct(r.get("ResistanceDistance","")),         st["td"]),
            Paragraph(_scan_comment(r),                             st["td"]),
        ])
    cws = [1.5*cm, 1.8*cm, 1.3*cm, 1.1*cm, 1.1*cm, 1.7*cm, 1.9*cm, CW-10.4*cm]
    t = Table(data, colWidths=cws, repeatRows=1)
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  DARK_BLUE),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))
    t.setStyle(TableStyle(style))
    return t


# ── Disclaimer box ────────────────────────────────────────────────────────────
def _disclaimer_box(text: str, st: dict) -> Table:
    t = Table([[Paragraph(text, st["body_sm"])]], colWidths=[CW])
    t.setStyle(TableStyle([
        ("BOX",           (0, 0), (-1, -1), 0.5, MID_GRAY),
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GRAY),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


# ── Labels ────────────────────────────────────────────────────────────────────
_LABELS = {
    "en": {
        "subtitle":       "Rules-Based Candidate Intelligence Report",
        "exec_date":      "Execution Date",
        "sel_symbols":    "Selected Symbols",
        "det_regime":     "Detected Regime",
        "scope":          "Scope",
        "logo_sub":       "",
        "no_symbols":     "None",
        "disclaimer_short": "Neutral technical research. Not investment advice.",
        "s1_exec":        "Executive Summary",
        "s2_obs":         "Key Daily Observations",
        "s3_regime":      "Regime Analysis",
        "s4_sel":         "Bridge Bot Selected Candidates",
        "s5_add":         "Additional Notable Candidates",
        "s5a_strong":     "A. Strong Watchlist Candidates",
        "s5b_second":     "B. Secondary Watchlist Candidates",
        "s6_scan":        "Daily Scan Assessment",
        "s7_comp":        "Comparative Summary Table",
        "s8_concl":       "Final Neutral Conclusion",
        "scan_comment":   "Scan Comment",
        "r_regime":       "REGIME", "r_breadth": "BREADTH",
        "m_score":        "SB-ITM SCORE", "m_profile": "PROFILE",
        "m_sup":          "SUPPORT DIST.", "m_res": "RESISTANCE DIST.",
        "struct_analysis":"Structure Analysis",
        "tech_position":  "Technical Position",
        "momentum_vol":   "Momentum / Volatility",
        "strengths":      "Strengths",
        "points_attn":    "Points of Attention",
        "neutral_reading":"Neutral SB-ITM Reading",
        "w_symbol":       "Symbol", "w_company": "Company",
        "w_score":        "Score",  "w_sup": "Support", "w_res": "Resistance",
        "w_obs":          "Observation",
        "role":           "Role", "role_sel": "Selected", "role_watch": "Watchlist",
        "no_cand":        "No candidates were selected by Bridge Bot today.",
        "no_watch":       "No additional candidates in this category.",
        "disclaimer_long": (
            "This report is rules-based technical research only. It does not recommend "
            "buying, selling, holding, or entering any financial instrument. It does not "
            "predict future market direction."
        ),
    },
    "fr": {
        "subtitle":       "Rapport d'Intelligence Candidats Basé sur des Règles",
        "exec_date":      "Date d'Exécution",
        "sel_symbols":    "Symboles Sélectionnés",
        "det_regime":     "Régime Détecté",
        "scope":          "Périmètre",
        "logo_sub":       "",
        "no_symbols":     "Aucun",
        "disclaimer_short": "Recherche technique neutre. Pas un conseil en investissement.",
        "s1_exec":        "Résumé Exécutif",
        "s2_obs":         "Observations Clés du Jour",
        "s3_regime":      "Analyse du Régime",
        "s4_sel":         "Candidats Sélectionnés par Bridge Bot",
        "s5_add":         "Candidats Notables Supplémentaires",
        "s5a_strong":     "A. Candidats Watchlist Forts",
        "s5b_second":     "B. Candidats Watchlist Secondaires",
        "s6_scan":        "Évaluation du Scan Quotidien",
        "s7_comp":        "Tableau Comparatif",
        "s8_concl":       "Conclusion Neutre Finale",
        "scan_comment":   "Commentaire Scan",
        "r_regime":       "RÉGIME", "r_breadth": "LARGEUR",
        "m_score":        "SCORE SB-ITM", "m_profile": "PROFIL",
        "m_sup":          "DIST. SUPPORT", "m_res": "DIST. RÉSISTANCE",
        "struct_analysis":"Analyse Structurelle",
        "tech_position":  "Position Technique",
        "momentum_vol":   "Momentum / Volatilité",
        "strengths":      "Points Forts",
        "points_attn":    "Points d'Attention",
        "neutral_reading":"Lecture Neutre SB-ITM",
        "w_symbol":       "Symbole", "w_company": "Société",
        "w_score":        "Score",   "w_sup": "Support", "w_res": "Résistance",
        "w_obs":          "Observation",
        "role":           "Rôle", "role_sel": "Sélectionné", "role_watch": "Watchlist",
        "no_cand":        "Aucun candidat n'a été sélectionné par Bridge Bot aujourd'hui.",
        "no_watch":       "Aucun candidat supplémentaire dans cette catégorie.",
        "disclaimer_long": (
            "Ce rapport est une recherche technique basée sur des règles uniquement. Il ne recommande pas "
            "d'acheter, de vendre, de conserver ou d'entrer dans un instrument financier. "
            "Il ne prédit pas l'évolution future des marchés."
        ),
    },
}


# ── PDF generation ────────────────────────────────────────────────────────────
def generate_pdf(output_path: Path, lang: str, title: str, report_date: str,
                 all_candidates: list[dict], selected_symbols: list[str],
                 ai_content: dict, regime_data: dict):
    st     = _styles()
    labels = _LABELS.get(lang, _LABELS["en"])
    csv_lu = {r.get("Symbol", "").upper(): r for r in all_candidates}
    story  = []

    # Cover
    story += _cover(title, report_date, selected_symbols,
                    regime_data.get("regime", "UNKNOWN"), labels, st)
    story.append(NextPageTemplate("content"))

    n = 1
    # 1. Executive Summary
    story += _sec(n, labels["s1_exec"], st); n += 1
    exec_sum = ai_content.get("executive_summary", "")
    if exec_sum:
        story.append(_callout(exec_sum, st))
    story.append(Spacer(1, 0.4*cm))

    # 2. Key Daily Observations
    story += _sec(n, labels["s2_obs"], st); n += 1
    obs_items = ai_content.get("key_observations", [])
    if obs_items:
        obs_text = "<br/>".join(f"• {o}" for o in obs_items)
        story.append(_callout(obs_text, st))
    story.append(Spacer(1, 0.4*cm))

    # 3. Regime Analysis
    story += _sec(n, labels["s3_regime"], st); n += 1
    story.append(_regime_table(regime_data, labels, st))
    story.append(Spacer(1, 0.3*cm))
    regime_text = ai_content.get("regime_analysis", "")
    if regime_text:
        story.append(Paragraph(regime_text, st["body"]))
    story.append(Spacer(1, 0.4*cm))

    # 4. Bridge Bot Selected Candidates
    story.append(PageBreak())
    story += _sec(n, labels["s4_sel"], st); n += 1
    ai_selected = ai_content.get("selected_candidates", [])
    if ai_selected:
        ai_lu = {d.get("symbol", "").upper(): d for d in ai_selected}
        for i, sym in enumerate(selected_symbols):
            if i > 0:
                story.append(PageBreak())
            csv_row = csv_lu.get(sym.upper(), {"Symbol": sym})
            ai_data = ai_lu.get(sym.upper(), {"symbol": sym})
            story += _candidate_card(csv_row, ai_data, st, labels)
    else:
        story.append(Paragraph(labels["no_cand"], st["body"]))
    story.append(Spacer(1, 0.2*cm))

    # 5. Additional Notable Candidates
    strong_watch = ai_content.get("strong_watchlist", [])
    second_watch = ai_content.get("secondary_watchlist", [])
    if strong_watch or second_watch:
        story.append(PageBreak())
        story += _sec(n, labels["s5_add"], st); n += 1
        if strong_watch:
            story.append(Paragraph(labels["s5a_strong"], st["subsection"]))
            story.append(HRFlowable(width="100%", thickness=0.5, color=ORANGE, spaceAfter=6))
            story.append(_watchlist_table(strong_watch, csv_lu, st, labels))
            story.append(Spacer(1, 0.4*cm))
        if second_watch:
            story.append(Paragraph(labels["s5b_second"], st["subsection"]))
            story.append(HRFlowable(width="100%", thickness=0.5, color=ORANGE, spaceAfter=6))
            story.append(_watchlist_table(second_watch, csv_lu, st, labels))
        story.append(Spacer(1, 0.4*cm))

    # 6. Daily Scan Assessment
    scan_assess = ai_content.get("scan_assessment", "")
    if scan_assess:
        story += _sec(n, labels["s6_scan"], st); n += 1
        story.append(Paragraph(scan_assess, st["body"]))
        story.append(Spacer(1, 0.4*cm))

    # 7. Comparative Summary
    if all_candidates:
        comp_table = _comparative_table(all_candidates, selected_symbols, st, labels)
        story.append(KeepTogether(_sec(n, labels["s7_comp"], st) + [comp_table])); n += 1
        story.append(Spacer(1, 0.4*cm))

    # 8. Final Conclusion
    story.append(PageBreak())
    story += _sec(n, labels["s8_concl"], st); n += 1
    concl = ai_content.get("final_conclusion", "")
    if concl:
        story.append(Paragraph(concl, st["body"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(_disclaimer_box(labels["disclaimer_long"], st))

    # Build doc with two page templates
    cover_frame   = Frame(LM, BM, CW, PH - TM - BM, id="cover")
    content_frame = Frame(LM, BM + 0.5*cm, CW, PH - TM - BM - 1.0*cm, id="content")
    cover_tpl   = PageTemplate(id="cover",   frames=[cover_frame],   onPage=_on_cover)
    content_tpl = PageTemplate(id="content", frames=[content_frame], onPage=_on_content)

    doc = BaseDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM,
    )
    doc.addPageTemplates([cover_tpl, content_tpl])
    doc.build(story)


# ── JSON archive ──────────────────────────────────────────────────────────────
def save_archive(data: dict, folder: str, date_str: str) -> Path:
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    out = p / f"{date_str}_AI_Report_Bot.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out


# ── Mock response ─────────────────────────────────────────────────────────────
_MOCK_CANDIDATES = [
    {"Symbol": "AAPL",  "CompanyName": "Apple Inc.",           "Score": "7",  "CandidateFlag": "TRUE",
     "Profile": "Momentum Near Support",    "RSI14": "52.3",  "ATR_Pct": "0.0124",
     "SupportDistance": "0.0185", "ResistanceDistance": "0.0420",
     "VolumeConfirmed": "TRUE",  "VolatilityFlag": "High",  "ReboundPattern": "FALSE",
     "TwoStrongGreenCandle": "TRUE",  "LiquidityFlag": "TRUE",
     "MA200Slope": "Up", "WeeksAboveMA200": "34", "VeryCloseToSupport": "FALSE", "NearResistance": "FALSE"},
    {"Symbol": "NVDA",  "CompanyName": "NVIDIA Corporation",   "Score": "6",  "CandidateFlag": "TRUE",
     "Profile": "High Volatility Momentum", "RSI14": "58.7",  "ATR_Pct": "0.0231",
     "SupportDistance": "0.0210", "ResistanceDistance": "0.0510",
     "VolumeConfirmed": "TRUE",  "VolatilityFlag": "High",  "ReboundPattern": "FALSE",
     "TwoStrongGreenCandle": "TRUE",  "LiquidityFlag": "TRUE",
     "MA200Slope": "Up", "WeeksAboveMA200": "28", "VeryCloseToSupport": "FALSE", "NearResistance": "FALSE"},
    {"Symbol": "MSFT",  "CompanyName": "Microsoft Corporation","Score": "5",  "CandidateFlag": "TRUE",
     "Profile": "Balanced Trend",           "RSI14": "49.1",  "ATR_Pct": "0.0098",
     "SupportDistance": "0.0095", "ResistanceDistance": "0.0380",
     "VolumeConfirmed": "TRUE",  "VolatilityFlag": "High",  "ReboundPattern": "TRUE",
     "TwoStrongGreenCandle": "FALSE", "LiquidityFlag": "TRUE",
     "MA200Slope": "Up", "WeeksAboveMA200": "52", "VeryCloseToSupport": "TRUE",  "NearResistance": "FALSE"},
    {"Symbol": "META",  "CompanyName": "Meta Platforms Inc.",  "Score": "5",  "CandidateFlag": "TRUE",
     "Profile": "Momentum Pullback",        "RSI14": "55.4",  "ATR_Pct": "0.0177",
     "SupportDistance": "0.0260", "ResistanceDistance": "0.0490",
     "VolumeConfirmed": "TRUE",  "VolatilityFlag": "High",  "ReboundPattern": "FALSE",
     "TwoStrongGreenCandle": "FALSE", "LiquidityFlag": "TRUE",
     "MA200Slope": "Up", "WeeksAboveMA200": "18", "VeryCloseToSupport": "FALSE", "NearResistance": "FALSE"},
    {"Symbol": "AMZN",  "CompanyName": "Amazon.com Inc.",      "Score": "4",  "CandidateFlag": "TRUE",
     "Profile": "Support Rebound",          "RSI14": "47.6",  "ATR_Pct": "0.0142",
     "SupportDistance": "0.0330", "ResistanceDistance": "0.0610",
     "VolumeConfirmed": "FALSE", "VolatilityFlag": "High",  "ReboundPattern": "FALSE",
     "TwoStrongGreenCandle": "FALSE", "LiquidityFlag": "TRUE",
     "MA200Slope": "Flat", "WeeksAboveMA200": "8",  "VeryCloseToSupport": "FALSE", "NearResistance": "FALSE"},
    {"Symbol": "GOOGL", "CompanyName": "Alphabet Inc.",        "Score": "3",  "CandidateFlag": "TRUE",
     "Profile": "Extended Momentum",        "RSI14": "44.2",  "ATR_Pct": "0.0115",
     "SupportDistance": "0.1410", "ResistanceDistance": "0.0090",
     "VolumeConfirmed": "FALSE", "VolatilityFlag": "Low",   "ReboundPattern": "FALSE",
     "TwoStrongGreenCandle": "FALSE", "LiquidityFlag": "TRUE",
     "MA200Slope": "Up", "WeeksAboveMA200": "41", "VeryCloseToSupport": "FALSE", "NearResistance": "TRUE"},
]
_MOCK_SELECTED_SYMS = ["AAPL", "NVDA"]

_MOCK = {
    "en": {
        "executive_summary": (
            "The 12 May 2026 morning workflow identifies a Bull regime with SPY and QQQ trading "
            "comfortably above MA200. Bridge Bot selected 2 candidates — AAPL and NVDA — both "
            "presenting near-support setups with confirmed momentum alignment in the current environment."
        ),
        "key_observations": [
            "Regime Bot classifies the environment as Bull; SPY and QQQ above MA200.",
            "VIX at 17.19 and SPY ATR% at 1.24% confirm a stable, moderate-volatility environment.",
            "Bridge Bot selected AAPL and NVDA as primary candidates for tomorrow's session.",
            "MSFT and META form a strong watchlist with solid score and near-support positioning.",
            "AMZN and GOOGL are secondary watchlist candidates with wider support distances.",
        ],
        "regime_analysis": (
            "The regime evidence is internally consistent. SPY closed well above its MA200 with a "
            "distance of +9.53%, and QQQ confirms the broad uptrend. VIX at 17.19 remains below "
            "the defensive threshold, and RSP breadth above MA50 supports the bull classification."
        ),
        "selected_candidates": [
            {
                "symbol": "AAPL",
                "profile": "Momentum Near Support",
                "structure_analysis": "AAPL is consolidating just above a well-defined weekly support zone. The price action shows a series of higher lows, maintaining structure integrity within the bull trend.",
                "technical_position": "Price is 1.85% above primary support. RSI at 52.3 is neutral and non-extended, leaving room for continuation. The MA50 is pointing upward and acting as a dynamic support layer.",
                "momentum_volatility": "ATR% of 1.24% reflects moderate intraday movement. Momentum oscillators are in neutral territory — no divergence detected. Volume profile supports accumulation behavior near current levels.",
                "strengths": ["Near key weekly support with structure intact", "RSI neutral — not overbought", "MA50 trending upward as dynamic support"],
                "points_of_attention": ["Broad market gap-down would pressure the setup", "ATR is moderate — intraday swings possible"],
                "neutral_reading": "AAPL presents a technically sound near-support setup within a bull regime. The position respects all SB-ITM structural filters. No directional recommendation is made — this is a rules-based observation only."
            },
            {
                "symbol": "NVDA",
                "profile": "High Momentum Breakout Watch",
                "structure_analysis": "NVDA has been building a base above its 50-day MA following a recent pullback. The current consolidation shows decreasing volume on down days, a pattern consistent with accumulation ahead of a potential continuation move.",
                "technical_position": "Price is 2.10% above primary support. RSI at 58.7 indicates above-average momentum without being in overbought territory. The stock is respecting its rising channel.",
                "momentum_volatility": "ATR% of 2.31% reflects NVDA's typically higher volatility profile. This is within expected range and does not represent an abnormal expansion. Momentum remains positive on the weekly timeframe.",
                "strengths": ["Constructive base above MA50", "Decreasing volume on pullback days — accumulation signal", "Strong sector momentum (semiconductors in bull mode)"],
                "points_of_attention": ["Higher ATR% requires wider risk management", "Any negative semiconductor news would have outsized impact"],
                "neutral_reading": "NVDA meets the SB-ITM scanner criteria with a higher-volatility profile. The setup is technically valid but demands appropriate position sizing given the ATR. This is a neutral observation — not a trading recommendation."
            },
        ],
        "strong_watchlist": [
            {"symbol": "MSFT", "observation": "Near support, low ATR — conservative profile. Score 5, RSI neutral."},
            {"symbol": "META", "observation": "Momentum building, support 2.60% below. Higher ATR warrants monitoring."},
        ],
        "secondary_watchlist": [
            {"symbol": "AMZN", "observation": "Score 4, support distance 3.30% — outside primary filter but worth monitoring."},
            {"symbol": "GOOGL", "observation": "Score 3, RSI 44.2 approaching oversold. Wide resistance distance limits upside clarity."},
        ],
        "scan_assessment": (
            "Today's scan produced 6 qualifying candidates across all score tiers. The quality distribution "
            "reflects a healthy but selective environment — only the top 2 met the Bridge Bot threshold. "
            "The remaining 4 form a valid watchlist for upcoming sessions."
        ),
        "final_conclusion": (
            "The neutral SB-ITM conclusion is constructive. Two candidates meet full positioning criteria "
            "within a confirmed bull regime. The watchlist provides 4 additional setups for monitoring. "
            "All observations are rules-based and carry no directional trading recommendation."
        ),
    },
    "fr": {
        "executive_summary": (
            "Le workflow matinal du 12 mai 2026 identifie un régime Haussier avec SPY et QQQ "
            "évoluant confortablement au-dessus de leur MA200. Bridge Bot a sélectionné 2 candidats "
            "— AAPL et NVDA — présentant tous deux des configurations proches du support avec alignement de momentum confirmé."
        ),
        "key_observations": [
            "Le Regime Bot classe l'environnement comme Haussier ; SPY et QQQ au-dessus de la MA200.",
            "VIX à 17,19 et ATR% SPY à 1,24% confirment un environnement stable avec volatilité modérée.",
            "Bridge Bot a sélectionné AAPL et NVDA comme candidats principaux pour la session de demain.",
            "MSFT et META forment une watchlist forte avec un bon score et un positionnement proche du support.",
            "AMZN et GOOGL sont des candidats watchlist secondaires avec des distances de support plus larges.",
        ],
        "regime_analysis": (
            "Les données de régime sont cohérentes. SPY clôture bien au-dessus de sa MA200 avec "
            "un écart de +9,53%, et QQQ confirme la tendance haussière large. Le VIX à 17,19 "
            "reste sous le seuil défensif, et la participation RSP au-dessus de la MM50 soutient la classification haussière."
        ),
        "selected_candidates": [
            {
                "symbol": "AAPL",
                "profile": "Momentum Proche du Support",
                "structure_analysis": "AAPL se consolide juste au-dessus d'une zone de support hebdomadaire bien définie. L'action des prix montre une série de plus bas croissants, maintenant l'intégrité de la structure dans la tendance haussière.",
                "technical_position": "Le prix est à 1,85% au-dessus du support principal. Le RSI à 52,3 est neutre et non-étendu, laissant de la place pour une continuation. La MA50 pointe vers le haut et agit comme un niveau de support dynamique.",
                "momentum_volatility": "L'ATR% de 1,24% reflète un mouvement intrajournalier modéré. Les oscillateurs de momentum sont en territoire neutre — aucune divergence détectée. Le profil de volume soutient un comportement d'accumulation aux niveaux actuels.",
                "strengths": ["Proche du support hebdomadaire clé avec structure intacte", "RSI neutre — non suracheté", "MA50 en tendance haussière comme support dynamique"],
                "points_of_attention": ["Un gap baissier du marché large exercerait une pression sur la configuration", "ATR modéré — fluctuations intrajournalières possibles"],
                "neutral_reading": "AAPL présente une configuration techniquement solide proche du support dans un régime haussier. La position respecte tous les filtres structurels SB-ITM. Aucune recommandation directionnelle — observation basée sur les règles uniquement."
            },
            {
                "symbol": "NVDA",
                "profile": "Surveillance Cassure Momentum Élevé",
                "structure_analysis": "NVDA consolide au-dessus de sa MA50 suite à un repli récent. La consolidation montre un volume décroissant les jours de baisse, un schéma cohérent avec une accumulation avant un mouvement de continuation potentiel.",
                "technical_position": "Le prix est à 2,10% au-dessus du support principal. Le RSI à 58,7 indique un momentum supérieur à la moyenne sans être en territoire de surachat. L'action respecte son canal haussier.",
                "momentum_volatility": "L'ATR% de 2,31% reflète le profil de volatilité typiquement plus élevé de NVDA. Ceci est dans la plage attendue et ne représente pas une expansion anormale. Le momentum reste positif sur le timeframe hebdomadaire.",
                "strengths": ["Base constructive au-dessus de la MA50", "Volume décroissant les jours de repli — signal d'accumulation", "Momentum sectoriel fort (semi-conducteurs en mode haussier)"],
                "points_of_attention": ["ATR% élevé nécessite une gestion du risque plus large", "Toute nouvelle négative sur les semi-conducteurs aurait un impact amplifié"],
                "neutral_reading": "NVDA répond aux critères du scanner SB-ITM avec un profil de volatilité plus élevé. La configuration est techniquement valide mais exige un dimensionnement de position approprié compte tenu de l'ATR. Observation neutre — pas une recommandation de trading."
            },
        ],
        "strong_watchlist": [
            {"symbol": "MSFT", "observation": "Proche du support, ATR faible — profil conservateur. Score 5, RSI neutre."},
            {"symbol": "META", "observation": "Momentum en construction, support à 2,60% en dessous. ATR élevé à surveiller."},
        ],
        "secondary_watchlist": [
            {"symbol": "AMZN", "observation": "Score 4, distance support 3,30% — hors filtre principal mais à surveiller."},
            {"symbol": "GOOGL", "observation": "Score 3, RSI 44,2 approchant la survente. Distance de résistance large limite la clarté haussière."},
        ],
        "scan_assessment": (
            "Le scan du jour a produit 6 candidats qualifiés sur tous les niveaux de score. La distribution "
            "de qualité reflète un environnement sain mais sélectif — seulement les 2 premiers ont satisfait "
            "le seuil Bridge Bot. Les 4 restants forment une watchlist valide pour les prochaines sessions."
        ),
        "final_conclusion": (
            "La conclusion neutre SB-ITM est constructive. Deux candidats répondent aux critères de "
            "positionnement complets dans un régime haussier confirmé. La watchlist offre 4 configurations "
            "supplémentaires à surveiller. Toutes les observations sont basées sur des règles et ne comportent "
            "aucune recommandation de trading directionnelle."
        ),
    },
}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SB-ITM AI Report Bot")
    parser.add_argument("--config",  default="config/ai_report_bot.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock",    action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging(cfg["outputs"]["log_folder"])
    log.info("AI Report Bot started")

    dry_run = args.dry_run or cfg.get("dry_run", False)
    mock    = args.mock
    if dry_run: log.info("DRY RUN mode")
    if mock:    log.info("MOCK mode — PDFs generated with placeholder content")

    today_str    = date.today().strftime("%Y%m%d")
    display_date = date.today().strftime("%Y-%m-%d")
    inp          = cfg["inputs"]

    # Detect files
    for label, folder_key, pattern_key, default_pat in [
        ("Scan CSV",        "scan_csv_folder",        "scan_csv_pattern",        "scan_*.csv"),
        ("Bridge Bot log",  "bridge_bot_log_folder",  "bridge_bot_log_pattern",  "*_Bridge_Bot.log"),
        ("Regime Bot log",  "regime_bot_log_folder",  "regime_bot_log_pattern",  "regime_*.log"),
    ]:
        try:
            f = find_latest_file(inp[folder_key], inp.get(pattern_key, default_pat))
            log.info("%s: %s", label, f.name)
            if label == "Scan CSV":        scan_path   = f
            elif label == "Bridge Bot log": bridge_path = f
            else:                           regime_path = f
        except FileNotFoundError as e:
            log.error("Missing input: %s", e); sys.exit(1)

    # Load data
    watchlist_min_score = int(cfg["report"].get("watchlist_min_score", 3))
    candidates, n_cand, n_total = load_scan_csv(scan_path)
    log.info("Scan: %d total, %d flagged candidates", n_total, n_cand)

    bridge_log = load_log_text(bridge_path)
    regime_log = load_log_text(regime_path)
    regime_data    = extract_regime_data(regime_log)
    selected_syms  = extract_bridge_symbols(bridge_log)
    selected, watchlist = split_candidates(candidates, selected_syms, watchlist_min_score)
    log.info("Regime: %s | Bridge selected: %s", regime_data["regime"], selected_syms or "None")

    if dry_run:
        log.info("DRY RUN complete — all inputs valid"); return

    # OpenAI or mock
    if mock:
        candidates     = _MOCK_CANDIDATES
        selected_syms  = _MOCK_SELECTED_SYMS
        selected, watchlist = split_candidates(candidates, selected_syms, watchlist_min_score)
        ai_response = _MOCK
        log.info("Using mock AI response")
    else:
        api_key = os.environ.get("OPENAI_API_KEY") or cfg["openai"].get("api_key", "")
        if not api_key or api_key.startswith("YOUR_"):
            log.error("OpenAI API key not configured"); sys.exit(1)
        client = OpenAI(api_key=api_key)
        prompt = build_prompt(selected, watchlist, selected_syms,
                              regime_data, bridge_log, display_date, n_total)
        log.info("Calling OpenAI (%s)...", cfg["openai"]["model"])
        try:
            ai_response = call_openai(client, prompt, cfg)
            log.info("OpenAI response received")
        except Exception as exc:
            log.error("OpenAI call failed: %s", exc); sys.exit(1)

        for lang in ("en", "fr"):
            if lang not in ai_response or not isinstance(ai_response[lang], dict):
                log.error("OpenAI response missing '%s' section", lang); sys.exit(1)

    # Archive
    archive_path = save_archive(
        {"date": today_str, "regime": regime_data, "selected": selected_syms,
         "candidates_count": len(candidates), "ai_response": ai_response},
        cfg["outputs"]["archive_folder"], today_str,
    )
    log.info("JSON archive: %s", archive_path)

    # PDFs
    reports_dir = Path(cfg["outputs"]["reports_folder"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    for lang, title_key, suffix in (("en","title_en","EN"), ("fr","title_fr","FR")):
        title    = cfg["report"].get(title_key, "SB-ITM Morning Report")
        out_path = reports_dir / f"{today_str}_SB-ITM_Morning_Candidate_Review_{suffix}.pdf"
        log.info("Generating %s PDF...", suffix)
        generate_pdf(out_path, lang, title, display_date,
                     candidates, selected_syms, ai_response[lang], regime_data)
        log.info("PDF saved: %s", out_path)

    log.info("AI Report Bot completed successfully")


if __name__ == "__main__":
    main()
