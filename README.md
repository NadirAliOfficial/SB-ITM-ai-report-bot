# SB-ITM AI Report Bot

Generates institutional-grade bilingual (EN/FR) PDF morning reports by combining outputs from the SB-ITM Stock Scanner, Bridge Bot, and Regime Bot — powered by OpenAI GPT-4o.

---

## What It Does

Runs each morning after the scan workflow completes. It:

1. Auto-detects the latest scan CSV, Bridge Bot log, and Regime Bot log
2. Sends all candidate data and scanner flags to GPT-4o for structured analysis
3. Generates two institutional PDF reports — English and French
4. Archives the raw AI JSON response for auditing

---

## Output

Each run produces:

- `reports/YYYYMMDD_SB-ITM_Morning_Candidate_Review_EN.pdf`
- `reports/YYYYMMDD_SB-ITM_Morning_Candidate_Review_FR.pdf`
- `reports/archive/YYYYMMDD_AI_Report_Bot.json`
- `logs/YYYY_MM_DD_AI_Report_Bot.log`

### Report Sections

1. Executive Summary
2. Key Daily Observations
3. Regime Analysis
4. Bridge Bot Selected Candidates (full per-candidate cards)
5. Additional Notable Candidates (Strong + Secondary Watchlist)
6. Daily Scan Assessment
7. Comparative Summary Table
8. Final Neutral Conclusion

---

## Setup

### Requirements

- Python 3.10+
- Windows (Scheduled Task compatible) or Linux/macOS

### Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Configure

Edit `config/ai_report_bot.yaml` and set the correct input folder paths:

```yaml
inputs:
  scan_csv_folder: "C:\\stock_scanner\\output"
  bridge_bot_log_folder: "C:\\ibkr-risk-bot\\logs"
  regime_bot_log_folder: "C:\\stock_scanner\\output"
```

### OpenAI API Key

Set via environment variable (recommended):

```bash
set OPENAI_API_KEY=sk-...        # Windows
export OPENAI_API_KEY=sk-...     # macOS/Linux
```

Or add directly to `config/ai_report_bot.yaml` under `openai.api_key`.

---

## Usage

### Normal run

```bash
.venv\Scripts\python ai_report_bot.py
```

### Dry run (file detection only, no PDF)

```bash
.venv\Scripts\python ai_report_bot.py --dry-run
```

### Mock run (full PDF with placeholder content, no OpenAI key needed)

```bash
.venv\Scripts\python ai_report_bot.py --mock
```

### Custom config path

```bash
.venv\Scripts\python ai_report_bot.py --config path\to\config.yaml
```

---

## Windows Scheduled Task

Schedule after Regime Bot completes (e.g. 09:15 AM):

```
Program: C:\ai-report-bot\.venv\Scripts\python.exe
Arguments: C:\ai-report-bot\ai_report_bot.py
Start in: C:\ai-report-bot
```

Set `OPENAI_API_KEY` in the system environment variables or in the Scheduled Task environment.

---

## Input Files

| Input | Pattern | Source Bot |
|---|---|---|
| Morning scan CSV | `scan_*.csv` | Stock Scanner |
| Bridge Bot log | `*_Bridge_Bot.log` | Bridge Bot |
| Regime Bot log | `regime_*.log` | Regime Bot |

The bot always picks the **most recent** file matching each pattern.

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai>=1.0.0` | GPT-4o API |
| `PyYAML>=6.0` | Config file parsing |
| `reportlab>=4.0` | PDF generation |

---

## Branding

- Dark Blue `#193F56`
- Orange `#F26022`
- A4 format, bilingual EN/FR
- Disclaimer: neutral technical research, not investment advice
<!-- updated: 2026-01-12-03 -->
