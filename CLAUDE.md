# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project works with **a dataset of simulated patient-physician medical interviews** (OSCE format), as described in the paper *"A dataset of simulated patient-physician medical interviews with a focus on respiratory cases"* (Scientific Data, 2022). The goal is NLP/AI research on medical conversations: symptom extraction, disease classification, speech-to-text error detection, or training medical dialogue models.

## Dataset Structure

```
dataset/
  audio/       # 272 MP3 recordings (~11 min avg each)
  text/        # 272 raw ASR transcripts (ALL CAPS, no punctuation, no speaker labels)
  cleantext/   # 272 manually corrected transcripts (proper punctuation, D:/P: speaker labels)
```

**File naming**: 3-letter category prefix + 4-digit case number (e.g. `RES0051.mp3`, `CAR0004.txt`)

| Prefix | Category        | Count | % |
|--------|-----------------|-------|---|
| RES    | Respiratory     | 214   | 78.7% |
| MSK    | Musculoskeletal | 46    | 16.9% |
| GAS    | Gastrointestinal| 6     | 2.2% |
| CAR    | Cardiac         | 5     | 1.8% |
| DER    | Dermatological  | 1     | 0.4% |
| GEN    | General         | —     | — |

## Transcript Format

**Raw (`text/`)**: All-caps, no punctuation, no speaker identification. Direct ASR output before correction.

**Clean (`cleantext/`)**: Manually corrected. Speaker turns marked with `D:` (doctor) and `P:` (patient). Spelling, grammar, and speech-to-text errors fixed. Key information missed by ASR was added back manually.

Each conversation follows OSCE history-taking structure: chief complaint → symptom characterization (onset, location, severity, quality, radiation, modifying factors) → associated symptoms → review of systems → past medical history → medications/allergies → social history → family history.

## Guide Pages

`guia/pagina-1.html` through `guia/pagina-4.html` are saved Gemini conversation pages (from `gemini.google.com/share/b79ba09fd567`). The actual content is JavaScript-rendered and not accessible as static text — open them in a browser to read the annotation/triage instructions they contain.
