# DAIC-WOZ Depression Feature Extraction Experiments

LLM-based structured feature extraction for depression-associated linguistic patterns in clinical interview transcripts.

## Project Overview

This project uses LLMs as **structured feature extractors** (not black-box classifiers) to identify interpretable linguistic, pragmatic, and narrative patterns associated with PHQ-8-based depression screening labels.

## Directory Structure

```
DAIC-WOZ-Experiments/
├── README.md                           # This file
├── docs/
│   └── project_summary.md              # Full project documentation
├── 01_rule_based_before_pilot/         # Stage 1: Rule-based sanity check
│   ├── rule_based_question_block_sanity_check.py
│   ├── question_block_features.csv     # Raw extracted features
│   └── feature_comparison_results.csv  # Statistical comparison results
└── 02_real_pilot_rubric_features/      # Stage 2: LLM-based extraction
    ├── llm_structured_feature_extractor.py
    └── prompts/
        └── rubric_schema.md            # LLM scoring rubric
```

## Quick Start

### Stage 1: Rule-Based Sanity Check

```bash
cd 01_rule_based_before_pilot
python3 rule_based_question_block_sanity_check.py
```

**Output**: Feature comparison between PHQ8=0 and PHQ8=1 groups

### Stage 2: LLM Feature Extraction

1. Configure API in `llm_structured_feature_extractor.py`
2. Review rubric in `prompts/rubric_schema.md`
3. Run extraction (API required)

## Key Results (Rule-Based)

| Feature | Cohen's d | Keep? |
|---------|-----------|-------|
| neg_density | **+0.146*** | YES |
| pos_density | -0.091 | Maybe |
| abrupt_ratio | +0.090 | Maybe |
| Others | < 0.06 | No |

*Only `neg_density` showed statistically significant difference (p=0.029)*

## Data Requirements

- DAIC-WOZ dataset in `/Users/user/Desktop/DAIC-WOZ/`
- Transcripts in `transcripts/` folder
- Labels in `labels/` folder

## Team

- **Prof. Claude**: Engineering, pipeline implementation, statistical methodology
- **Prof. Codex**: Framing, feature taxonomy, paper writing

---

*Part of the Team Olivier research collaboration*
