# DAIC-WOZ Depression Text Feature Project Summary

## 1. Project Goal

Treat the LLM as a **structured feature extractor** that identifies interpretable linguistic, pragmatic, and narrative patterns associated with PHQ-8-based depression screening labels in DAIC-WOZ interview transcripts.

---

## 2. Data Overview

### Dataset
- **Name**: DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz)
- **Format**: Semi-structured clinical interviews conducted by virtual agent "Ellie"
- **Transcript structure**: Alternating `Ellie` (interviewer) and `Participant` turns

### Labels
- **Source files**:
  - `train_split_Depression_AVEC2017.csv`
  - `dev_split_Depression_AVEC2017.csv`
- **Key variables**:
  - `PHQ8_Binary`: 0 (depression-negative) vs 1 (depression-positive)
  - `PHQ8_Score`: Continuous score (0-24)

### Important Framing Notes
**Recommended terminology**:
- "PHQ-8-based depression screening labels"
- "depression-positive vs depression-negative"

**Avoid**:
- "actual depression patients" (overstates clinical validity)
- "depression diagnosis" (PHQ-8 is a screening tool, not clinical diagnosis)

---

## 3. Core Research Question

> Which linguistic, discourse-level, and narrative features in DAIC-WOZ transcripts are associated with PHQ-8-based depression screening labels, and can these features be extracted in an interpretable and reproducible way?

---

## 4. Multi-Level Analysis Design

### Level 1: Utterance-Level
**Purpose**: Capture local surface cues

**Features extracted**:
- Lexical negativity counts
- Abrupt closure markers
- Vague expressions
- Fragment detection

### Level 2: Question-Block-Level (PRIMARY UNIT)
**Definition**: One `Ellie` question + all subsequent `Participant` responses until next Ellie turn

**Why this level matters**:
- Preserves conversational context
- Captures elaboration and responsiveness differences
- Allows question-specific comparison across participants

**Features extracted**:
- Response length (word count)
- Negative/positive affect density
- Social word density
- Hedge density
- Abrupt closure ratio
- Fragment ratio
- Lexical diversity (TTR)

### Level 3: Session-Level
**Purpose**: Aggregate question-block features across full interview

**Aggregation methods**:
- Mean, variance across blocks
- Ratios (e.g., proportion of blocks with abrupt closure)
- Question-category-specific means

---

## 5. Why Pilot Test?

### Purpose of Pilot Testing

The pilot test is **critical** before full-scale LLM extraction. It serves these purposes:

1. **Validate variance**: Confirm LLM scoring produces meaningful variance across transcripts (not all 3s or 4s)

2. **Check reliability**: Run the same response 3-5 times to measure intra-LLM consistency
   - Compute ICC (Intraclass Correlation Coefficient) or Krippendorff's alpha
   - Minimum acceptable ICC: 0.70 (fair), 0.80+ (good)

3. **Verify evidence spans**: Ensure extracted quotes are meaningful and traceable back to the response

4. **Identify differentiating dimensions**: Which rubric dimensions actually separate PHQ8 groups?
   - Some dimensions may show strong signal (keep)
   - Some may show no discrimination (drop or refine)

5. **Refine anchors**: Adjust anchor examples if scores cluster unexpectedly

6. **Filter weak features**: Drop dimensions that show no discrimination or poor reliability before investing in full extraction

### Pilot Test is NOT About Final Performance

Don't confuse pilot test with model evaluation:
- Pilot = "Do these features make sense and are they extractable reliably?"
- Evaluation = "Do these features predict PHQ-8 labels well?" (later stage)

---

## 6. Feature Taxonomy

### 6.1 Rule-Based Features (Stage 1)

| Feature | Description | Extraction Method |
|---------|-------------|-------------------|
| `response_length` | Word count per question-block | Token count |
| `fragment_ratio` | Proportion of utterances < 5 words | Count-based |
| `neg_density` | Negative affect words per 100 words | Lexicon matching |
| `pos_density` | Positive affect words per 100 words | Lexicon matching |
| `social_density` | Social/relationship words per 100 words | Lexicon matching |
| `hedge_density` | Hedge expressions per 100 words | Pattern matching |
| `vague_density` | Vague terms per 100 words | Lexicon matching |
| `abrupt_ratio` | Proportion of responses with closure markers | Pattern matching |
| `uncertainty_density` | Uncertainty markers per 100 words | Pattern matching |
| `lexical_diversity` | Type-token ratio (TTR) | Vocabulary analysis |

### 6.2 LLM Rubric Features (Stage 2)

#### Core Features (Original 5)
| Feature | Description | Scale |
|---------|-------------|-------|
| `detail_richness` | Amount of specific detail, examples, anecdotes | 1-5 |
| `narrative_coherence` | Temporal/causal organization of response | 1-5 |
| `social_connectedness` | Warmth and depth in describing relationships | 1-5 |
| `engagement_level` | Willingness to elaborate vs. avoidance | 1-5 |
| `emotional_valence` | Overall affective tone (negative to positive) | 1-5 |

#### Extended Features (Added Based on Discussion)
| Feature | Description | Scale | Why Added |
|---------|-------------|-------|-----------|
| `episodic_specificity` | Specific episodes vs generic statements | 1-5 | Depression linked to overgeneralized memory |
| `disengagement_avoidance` | Conversation-closing behavior | 1-5 | Captures active avoidance patterns |
| `interpersonal_warmth` | Warmth in relationship descriptions | 1-5 | Narrower than social_connectedness |
| `laughter_context` | Context of `<laughter>` annotations | Categorical | Positive vs nervous/avoidance laughter |

---

## 7. Sanity Check Results (Rule-Based)

### Analysis Summary
- **Total question-blocks analyzed**: 1,104
- **Depression-negative (PHQ8=0)**: 791 blocks
- **Depression-positive (PHQ8=1)**: 313 blocks

### Feature Comparison Results

| Feature | Mean (Ctrl) | Mean (Dep) | Cohen's d | p-value | Keep? |
|---------|------------|------------|-----------|---------|-------|
| `neg_density` | 0.0036 | 0.0071 | **+0.146** | 0.029* | **YES** |
| `pos_density` | 0.0183 | 0.0145 | -0.091 | 0.173 | Maybe |
| `abrupt_ratio` | 0.0851 | 0.1069 | +0.090 | 0.179 | Maybe |
| `lexical_diversity` | 0.7990 | 0.8086 | +0.060 | 0.372 | No |
| `uncertainty_density` | 0.0076 | 0.0092 | +0.047 | 0.480 | No |
| `hedge_density` | 0.0135 | 0.0155 | +0.045 | 0.498 | No |
| `response_length` | 36.79 | 35.14 | -0.040 | 0.544 | No |
| `social_density` | 0.0258 | 0.0237 | -0.035 | 0.602 | No |
| `fragment_ratio` | 0.3999 | 0.4086 | +0.024 | 0.719 | No |
| `vague_density` | 0.0092 | 0.0093 | +0.002 | 0.976 | No |

### Key Findings

1. **`neg_density` is the only statistically significant feature** (p < 0.05)
   - Depression-positive group uses ~2x more negative affect words

2. **Surprising null results**:
   - `response_length`: No significant difference (contradicts "depressed people speak less" assumption)
   - `vague_density`, `fragment_ratio`: Nearly identical between groups

3. **Implications for LLM features**:
   - Surface lexical features have limited discriminative power
   - **Discourse-level features** (narrative coherence, engagement, detail richness) may be more important
   - This justifies the need for LLM-based scoring

### Feature Decision Summary

**Keep (Rule-based)**:
- `neg_density` (significant)
- `pos_density` (theoretical importance: anhedonia)
- `abrupt_ratio` (borderline, theoretically relevant)

**Remove**:
- `vague_density`
- `fragment_ratio`
- `response_length` (as standalone feature)
- `social_density` (rule-based version weak; use LLM version)
- `uncertainty_density`
- `hedge_density`

**Defer to speech modality**:
- Filler/disfluency features (`uh`, `um`) - better captured with audio

---

## 8. Target Question Subset

### Priority Questions for LLM Analysis

| Category | Question Pattern | Why Important |
|----------|-----------------|---------------|
| Self-description | "How would you describe yourself?" | Self-concept, negative self-view |
| Social relationships | "How would your best friend describe you?" | Interpersonal warmth |
| Social relationships | "How close are you to your family?" | Social connectedness |
| Positive affect recall | "When was the last time you felt really happy?" | Anhedonia detection |
| Daily life/interests | "What do you do for fun?" | Engagement, pleasure capacity |
| Daily life/interests | "What do you do to relax?" | Behavioral activation |
| Negative affect recall | "When was the last time you argued with someone?" | Emotional processing |
| Life narrative | "Is there anything you regret?" | Rumination, negative cognition |
| Current state | "How are you doing today?" | Baseline mood assessment |

### Low-Priority Questions (Skip in Pilot)
- "Where are you from originally?"
- "What did you study at school?"
- "Do you travel a lot?"
- Simple factual/backchannel responses

---

## 9. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: RULE-BASED                          │
│              (01_rule_based_before_pilot/)                      │
├─────────────────────────────────────────────────────────────────┤
│  1. Load transcripts + labels                                   │
│  2. Segment into question-blocks                                │
│  3. Filter to target questions                                  │
│  4. Extract rule-based features                                 │
│  5. Compare distributions (PHQ8=0 vs 1)                         │
│  6. Identify weak features for removal                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: LLM RUBRIC                          │
│            (02_real_pilot_rubric_features/)                     │
├─────────────────────────────────────────────────────────────────┤
│  Purpose: PILOT TEST to validate LLM features                   │
│                                                                 │
│  1. Select question-blocks for LLM scoring                      │
│  2. Apply structured rubric prompt (8 dimensions + laughter)    │
│  3. Extract scores + evidence spans                             │
│  4. Check reliability (same response 3-5 times)                 │
│  5. Identify differentiating dimensions                         │
│  6. Refine rubric before full extraction                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 3: MULTIMODAL                          │
│                     (Future work)                               │
├─────────────────────────────────────────────────────────────────┤
│  1. Extract speech embeddings (Whisper v3)                      │
│  2. Fuse text + speech features                                 │
│  3. Train multimodal classifier                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Transcript Observations

### Qualitative Patterns from Reading Transcripts

**Observed in depression-positive transcripts**:
- `<laughter>` after avoidance statements: `"i don't know <laughter>"`
- Fragmentary friend descriptions: `"chocolate tall thin"` (attribute listing without relationship)
- Self-repair/restarts: `"i i i'm"`
- Abrupt closures: `"that's it"`, `"not at all"`

**Observed in both groups**:
- `<laughter>` can appear with positive OR negative content
- Response length alone doesn't discriminate (some dep+ have long responses about trauma)

**Implication for LLM features**:
- Context matters more than simple counts
- `laughter_context` should classify laughter as positive/negative/avoidance
- Relationship description quality matters more than length

---

## 11. Modeling Position

### Text Branch Should Stand Alone
- The text feature extraction pipeline should be independently valid
- Speech features will be added later as a separate modality
- This allows contribution even if multimodal fusion doesn't improve results

### Do NOT Overemphasize at Text Stage
- Filler/disfluency features (`uh`, `um`, pauses)
- These are better captured from audio with timing information

---

## 12. Writing/Framing Guidelines

### Preferred Language
- "depression-associated linguistic patterns"
- "PHQ-8-based screening labels"
- "structured feature extraction"
- "interpretable discourse and narrative signals"

### Avoid
- "LLM predicts depression" (makes it sound like black-box diagnosis)
- "actual depression patients" (overstates clinical validity)
- "depression diagnosis" (PHQ-8 is screening, not diagnosis)

### Paper Narrative Structure
1. **Problem**: Existing approaches use keyword counts or black-box classifiers
2. **Gap**: Clinical interviews contain richer discourse-level signals
3. **Approach**: Use LLM as structured feature extractor, not judge
4. **Contribution**: Interpretable, reproducible features at multiple granularities

---

## 13. File Structure

```
DAIC-WOZ-Experiments/
├── README.md
├── docs/
│   └── project_summary.md          (this file)
├── 01_rule_based_before_pilot/
│   ├── rule_based_question_block_sanity_check.py
│   ├── question_block_features.csv
│   └── feature_comparison_results.csv
└── 02_real_pilot_rubric_features/
    ├── llm_structured_feature_extractor.py   (LLM-only, no rule-based)
    ├── pilot_llm_features.csv                (output after running pilot)
    └── prompts/
        └── rubric_schema.md                  (8 dimensions + laughter_context)
```

---

## 14. Immediate Next Steps

1. **Choose LLM API**: OpenAI GPT-4 or Anthropic Claude
2. **Configure `call_llm_api()`** in `llm_structured_feature_extractor.py`
3. **Run pilot on ~20 participants**, 5 target questions each
4. **Check reliability**: Same response 3-5 times, compute ICC
5. **Analyze pilot results**:
   - Which dimensions show variance?
   - Which dimensions differentiate PHQ8 groups?
   - Are evidence spans meaningful?
6. **Refine rubric** based on pilot findings
7. **Full extraction** after rubric is validated

---

*Last updated: 2026-04-06*
*Maintained by Team Olivier (Prof. Claude & Prof. Codex)*
