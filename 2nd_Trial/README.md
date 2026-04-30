# MDTD: Multimodal Depression Temporal Detector

A multimodal temporal model for depression detection. Explicitly models **disagreement** between text and speech, and **temporal dynamics** to enable interpretable and clinically meaningful predictions.

## Core Hypothesis

> Depressed patients are likely to show **disagreement** between **what they say (text)** and **how they say it (speech)**.
> Additionally, **engagement and energy decrease** as the conversation progresses.

---

## Core Ideas

### 1: Cross-Modal Disagreement via Contrastive Alignment

**Problem**:
- Existing multimodal approaches simply concatenate or late-fuse text and speech
- Text encoder (RoBERTa) and Speech encoder (Whisper) are trained in different embedding spaces, making direct comparison impossible

**Proposal**:
1. Use contrastive learning to **align both modalities in the same space**
2. Explicitly measure **disagreement** as a feature in the aligned space

```
┌─────────────────────────────────────────────────────────────┐
│                   Joint Space Alignment                     │
│                                                             │
│  Text ──→ RoBERTa ──→ Projection ──┐                        │
│                                    ├──→ Contrastive Learning│
│  Speech ──→ Whisper ──→ Projection ┘                        │
│                                                             │
│  Objective: Same utterance (text, speech) → close           │
│             Different utterance (text, speech) → far        │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Disagreement Measurement                   │
│                                                             │
│  Text: "Yes, I'm fine" (positive content)                   │
│  Speech: slow pace, low tone, long pause (negative signal)  │
│                          ↓                                  │
│  Disagreement = 1 - cosine_similarity(text_emb, speech_emb) │
│                          ↓                                  │
│  High Disagreement Score → Depression risk signal           │
└─────────────────────────────────────────────────────────────┘
```

**Clinical Significance**:
- Depressed patients say "I'm fine" to meet social expectations, but lethargy is revealed in their voice
- This disagreement is interpretable and explainable to clinicians

---

### 2: Hybrid Temporal Dynamics

**Problem**: Existing research analyzes single snapshots (session averages).

**Proposal**: Track temporal changes using a hybrid approach combining **LLM Rubric + Speech Encoder**.

```
┌─────────────────────────────────────────────────────────────┐
│                     Question Block i                        │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌────────────┐     ┌────────────┐     ┌────────────┐
    │   Text     │     │   Speech   │     │    Text    │
    │  Encoder   │     │  Encoder   │     │  → LLM     │
    │            │     │            │     │   Rubric   │
    └─────┬──────┘     └─────┬──────┘     └─────┬──────┘
          │                  │                  │
          ▼                  ▼                  ▼
       t_emb              s_emb           engagement=3
       (256-d)           (256-d)          detail=4
          │                  │            valence=2
          │                  │                  │
          └────────┬─────────┘                  │
                   ▼                            │
            Disagreement                        │
                   │                            │
                   └──────────┬─────────────────┘
                              ▼
                    Block Feature Vector
                    [t; s; disagree; llm_scores]
```

**Tracked Changes**:

| Source | Features | Measurement |
|--------|----------|-------------|
| **Cross-Modal** | Disagreement trend | Encoder-based, automatic |
| **LLM Rubric** | engagement, detail_richness, emotional_valence | LLM evaluation, interpretable |
| **Speech** | Prosody features (pace, F0, pause) | Encoder-based |

**Temporal Change Patterns**:
```
Early → Mid → Late conversation
  ↓       ↓       ↓
engagement: 3 → 2 → 1  (fatigue = depression signal)
disagreement: 0.2 → 0.3 → 0.5 (increasing = accumulated fatigue)
```

**Why Hybrid?**

| Method | Pros | Cons |
|--------|------|------|
| Encoder only | End-to-end, fast | Black-box |
| LLM only | Interpretable | No speech info, costly |
| **Hybrid** | Interpretable + Multimodal | Higher complexity |

**Clinical Significance**:
- Depression-related "fatigue" manifests in later parts of conversation
- LLM rubric explains **why** changes occurred ("engagement dropped 3→1")
- Speech captures **how** changes occurred (slower pace, longer pauses)

---

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Full Interview Session                    │
│ [Block 1] ─── [Block 2] ─── [Block 3] ─── ... ─── [Block N] │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Per-Block Hybrid Feature Extraction              │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Encoder Branch (Automatic)                              │ │
│ │   Text ──→ RoBERTa ──→ Projection ──→ t_i (256-d)       │ │
│ │   Speech ──→ Whisper ──→ Projection ──→ s_i (256-d)     │ │
│ │   Disagreement: d_i = 1 - cos(t_i, s_i)                 │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ LLM Rubric Branch (Interpretable)                       │ │
│ │   Text ──→ LLM API ──→ engagement_i      (1-5)          │ │
│ │                    ──→ detail_richness_i (1-5)          │ │
│ │                    ──→ emotional_valence_i (1-5)        │ │
│ │                    ──→ social_warmth_i   (1-5)          │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Block Feature: h_i = [t_i; s_i; d_i; llm_scores_i]          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Temporal Modeling                        │
│                                                             │
│ h_1 ──→ h_2 ──→ h_3 ──→ ... ──→ h_N                         │
│                  │                                          │
│                  ▼                                          │
│           Bi-LSTM (2 layers)                                │
│                  │                                          │
│                  ▼                                          │
│ Temporal Slopes:                                            │
│   - disagreement_slope (Encoder-based)                      │
│   - engagement_slope (LLM-based)                            │
│   - detail_slope (LLM-based)                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Classification Head                       │
│                                                             │
│ Input: [LSTM_hidden; slopes; mean_features]                 │
│                        │                                    │
│                        ▼                                    │
│                 MLP + Sigmoid                               │
│                        │                                    │
│                        ▼                                    │
│               P(Depression) = 0.73                          │
└─────────────────────────────────────────────────────────────┘
```

---

## LLM Rubric Dimensions (Example)

| Dimension | Scale | Description |
|-----------|-------|-------------|
| `engagement` | 1-5 | Conversation engagement, willingness to elaborate |
| `detail_richness` | 1-5 | Specific information (time, place, person) |
| `emotional_valence` | 1-5 | Emotional tone (1=negative, 5=positive) |
| `social_warmth` | 1-5 | Warmth in relationship descriptions |
| `episodic_specificity` | 1-5 | Specific episodes vs overgeneralization |

---

## Training Objectives

```
Total Loss = λ₁·L_classification + λ₂·L_contrastive

1. Classification Loss (Main Task)
   L_cls = BCE(P(depression), y_true)

2. Contrastive Loss (Joint Space Alignment)
   L_contra = InfoNCE(text_embeddings, speech_embeddings)
```

---

## Project Structure

```
2nd_Trial/
├── README.md              # This document
├── config.py              # Hyperparameter settings
├── dataset.py             # DAIC-WOZ data loading and preprocessing
├── model.py               # DepressionDetector model definition
├── train.py               # Training pipeline
├── evaluate.py            # Evaluation and interpretation report
└── utils.py               # Utility functions
```

---

## Interpretable Output Example

```json
{
  "participant_id": 301,
  "depression_probability": 0.73,
  "risk_level": "HIGH",

  "cross_modal_analysis": {
    "avg_disagreement": 0.42,
    "interpretation": "Mismatch between verbal content and vocal tone"
  },

  "temporal_analysis": {
    "engagement_trend": "DECLINING",
    "engagement_slope": -0.15,
    "disagreement_trend": "INCREASING",
    "disagreement_slope": 0.08,
    "interpretation": "Engagement decreases (3→1), disagreement increases toward end"
  },

  "llm_rubric_summary": {
    "avg_engagement": 2.1,
    "avg_detail_richness": 1.8,
    "avg_emotional_valence": 2.3,
    "interpretation": "Overall low engagement, specificity, and negative affect"
  },

  "high_risk_blocks": [
    {
      "block": 3,
      "question": "How have you been feeling?",
      "disagreement": 0.51,
      "engagement": 1,
      "detail": 2
    },
    {
      "block": 7,
      "question": "Tell me about your relationships",
      "disagreement": 0.48,
      "engagement": 2,
      "social_warmth": 1
    }
  ]
}
```

---

## Dataset

- **DAIC-WOZ**: Distress Analysis Interview Corpus
- **Format**: Semi-structured clinical interview with virtual agent "Ellie"
- **Unit**: Question-block (Ellie question + participant responses)
- **Label**: PHQ-8 Binary (0 = normal, 1 = depressed)

---

## Reference

- CLIP: Learning Transferable Visual Models From Natural Language Supervision
- DAIC-WOZ: The Distress Analysis Interview Corpus
- Whisper: Robust Speech Recognition via Large-Scale Weak Supervision
