"""
Question-block level sanity check for DAIC-WOZ depression feature extraction.
Segments transcripts by Ellie questions, extracts rule-based features per block,
and compares distributions between PHQ8_Binary=0 vs 1 groups.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import re
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("/Users/user/Desktop/DAIC-WOZ")
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
LABEL_FILES = [
    DATA_DIR / "labels" / "train_split_Depression_AVEC2017.csv",
    DATA_DIR / "labels" / "dev_split_Depression_AVEC2017.csv"
]

# Target questions (partial match, case-insensitive)
# These are open-ended questions where elaboration differences may appear
TARGET_QUESTION_PATTERNS = [
    r"how would you describe yourself",
    r"how would your best friend describe you",
    r"how close.*family",
    r"tell me about.*family",
    r"tell me about.*relationship",
    r"when was the last time you felt really happy",
    r"last time you.*happy",
    r"what makes you (angry|upset)",
    r"when did you last argue",
    r"last.*argument",
    r"what do you do.*fun",
    r"what do you do.*relax",
    r"what do you enjoy",
    r"do you consider yourself.*introvert",
    r"how are you at making decisions",
    r"is there anything you regret",
    r"if you could go back.*change",
    r"how easy.*for you to get a good night",
    r"what are you most proud of",
    r"what would you want to change about yourself",
]

# Lexicons
NEGATIVE_WORDS = {
    'sad', 'depressed', 'unhappy', 'miserable', 'hopeless', 'worthless',
    'tired', 'exhausted', 'lonely', 'alone', 'empty', 'numb',
    'anxious', 'worried', 'stressed', 'frustrated', 'angry', 'upset',
    'hurt', 'pain', 'suffering', 'struggle', 'difficult', 'hard',
    'hate', 'awful', 'terrible', 'horrible', 'bad', 'worse', 'worst',
    'fail', 'failed', 'failure', 'losing', 'lost', 'cry', 'crying'
}

POSITIVE_WORDS = {
    'happy', 'joy', 'excited', 'glad', 'pleased', 'wonderful',
    'great', 'good', 'amazing', 'awesome', 'fantastic', 'excellent',
    'love', 'loved', 'enjoy', 'enjoyed', 'fun', 'laugh', 'laughing',
    'hope', 'hopeful', 'confident', 'proud', 'grateful', 'thankful',
    'peaceful', 'calm', 'relaxed', 'comfortable', 'satisfied'
}

SOCIAL_WORDS = {
    'friend', 'friends', 'family', 'mom', 'dad', 'mother', 'father',
    'brother', 'sister', 'husband', 'wife', 'boyfriend', 'girlfriend',
    'partner', 'people', 'someone', 'together', 'relationship', 'close',
    'talk', 'talking', 'hang', 'hanging', 'visit', 'meet', 'met'
}

ABRUPT_CLOSURES = [
    "i don't know", "i dunno", "don't know",
    "that's it", "that's about it", "that's all",
    "nothing much", "not much", "nothing really",
    "i guess", "i suppose", "whatever",
    "not really", "no", "nope", "yeah", "yes", "sure"
]

HEDGE_WORDS = {
    'maybe', 'perhaps', 'probably', 'possibly', 'might', 'could',
    'kind of', 'kinda', 'sort of', 'sorta', 'somewhat', 'fairly',
    'i think', 'i guess', 'i suppose', 'i believe'
}

VAGUE_WORDS = {'stuff', 'things', 'something', 'anything', 'whatever', 'somehow'}

UNCERTAINTY_MARKERS = {"i don't know", "i'm not sure", "not sure", "maybe", "i guess"}


# ============================================================
# Helper Functions
# ============================================================

def load_labels():
    """Load and combine train/dev labels."""
    dfs = []
    for f in LABEL_FILES:
        if f.exists():
            dfs.append(pd.read_csv(f))
    labels = pd.concat(dfs, ignore_index=True)
    return labels.set_index('Participant_ID')[['PHQ8_Binary', 'PHQ8_Score']].to_dict('index')


def load_transcript(participant_id):
    """Load a single transcript CSV."""
    path = TRANSCRIPT_DIR / f"{participant_id}_TRANSCRIPT.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep='\t')
        if 'speaker' not in df.columns:
            df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def matches_target_question(question_text):
    """Check if question matches any target pattern."""
    q_lower = question_text.lower().strip()
    for pattern in TARGET_QUESTION_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def segment_question_blocks(transcript_df):
    """
    Segment transcript into question-blocks.
    Each block = one Ellie question + all following Participant responses until next Ellie turn.
    """
    blocks = []
    current_question = None
    current_responses = []

    for _, row in transcript_df.iterrows():
        speaker = str(row.get('speaker', '')).strip()
        value = str(row.get('value', '')).strip()

        if speaker == 'Ellie':
            # Save previous block if exists
            if current_question and current_responses:
                blocks.append({
                    'question': current_question,
                    'responses': current_responses,
                    'combined_text': ' '.join(current_responses)
                })
            current_question = value
            current_responses = []
        elif speaker == 'Participant':
            if value and value.lower() not in ['', 'nan']:
                current_responses.append(value)

    # Don't forget last block
    if current_question and current_responses:
        blocks.append({
            'question': current_question,
            'responses': current_responses,
            'combined_text': ' '.join(current_responses)
        })

    return blocks


def extract_block_features(block):
    """Extract rule-based features from a single question-block."""
    text = block['combined_text'].lower()
    words = re.findall(r'\b\w+\b', text)
    n_words = len(words) if words else 1  # avoid div by zero

    # Response length
    response_length = len(words)
    n_responses = len(block['responses'])

    # Sentence/fragment analysis
    sentences = re.split(r'[.!?]+', block['combined_text'])
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = len(sentences) if sentences else 1
    avg_sentence_length = n_words / n_sentences

    # Fragment detection (very short responses, < 5 words)
    fragments = sum(1 for r in block['responses'] if len(r.split()) < 5)
    fragment_ratio = fragments / n_responses if n_responses > 0 else 0

    # Lexicon densities
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    social_count = sum(1 for w in words if w in SOCIAL_WORDS)
    vague_count = sum(1 for w in words if w in VAGUE_WORDS)

    neg_density = neg_count / n_words
    pos_density = pos_count / n_words
    social_density = social_count / n_words
    vague_density = vague_count / n_words

    # Hedge detection
    hedge_count = sum(1 for h in HEDGE_WORDS if h in text)
    hedge_density = hedge_count / n_words

    # Abrupt closure detection
    abrupt_count = 0
    for response in block['responses']:
        r_lower = response.lower().strip()
        for closure in ABRUPT_CLOSURES:
            if r_lower == closure or r_lower.startswith(closure + ' ') or r_lower.endswith(' ' + closure):
                abrupt_count += 1
                break
    abrupt_ratio = abrupt_count / n_responses if n_responses > 0 else 0

    # Uncertainty markers
    uncertainty_count = sum(1 for u in UNCERTAINTY_MARKERS if u in text)
    uncertainty_density = uncertainty_count / n_words

    # Lexical diversity (type-token ratio)
    unique_words = set(words)
    lexical_diversity = len(unique_words) / n_words if n_words > 0 else 0

    return {
        'response_length': response_length,
        'n_responses': n_responses,
        'avg_sentence_length': avg_sentence_length,
        'fragment_ratio': fragment_ratio,
        'neg_density': neg_density,
        'pos_density': pos_density,
        'social_density': social_density,
        'vague_density': vague_density,
        'hedge_density': hedge_density,
        'abrupt_ratio': abrupt_ratio,
        'uncertainty_density': uncertainty_density,
        'lexical_diversity': lexical_diversity
    }


def compute_effect_size(group0, group1):
    """Compute Cohen's d."""
    n0, n1 = len(group0), len(group1)
    if n0 < 2 or n1 < 2:
        return np.nan
    pooled_std = np.sqrt(((n0-1)*np.var(group0, ddof=1) + (n1-1)*np.var(group1, ddof=1)) / (n0+n1-2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(group1) - np.mean(group0)) / pooled_std


# ============================================================
# Main Analysis
# ============================================================

def main():
    print("=" * 60)
    print("DAIC-WOZ Question-Block Level Sanity Check")
    print("=" * 60)

    # Load labels
    labels = load_labels()
    print(f"\nLoaded labels for {len(labels)} participants")

    # Process all transcripts
    all_block_features = []
    participant_ids = [int(p.stem.split('_')[0]) for p in TRANSCRIPT_DIR.glob('*_TRANSCRIPT.csv')]

    for pid in sorted(participant_ids):
        if pid not in labels:
            continue

        transcript = load_transcript(pid)
        if transcript is None:
            continue

        blocks = segment_question_blocks(transcript)
        phq_binary = labels[pid]['PHQ8_Binary']
        phq_score = labels[pid]['PHQ8_Score']

        for block in blocks:
            if matches_target_question(block['question']):
                features = extract_block_features(block)
                features['participant_id'] = pid
                features['phq8_binary'] = phq_binary
                features['phq8_score'] = phq_score
                features['question'] = block['question'][:80]  # truncate for display
                all_block_features.append(features)

    df = pd.DataFrame(all_block_features)
    print(f"\nExtracted {len(df)} question-blocks from target questions")
    print(f"PHQ8_Binary=0: {len(df[df['phq8_binary']==0])} blocks")
    print(f"PHQ8_Binary=1: {len(df[df['phq8_binary']==1])} blocks")

    # Save raw data
    output_path = DATA_DIR / "question_block_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved features to: {output_path}")

    # Group comparison
    print("\n" + "=" * 60)
    print("GROUP COMPARISON: PHQ8_Binary=0 vs PHQ8_Binary=1")
    print("=" * 60)

    feature_cols = [
        'response_length', 'avg_sentence_length', 'fragment_ratio',
        'neg_density', 'pos_density', 'social_density', 'vague_density',
        'hedge_density', 'abrupt_ratio', 'uncertainty_density', 'lexical_diversity'
    ]

    group0 = df[df['phq8_binary'] == 0]
    group1 = df[df['phq8_binary'] == 1]

    results = []
    for col in feature_cols:
        g0_vals = group0[col].dropna()
        g1_vals = group1[col].dropna()

        mean0 = g0_vals.mean()
        mean1 = g1_vals.mean()

        t_stat, p_val = stats.ttest_ind(g0_vals, g1_vals)
        d = compute_effect_size(g0_vals.values, g1_vals.values)

        # Mann-Whitney U for robustness
        u_stat, u_pval = stats.mannwhitneyu(g0_vals, g1_vals, alternative='two-sided')

        results.append({
            'feature': col,
            'mean_ctrl': mean0,
            'mean_dep': mean1,
            'diff': mean1 - mean0,
            't_stat': t_stat,
            'p_value': p_val,
            'cohens_d': d,
            'u_pval': u_pval
        })

        sig = "*" if p_val < 0.05 else ""
        direction = "↑" if mean1 > mean0 else "↓"
        print(f"{col:25s}: ctrl={mean0:.4f}, dep={mean1:.4f}, d={d:+.3f} {direction} {sig}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Features ranked by |Cohen's d|")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    results_df['abs_d'] = results_df['cohens_d'].abs()
    results_df = results_df.sort_values('abs_d', ascending=False)

    print("\nTop features (|d| > 0.1):")
    for _, row in results_df.iterrows():
        if abs(row['cohens_d']) > 0.1:
            sig = "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            print(f"  {row['feature']:25s}: d={row['cohens_d']:+.3f}, p={row['p_value']:.4f} {sig}")

    print("\nWeak features (|d| < 0.1) - candidates for removal:")
    for _, row in results_df.iterrows():
        if abs(row['cohens_d']) <= 0.1:
            print(f"  {row['feature']:25s}: d={row['cohens_d']:+.3f}")

    # Save results
    results_df.to_csv(DATA_DIR / "feature_comparison_results.csv", index=False)
    print(f"\nSaved comparison results to: {DATA_DIR / 'feature_comparison_results.csv'}")

    return df, results_df


if __name__ == "__main__":
    df, results = main()
