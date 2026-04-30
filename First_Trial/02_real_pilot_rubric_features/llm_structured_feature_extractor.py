"""
DAIC-WOZ LLM-Only Rubric Feature Extraction
============================================
Stage 2: Real Pilot Test with LLM Rubric Features

This module extracts DISCOURSE-LEVEL features using LLM-based structured scoring.
All rule-based lexical features have been moved to Stage 1 (01_rule_based_before_pilot/).

PURPOSE OF PILOT TEST:
1. Validate that LLM scoring produces variance across transcripts
2. Check inter-run reliability (same response scored 3-5 times)
3. Verify evidence spans are meaningful and extractable
4. Identify which rubric dimensions differentiate PHQ8 groups
5. Refine rubric anchors before full-scale extraction

"""

import csv
import re
import json
import os
import time
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import statistics

import requests

# =============================================================================
# TARGET QUESTIONS FOR LLM SCORING
# =============================================================================

# Only open-ended questions where elaboration is expected
# These are prioritized for LLM scoring based on pilot analysis
TARGET_QUESTIONS = {
    # Self/personality description
    'introvert': r'do you consider yourself an introvert',
    'best_friend_describe': r'how would your best friend describe you',
    'best_qualities': r'what would you say are some of your best qualities',

    # Social relationships
    'positive_influence': r"who'?s someone that'?s been a positive influence",
    'close_to_family': r'how close are you to your family',
    'tell_me_about_kids': r'tell me about your kids',

    # Emotion/affect recall
    'felt_happy': r'when was the last time you felt really happy',
    'things_for_fun': r'what are some things you like to do for fun',
    'do_to_relax': r'what do you do to relax',
    'good_mood': r'what are some things that usually put you in a good mood',

    # Conflict/negative emotion
    'argued_with_someone': r'when was the last time you argued with someone',
    'controlling_temper': r'how are you at controlling your temper',
    'when_annoyed': r'what do you do when you.?re annoyed',
    'make_you_mad': r'what are some things that make you really mad',

    # Life narrative
    'memorable_experience': r"what'?s one of your most memorable experiences",
    'anything_regret': r'is there anything you regret',
    'proud_of': r'what are you most proud of',

    # Current state
    'how_doing_today': r'how are you doing today',
    'feeling_lately': r'how have you been feeling lately',
    'sleep': r'how easy is it for you to get a good night.?s sleep',
}


def match_target_question(ellie_text: str) -> Optional[str]:
    """Check if Ellie's question matches one of our target questions."""
    text_lower = ellie_text.lower()
    for key, pattern in TARGET_QUESTIONS.items():
        if re.search(pattern, text_lower):
            return key
    return None


# =============================================================================
# DATA CLASSES FOR LLM OUTPUT
# =============================================================================

@dataclass
class LLMRubricScores:
    """LLM-scored rubric features for a question-response block."""
    # Core discourse features
    detail_richness: Optional[int] = None  # 1-5
    narrative_coherence: Optional[int] = None  # 1-5
    social_connectedness: Optional[int] = None  # 1-5
    engagement_level: Optional[int] = None  # 1-5
    emotional_valence: Optional[int] = None  # 1-5 (1=very negative, 5=very positive)

    # Extended features (added based on pilot analysis)
    episodic_specificity: Optional[int] = None  # 1-5: specific episode vs generic
    disengagement_avoidance: Optional[int] = None  # 1-5: conversation-closing behavior
    interpersonal_warmth: Optional[int] = None  # 1-5: warmth in describing relationships

    # Annotation-aware features
    laughter_context: Optional[str] = None  # 'positive', 'negative', 'avoidance', 'neutral', 'none'

    # Evidence spans
    evidence: dict = field(default_factory=dict)


@dataclass
class QuestionBlockOutput:
    """Complete output for a question-response block."""
    participant_id: int
    phq8_binary: int
    phq8_score: int
    question_key: Optional[str]
    ellie_question: str
    participant_response: str
    word_count: int

    # LLM scores
    scores: LLMRubricScores = field(default_factory=LLMRubricScores)

    # Metadata
    has_laughter: bool = False
    has_sigh: bool = False
    llm_raw_response: Optional[str] = None
    scoring_error: Optional[str] = None


# =============================================================================
# TRANSCRIPT PARSING
# =============================================================================

def parse_transcript(filepath: Path) -> list[dict]:
    """Parse DAIC-WOZ transcript CSV into list of turns."""
    turns = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            turns.append({
                'start_time': float(row['start_time']),
                'stop_time': float(row['stop_time']),
                'speaker': row['speaker'].strip(),
                'value': row['value'].strip()
            })
    return turns


def segment_question_blocks(turns: list[dict]) -> list[dict]:
    """
    Segment transcript into question-response blocks.
    A block = one Ellie question + all subsequent Participant turns until next Ellie turn.
    """
    blocks = []
    current_block = None

    for turn in turns:
        speaker = turn['speaker']
        value = turn['value']

        if speaker == 'Ellie':
            if current_block is not None and current_block['participant_turns']:
                blocks.append(current_block)
            current_block = {
                'ellie_question': value,
                'participant_turns': []
            }
        elif speaker == 'Participant' and current_block is not None:
            current_block['participant_turns'].append(value)

    if current_block is not None and current_block['participant_turns']:
        blocks.append(current_block)

    return blocks


def count_words(text: str) -> int:
    """Count words excluding annotations like <laughter>."""
    clean = re.sub(r'<[^>]+>', '', text)
    return len(clean.split())


def detect_annotations(text: str) -> dict:
    """Detect non-verbal annotations in text."""
    text_lower = text.lower()
    return {
        'has_laughter': '<laughter>' in text_lower,
        'has_sigh': '<sigh>' in text_lower,
        'has_deep_breath': '<deep breath>' in text_lower,
        'has_sniffle': '<sniffle>' in text_lower,
    }


# =============================================================================
# LLM PROMPT CONSTRUCTION
# =============================================================================

LLM_PROMPT_TEMPLATE = """You are analyzing a clinical interview response for linguistic features.
The participant was asked: "{question}"

Their response was: "{response}"

Rate the following dimensions using ONLY integers from 1 to 5.

## DIMENSIONS

1. DETAIL_RICHNESS (amount of specific, concrete information)
   1 = Minimal (single words/fragments: "Good." "Fine." "Yeah.")
   2 = Brief (one simple statement: "I like movies.")
   3 = Moderate (some description: "I like watching movies, usually comedies.")
   4 = Good (specific examples: "I enjoy comedies, especially on weekends.")
   5 = Rich (anecdotes, emotions, specifics: detailed stories with context)

2. NARRATIVE_COHERENCE (organization and followability)
   1 = Fragmented ("Friends... I don't know. He's... tall. Whatever.")
   2 = Jumpy (disconnected statements)
   3 = Basic (understandable but simple)
   4 = Well-organized (clear temporal/causal connections)
   5 = Highly coherent (clear structure, logical flow)

3. SOCIAL_CONNECTEDNESS (warmth and depth in describing relationships)
   1 = Isolated/Dismissive ("I don't really have friends.")
   2 = Minimal/Distant ("I have some friends.")
   3 = Neutral ("My friend Jake is cool.")
   4 = Warm ("Jake is one of my closest friends.")
   5 = Rich emotional investment (deep descriptions of bonds)

4. ENGAGEMENT_LEVEL (willingness to participate vs avoidance)
   1 = Avoidant ("I don't know." "Not really." "That's it.")
   2 = Passive (brief, no elaboration despite opportunity)
   3 = Adequate (responds appropriately)
   4 = Engaged (willing to elaborate)
   5 = Highly engaged (extends discussion, shows genuine interest)

5. EMOTIONAL_VALENCE (overall affective tone)
   1 = Pervasive negative (sadness, hopelessness)
   2 = Mostly negative
   3 = Mixed/Neutral
   4 = Mostly positive
   5 = Predominantly positive (joy, enthusiasm)

6. EPISODIC_SPECIFICITY (specific episodes vs generic statements)
   1 = Entirely generic ("I like things." "People are okay.")
   2 = Mostly generic with vague hints
   3 = Some specific references but undeveloped
   4 = Clear specific episode mentioned
   5 = Vivid, detailed specific episode with context

7. DISENGAGEMENT_AVOIDANCE (conversation-closing behavior)
   1 = Highly disengaged (multiple closures: "I don't know, that's it, whatever")
   2 = Notable avoidance (quick to end topics)
   3 = Neutral (neither avoiding nor extending)
   4 = Engaged (willing to continue)
   5 = Actively extends conversation

8. INTERPERSONAL_WARMTH (warmth specifically in relationship descriptions)
   1 = Cold/Dismissive about others
   2 = Distant or neutral about others
   3 = Matter-of-fact about relationships
   4 = Some warmth evident
   5 = Clear affection and emotional connection

9. LAUGHTER_CONTEXT (if <laughter> appears in the response)
   - "positive": laughter with positive content (joy, humor)
   - "negative": laughter with negative content (nervous, defensive)
   - "avoidance": laughter with avoidance/uncertainty ("I don't know <laughter>")
   - "neutral": laughter with neutral content
   - "none": no laughter annotation present

Output ONLY valid JSON in this exact format:
{{
  "detail_richness": {{"score": <int 1-5>, "evidence": "<exact quote>"}},
  "narrative_coherence": {{"score": <int 1-5>, "evidence": "<exact quote>"}},
  "social_connectedness": {{"score": <int 1-5>, "evidence": "<exact quote>"}},
  "engagement_level": {{"score": <int 1-5>, "evidence": "<exact quote>"}},
  "emotional_valence": {{"score": <int 1-5>, "evidence": "<exact quote>"}},
  "episodic_specificity": {{"score": <int 1-5>, "evidence": "<exact quote>"}},
  "disengagement_avoidance": {{"score": <int 1-5>, "evidence": "<exact quote>"}},
  "interpersonal_warmth": {{"score": <int 1-5>, "evidence": "<exact quote>"}},
  "laughter_context": "<positive|negative|avoidance|neutral|none>"
}}"""


def create_llm_prompt(question: str, response: str) -> str:
    """Create the LLM prompt for a question-response block."""
    return LLM_PROMPT_TEMPLATE.format(question=question, response=response)


# =============================================================================
# LLM API CONFIGURATION
# =============================================================================

# Naver AI Gateway (OpenAI-compatible endpoint)
OPENAI_URL = "https://namc-aigw.io.naver.com/chat/completions"
OPENAI_API_KEY = "sk-stVy1tWqoEcp6s37lvQ0xg"
OPENAI_MODEL = "gpt-oss-120b"


def call_llm_api(prompt: str, max_retries: int = 5, base_delay: float = 2.0) -> dict:
    """
    Call LLM API to score rubric features with exponential backoff retry.
    Uses Naver AI Gateway (OpenAI-compatible chat/completions endpoint).

    Args:
        prompt: The prompt to send to the LLM
        max_retries: Maximum number of retry attempts for rate limiting (default: 5)
        base_delay: Base delay in seconds for exponential backoff (default: 2.0)

    Returns:
        dict with rubric scores and evidence
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Please set it via: export OPENAI_API_KEY='your-key'"
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "max_tokens": 2000,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful research assistant analyzing clinical interview responses. "
                    "Return only valid JSON matching the requested schema. "
                    "Be precise with evidence quotes - use exact text from the response."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                OPENAI_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )

            # Handle rate limiting with exponential backoff
            if response.status_code == 429:
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"    Rate limited (429). Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    continue
                else:
                    raise ValueError(f"Rate limit exceeded after {max_retries} retries. Last response: {response.text}")

            response.raise_for_status()
            data = response.json()

            # Handle different response structures
            content = extract_content_from_response(data)
            return parse_llm_response(content)

        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"    Request error: {e}. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                continue
            raise ValueError(f"Request failed after {max_retries} retries: {e}")

    raise ValueError(f"Unexpected exit from retry loop. Last exception: {last_exception}")


def extract_content_from_response(data: dict) -> str:
    """
    Extract text content from API response, handling various response structures.

    OpenAI-compatible APIs may return content in different formats:
    - String: "content": "..."
    - List of blocks: "content": [{"type": "text", "text": "..."}]
    - Refusal: "refusal": "..."
    """
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected API response structure (no choices/message): {json.dumps(data, ensure_ascii=False)[:500]}")

    # Check for refusal
    if message.get("refusal"):
        raise ValueError(f"API refused to respond: {message['refusal']}")

    content = message.get("content")

    if content is None:
        raise ValueError(f"No content in API response. Full message: {json.dumps(message, ensure_ascii=False)[:500]}")

    # Handle string content (most common)
    if isinstance(content, str):
        return content

    # Handle list content (OpenAI vision/structured format)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif "text" in block:
                    text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        if text_parts:
            return "\n".join(text_parts)
        raise ValueError(f"Content is a list but no text found: {json.dumps(content, ensure_ascii=False)[:500]}")

    # Handle dict content (rare but possible)
    if isinstance(content, dict):
        if "text" in content:
            return content["text"]
        raise ValueError(f"Content is a dict but no text field: {json.dumps(content, ensure_ascii=False)[:500]}")

    raise ValueError(f"Unexpected content type {type(content).__name__}: {str(content)[:500]}")


def parse_llm_response(raw_response: str) -> dict:
    """Parse LLM JSON response with error handling."""
    if not raw_response or not raw_response.strip():
        raise ValueError("Empty response from LLM")

    try:
        # Try to extract JSON from response (handles markdown code blocks too)
        # First try: look for JSON object
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}. Response was: {raw_response[:500]}")


def score_block_with_llm(
    participant_id: int,
    phq8_binary: int,
    phq8_score: int,
    block: dict
) -> QuestionBlockOutput:
    """Score a question-response block using LLM."""

    participant_text = ' '.join(block['participant_turns'])
    annotations = detect_annotations(participant_text)

    output = QuestionBlockOutput(
        participant_id=participant_id,
        phq8_binary=phq8_binary,
        phq8_score=phq8_score,
        question_key=match_target_question(block['ellie_question']),
        ellie_question=block['ellie_question'],
        participant_response=participant_text,
        word_count=count_words(participant_text),
        has_laughter=annotations['has_laughter'],
        has_sigh=annotations['has_sigh']
    )

    # Create prompt and call LLM
    prompt = create_llm_prompt(block['ellie_question'], participant_text)

    try:
        result = call_llm_api(prompt)

        def safe_get_score(key: str) -> Optional[int]:
            """Safely extract score from nested dict."""
            item = result.get(key)
            if isinstance(item, dict):
                return item.get('score')
            return None

        def safe_get_evidence(key: str) -> str:
            """Safely extract evidence from nested dict."""
            item = result.get(key)
            if isinstance(item, dict):
                return item.get('evidence', '')
            return ''

        # Extract scores
        output.scores.detail_richness = safe_get_score('detail_richness')
        output.scores.narrative_coherence = safe_get_score('narrative_coherence')
        output.scores.social_connectedness = safe_get_score('social_connectedness')
        output.scores.engagement_level = safe_get_score('engagement_level')
        output.scores.emotional_valence = safe_get_score('emotional_valence')
        output.scores.episodic_specificity = safe_get_score('episodic_specificity')
        output.scores.disengagement_avoidance = safe_get_score('disengagement_avoidance')
        output.scores.interpersonal_warmth = safe_get_score('interpersonal_warmth')
        output.scores.laughter_context = result.get('laughter_context', 'none')

        # Extract evidence
        output.scores.evidence = {
            'detail_richness': safe_get_evidence('detail_richness'),
            'narrative_coherence': safe_get_evidence('narrative_coherence'),
            'social_connectedness': safe_get_evidence('social_connectedness'),
            'engagement_level': safe_get_evidence('engagement_level'),
            'emotional_valence': safe_get_evidence('emotional_valence'),
            'episodic_specificity': safe_get_evidence('episodic_specificity'),
            'disengagement_avoidance': safe_get_evidence('disengagement_avoidance'),
            'interpersonal_warmth': safe_get_evidence('interpersonal_warmth'),
        }

        output.llm_raw_response = json.dumps(result)

    except ValueError as e:
        output.scoring_error = f"ValueError: {e}"
    except requests.exceptions.HTTPError as e:
        output.scoring_error = f"HTTPError: {e.response.status_code} - {e.response.text[:300] if e.response else str(e)}"
    except requests.exceptions.RequestException as e:
        output.scoring_error = f"RequestError: {type(e).__name__}: {e}"
    except Exception as e:
        output.scoring_error = f"UnexpectedError: {type(e).__name__}: {e}"

    return output


# =============================================================================
# LABEL LOADING
# =============================================================================

def load_labels(labels_dir: Path) -> dict[int, dict]:
    """Load PHQ-8 labels from DAIC-WOZ label files."""
    labels = {}

    for split in ['train', 'dev', 'test']:
        filepath = labels_dir / f'{split}_split_Depression_AVEC2017.csv'
        if filepath.exists():
            with open(filepath, 'r', newline='', encoding='utf-8') as f:
                content = f.read()
                if content.startswith('\ufeff'):
                    content = content[1:]
                lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
                reader = csv.DictReader(lines)
                for row in reader:
                    if not row or 'Participant_ID' not in row:
                        continue
                    try:
                        pid = int(row['Participant_ID'])
                        labels[pid] = {
                            'phq8_binary': int(row['PHQ8_Binary']),
                            'phq8_score': int(row['PHQ8_Score']),
                            'split': split
                        }
                    except (ValueError, KeyError):
                        continue

    return labels


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

CSV_FIELDNAMES = [
    'participant_id', 'phq8_binary', 'phq8_score',
    'question_key', 'ellie_question', 'participant_response', 'word_count',
    'detail_richness', 'narrative_coherence', 'social_connectedness',
    'engagement_level', 'emotional_valence', 'episodic_specificity',
    'disengagement_avoidance', 'interpersonal_warmth', 'laughter_context',
    'evidence_detail_richness', 'evidence_narrative_coherence',
    'evidence_social_connectedness', 'evidence_engagement_level',
    'evidence_emotional_valence', 'evidence_episodic_specificity',
    'evidence_disengagement_avoidance', 'evidence_interpersonal_warmth',
    'has_laughter', 'has_sigh', 'scoring_error'
]


def output_to_row(output: 'QuestionBlockOutput') -> dict:
    """Convert QuestionBlockOutput to CSV row dict."""
    return {
        'participant_id': output.participant_id,
        'phq8_binary': output.phq8_binary,
        'phq8_score': output.phq8_score,
        'question_key': output.question_key,
        'ellie_question': output.ellie_question,
        'participant_response': output.participant_response,
        'word_count': output.word_count,
        'detail_richness': output.scores.detail_richness,
        'narrative_coherence': output.scores.narrative_coherence,
        'social_connectedness': output.scores.social_connectedness,
        'engagement_level': output.scores.engagement_level,
        'emotional_valence': output.scores.emotional_valence,
        'episodic_specificity': output.scores.episodic_specificity,
        'disengagement_avoidance': output.scores.disengagement_avoidance,
        'interpersonal_warmth': output.scores.interpersonal_warmth,
        'laughter_context': output.scores.laughter_context,
        'evidence_detail_richness': output.scores.evidence.get('detail_richness', ''),
        'evidence_narrative_coherence': output.scores.evidence.get('narrative_coherence', ''),
        'evidence_social_connectedness': output.scores.evidence.get('social_connectedness', ''),
        'evidence_engagement_level': output.scores.evidence.get('engagement_level', ''),
        'evidence_emotional_valence': output.scores.evidence.get('emotional_valence', ''),
        'evidence_episodic_specificity': output.scores.evidence.get('episodic_specificity', ''),
        'evidence_disengagement_avoidance': output.scores.evidence.get('disengagement_avoidance', ''),
        'evidence_interpersonal_warmth': output.scores.evidence.get('interpersonal_warmth', ''),
        'has_laughter': output.has_laughter,
        'has_sigh': output.has_sigh,
        'scoring_error': output.scoring_error,
    }


def load_existing_results(output_path: Path) -> set[tuple[int, str]]:
    """
    Load already-processed (participant_id, question_key) pairs from existing CSV.
    This allows resuming interrupted runs.
    """
    processed = set()
    if output_path.exists():
        with open(output_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pid = int(row['participant_id'])
                    qkey = row['question_key']
                    processed.add((pid, qkey))
                except (ValueError, KeyError):
                    continue
    return processed


def extract_features(
    transcripts_dir: Path,
    labels_dir: Path,
    output_path: Path,
    target_question_keys: Optional[list] = None,
    request_delay: float = 0.5
) -> int:
    """
    Run full LLM feature extraction on all transcripts with incremental saving.

    Each block is appended to the CSV immediately after scoring, so progress
    is preserved even if the script is interrupted.

    Args:
        transcripts_dir: Path to transcript CSV files
        labels_dir: Path to label CSV files
        output_path: Where to save output CSV
        target_question_keys: List of question keys to process (None = all TARGET_QUESTIONS)
        request_delay: Delay between API requests in seconds (default: 0.5)

    Returns:
        Number of blocks processed
    """
    labels = load_labels(labels_dir)
    print(f"Loaded labels for {len(labels)} participants")

    # Load existing results to allow resuming
    processed = load_existing_results(output_path)
    if processed:
        print(f"Found {len(processed)} already-processed blocks, will skip these")

    # Prepare CSV file (create with header if new, otherwise append mode)
    file_exists = output_path.exists() and len(processed) > 0
    csv_file = open(output_path, 'a' if file_exists else 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    transcript_files = sorted(transcripts_dir.glob('*_TRANSCRIPT.csv'))
    print(f"Found {len(transcript_files)} transcript files")

    # Use all target questions if not specified
    if target_question_keys is None:
        target_question_keys = list(TARGET_QUESTIONS.keys())

    total_processed = 0
    total_errors = 0
    total_skipped = len(processed)

    try:
        for filepath in transcript_files:
            participant_id = int(filepath.stem.split('_')[0])

            if participant_id not in labels:
                print(f"Warning: No label for participant {participant_id}, skipping")
                continue

            label_info = labels[participant_id]
            turns = parse_transcript(filepath)
            blocks = segment_question_blocks(turns)

            print(f"Processing participant {participant_id} ({label_info['split']}, PHQ8={label_info['phq8_score']})...")

            for block in blocks:
                question_key = match_target_question(block['ellie_question'])

                # Only process target questions
                if question_key is None:
                    continue

                if target_question_keys and question_key not in target_question_keys:
                    continue

                # Skip if already processed
                if (participant_id, question_key) in processed:
                    continue

                print(f"  Scoring {participant_id} - {question_key}...", end=" ", flush=True)

                output = score_block_with_llm(
                    participant_id=participant_id,
                    phq8_binary=label_info['phq8_binary'],
                    phq8_score=label_info['phq8_score'],
                    block=block
                )

                # Write immediately to CSV
                writer.writerow(output_to_row(output))
                csv_file.flush()  # Ensure it's written to disk

                total_processed += 1
                if output.scoring_error:
                    total_errors += 1
                    print(f"ERROR: {output.scoring_error[:60]}...")
                else:
                    print("OK")

                # Add delay between requests to avoid rate limiting
                time.sleep(request_delay)

    finally:
        csv_file.close()

    print()
    print("=" * 60)
    print(f"Extraction complete.")
    print(f"  Total processed this run: {total_processed}")
    print(f"  Errors this run: {total_errors}")
    print(f"  Previously processed (skipped): {total_skipped}")
    print(f"  Total in output file: {total_processed + total_skipped}")
    print(f"  Output saved to: {output_path}")

    return total_processed


def save_outputs_to_csv(outputs: list[QuestionBlockOutput], output_path: Path):
    """Save QuestionBlockOutput list to CSV (batch mode, overwrites file)."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for output in outputs:
            writer.writerow(output_to_row(output))


# =============================================================================
# RELIABILITY CHECK (for pilot validation)
# =============================================================================

def check_reliability(
    transcript_path: Path,
    labels: dict,
    block_index: int,
    n_runs: int = 5
) -> dict:
    """
    Run the same block through LLM multiple times to check consistency.

    Returns:
        dict with scores from each run and ICC estimate
    """
    participant_id = int(transcript_path.stem.split('_')[0])
    label_info = labels[participant_id]

    turns = parse_transcript(transcript_path)
    blocks = segment_question_blocks(turns)

    if block_index >= len(blocks):
        raise ValueError(f"Block index {block_index} out of range")

    block = blocks[block_index]
    results = []

    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs}...")
        output = score_block_with_llm(
            participant_id=participant_id,
            phq8_binary=label_info['phq8_binary'],
            phq8_score=label_info['phq8_score'],
            block=block
        )
        results.append(output)

    # Compute simple agreement metrics
    dimensions = [
        'detail_richness', 'narrative_coherence', 'social_connectedness',
        'engagement_level', 'emotional_valence', 'episodic_specificity',
        'disengagement_avoidance', 'interpersonal_warmth'
    ]

    reliability = {}
    for dim in dimensions:
        scores = [getattr(r.scores, dim) for r in results if getattr(r.scores, dim) is not None]
        if len(scores) >= 2:
            reliability[dim] = {
                'scores': scores,
                'mean': statistics.mean(scores),
                'stdev': statistics.stdev(scores) if len(scores) > 1 else 0,
                'range': max(scores) - min(scores),
                'all_same': len(set(scores)) == 1
            }

    return {
        'question': block['ellie_question'],
        'response': ' '.join(block['participant_turns']),
        'reliability': reliability,
        'n_runs': n_runs
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Paths
    BASE_DIR = Path('/Users/user/Desktop/DAIC-WOZ')
    TRANSCRIPTS_DIR = BASE_DIR / 'transcripts'
    LABELS_DIR = BASE_DIR / 'labels'
    OUTPUT_DIR = Path('/Users/user/Desktop/DAIC-WOZ-Experiments/02_real_pilot_rubric_features')
    OUTPUT_PATH = OUTPUT_DIR / 'full_llm_features.csv'

    # Full extraction configuration - all 20 target questions
    TARGET_QUESTION_KEYS = list(TARGET_QUESTIONS.keys())

    print("=" * 60)
    print("DAIC-WOZ LLM Rubric Feature Extraction - Full Experiment")
    print("=" * 60)
    print()
    print("CONFIGURATION:")
    print(f"  Target questions: {len(TARGET_QUESTION_KEYS)} questions")
    print(f"  API: Naver AI Gateway (OpenAI-compatible)")
    print(f"  Model: {OPENAI_MODEL}")
    print(f"  API Key configured: {'Yes' if OPENAI_API_KEY else 'No - set OPENAI_API_KEY env var'}")
    print(f"  Output: {OUTPUT_PATH}")
    print()
    print("FEATURES:")
    print("  - Exponential backoff retry on rate limiting (429)")
    print("  - Incremental save (each block saved immediately)")
    print("  - Resume support (skips already-processed blocks)")
    print()

    # Run full extraction
    try:
        n_processed = extract_features(
            transcripts_dir=TRANSCRIPTS_DIR,
            labels_dir=LABELS_DIR,
            output_path=OUTPUT_PATH,
            target_question_keys=TARGET_QUESTION_KEYS,
            request_delay=0.5  # 500ms between requests to avoid rate limiting
        )

    except ValueError as e:
        print(f"\nConfiguration error: {e}")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved to CSV.")
        print("Re-run the script to resume from where you left off.")
