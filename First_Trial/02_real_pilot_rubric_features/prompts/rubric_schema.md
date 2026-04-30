# LLM Rubric Schema for Depression-Associated Feature Extraction

## Overview

This document defines the structured rubric for LLM-based feature scoring.

The LLM is used as a **feature extractor**, not a **depression detector**.
Each dimension captures a linguistic/discourse property that may be associated with depression.

---

## Why Pilot Test?

The pilot test serves several critical purposes:

1. **Validate variance**: Confirm LLM scoring produces meaningful variance across transcripts
2. **Check reliability**: Run same response 3-5 times to measure intra-LLM consistency (ICC)
3. **Verify evidence spans**: Ensure extracted quotes are meaningful and traceable
4. **Identify differentiating dimensions**: Which rubric dimensions actually separate PHQ8 groups?
5. **Refine anchors**: Adjust anchor examples before full-scale extraction
6. **Filter weak features**: Drop dimensions that show no discrimination or poor reliability

---

## Rubric Dimensions (8 numeric + 1 categorical)

### Core Discourse Features (Original 5)

#### 1. DETAIL_RICHNESS
**What it measures**: Amount of specific, concrete information in the response.

**Why it matters**: Depression is associated with reduced elaboration and autobiographical specificity (overgeneralized memory). Participants may give "bare minimum" responses.

| Score | Description | Example |
|-------|-------------|---------|
| 1 | Minimal | "Good." / "Fine." / "Yeah." |
| 2 | Brief | "I like movies." |
| 3 | Moderate | "I like watching movies, usually comedies." |
| 4 | Good | "I enjoy watching comedies, especially on weekends. Last week I watched this really funny film." |
| 5 | Rich | "I love watching comedies—there's something about laughing after a long week that just resets everything. Last Saturday, my roommate and I watched this old Steve Martin movie and couldn't stop laughing." |

---

#### 2. NARRATIVE_COHERENCE
**What it measures**: How well-organized and followable the response is.

**Why it matters**: Cognitive symptoms of depression (concentration difficulties, slowed thinking) may manifest as fragmented or disorganized speech.

| Score | Description | Example |
|-------|-------------|---------|
| 1 | Fragmented | "Friends... I don't know. He's... tall. Whatever." |
| 2 | Jumpy | "I have a friend. He likes sports. We met somewhere. It's fine." |
| 3 | Basic | "I have a friend named Tom. We hang out sometimes. He's a nice guy." |
| 4 | Well-organized | "I've known Tom since college. We met in a chemistry class and started studying together. Now we grab coffee every few weeks." |
| 5 | Highly coherent | "Tom and I go way back—we met freshman year in this impossible chemistry class. I remember we both failed the first quiz and ended up forming a study group. That's when I realized how funny he was. Now, even though we're in different cities, we still video call every Sunday." |

---

#### 3. SOCIAL_CONNECTEDNESS
**What it measures**: Warmth, depth, and emotional investment when describing relationships.

**Why it matters**: Social withdrawal and perceived isolation are core features of depression. Responses may show emotional distance or dismissiveness.

| Score | Description | Example |
|-------|-------------|---------|
| 1 | Isolated/Dismissive | "I don't really have friends." / "My family? We don't talk." |
| 2 | Minimal/Distant | "I have some friends. We hang out occasionally." |
| 3 | Neutral | "My friend Jake is pretty cool. We play video games together." |
| 4 | Warm | "Jake is one of my closest friends. He's always there when I need to talk." |
| 5 | Rich emotional investment | "Jake means the world to me. When I went through that rough patch last year, he drove three hours just to check on me. That's the kind of friend he is." |

---

#### 4. ENGAGEMENT_LEVEL
**What it measures**: Willingness to participate in the conversation vs. avoidance/disengagement.

**Why it matters**: Psychomotor retardation, anhedonia, and fatigue in depression can manifest as reluctance to engage or minimal-effort responses.

| Score | Description | Example |
|-------|-------------|---------|
| 1 | Avoidant | "I don't know." / "Not really." / "That's it." |
| 2 | Passive | "I guess I like music." (no elaboration despite opportunity) |
| 3 | Adequate | "I like listening to music. Mostly rock." |
| 4 | Engaged | "Music is a big part of my life. I play guitar and I've been getting into jazz lately." |
| 5 | Highly engaged | "Oh, I love music! I've been playing guitar for ten years now. Recently I've been exploring jazz—there's this John Coltrane album that completely changed how I think about improvisation." |

---

#### 5. EMOTIONAL_VALENCE
**What it measures**: Overall affective tone of the response (negative to positive).

**Why it matters**: Persistent negative affect is a defining feature of depression. Responses may be colored by sadness, hopelessness, or frustration even when discussing neutral topics.

| Score | Description | Example |
|-------|-------------|---------|
| 1 | Pervasive negative | "Everything's been pretty awful. I don't see the point anymore." |
| 2 | Mostly negative | "Work's been stressful. I'm tired all the time. But I guess it's fine." |
| 3 | Mixed/Neutral | "Some days are good, some days are bad. Pretty average, I guess." |
| 4 | Mostly positive | "Things are going well. Work's busy but I enjoy it." |
| 5 | Predominantly positive | "I've been feeling great! Just got back from a trip with friends and I'm so energized." |

---

### Extended Features (Added Based on Pilot Analysis)

#### 6. EPISODIC_SPECIFICITY
**What it measures**: Whether the response contains specific episodes vs. generic/abstract statements.

**Why it matters**: Depression is associated with "overgeneral autobiographical memory" - difficulty recalling specific events. This is distinct from detail_richness (which measures amount) - episodic_specificity measures whether content is tied to a specific time/place.

| Score | Description | Example |
|-------|-------------|---------|
| 1 | Entirely generic | "I like things." / "People are okay." / "I do stuff." |
| 2 | Mostly generic | "I usually hang out with friends." (no specific instance) |
| 3 | Some specific references | "I went to a concert once." (mentioned but undeveloped) |
| 4 | Clear specific episode | "Last month I went to a Taylor Swift concert with my sister." |
| 5 | Vivid, detailed episode | "Last month I went to the Taylor Swift concert with my sister—we'd been planning it for six months. When she played our favorite song, we both started crying. I'll never forget that moment." |

---

#### 7. DISENGAGEMENT_AVOIDANCE
**What it measures**: Conversation-closing behavior and reluctance to continue.

**Why it matters**: This captures active avoidance patterns (e.g., repeated "I don't know", quick topic closures) that may indicate withdrawal or low energy. Different from low engagement_level—this specifically tracks closure markers.

| Score | Description | Example |
|-------|-------------|---------|
| 1 | Highly disengaged | Multiple closures: "I don't know, that's it, whatever" |
| 2 | Notable avoidance | Quick to end topics: "Not much to say about that." |
| 3 | Neutral | Neither avoiding nor extending |
| 4 | Willing to continue | Answers fully, open to follow-up |
| 5 | Actively extends | Provides additional context, asks questions back |

---

#### 8. INTERPERSONAL_WARMTH
**What it measures**: Warmth and affection specifically in relationship descriptions.

**Why it matters**: This is narrower than social_connectedness. A person might mention many relationships (high social_connectedness) but describe them coldly. Interpersonal_warmth captures the affective quality.

| Score | Description | Example |
|-------|-------------|---------|
| 1 | Cold/Dismissive | "My mom? She's around, I guess." |
| 2 | Distant/Neutral | "My mom lives nearby. We see each other sometimes." |
| 3 | Matter-of-fact | "My mom is nice. She helps out when she can." |
| 4 | Some warmth evident | "My mom is great—she always knows when I need a call." |
| 5 | Clear affection | "I love my mom so much. She's my rock. Every Sunday we have dinner together and catch up on everything." |

---

### Annotation-Aware Feature (Categorical)

#### 9. LAUGHTER_CONTEXT
**What it measures**: The emotional context surrounding `<laughter>` annotations.

**Why it matters**: Laughter can indicate genuine positive affect OR nervous/defensive behavior. Laughter following avoidance statements ("I don't know <laughter>") may signal discomfort, while laughter with positive content may signal genuine joy.

| Value | Description | Example |
|-------|-------------|---------|
| `positive` | Laughter with positive content | "I love camping <laughter> it's so fun" |
| `negative` | Laughter with negative content | "not get so pissed off <laughter>" |
| `avoidance` | Laughter with uncertainty/avoidance | "i don't know <laughter>" |
| `neutral` | Laughter with neutral content | "the weather is nice <laughter>" |
| `none` | No laughter annotation present | (no `<laughter>` tag in text) |

---

## Prompt Template

```text
You are analyzing a clinical interview response for linguistic features.
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
{
  "detail_richness": {"score": <int 1-5>, "evidence": "<exact quote>"},
  "narrative_coherence": {"score": <int 1-5>, "evidence": "<exact quote>"},
  "social_connectedness": {"score": <int 1-5>, "evidence": "<exact quote>"},
  "engagement_level": {"score": <int 1-5>, "evidence": "<exact quote>"},
  "emotional_valence": {"score": <int 1-5>, "evidence": "<exact quote>"},
  "episodic_specificity": {"score": <int 1-5>, "evidence": "<exact quote>"},
  "disengagement_avoidance": {"score": <int 1-5>, "evidence": "<exact quote>"},
  "interpersonal_warmth": {"score": <int 1-5>, "evidence": "<exact quote>"},
  "laughter_context": "<positive|negative|avoidance|neutral|none>"
}
```

---

## Output Schema

The LLM must return valid JSON matching this schema:

```json
{
  "detail_richness": {
    "score": 3,
    "evidence": "I like watching movies, usually comedies."
  },
  "narrative_coherence": {
    "score": 4,
    "evidence": "I've known Tom since college. We met in a chemistry class..."
  },
  "social_connectedness": {
    "score": 2,
    "evidence": "I have some friends. We hang out occasionally."
  },
  "engagement_level": {
    "score": 3,
    "evidence": "I like listening to music. Mostly rock."
  },
  "emotional_valence": {
    "score": 3,
    "evidence": "Some days are good, some days are bad."
  },
  "episodic_specificity": {
    "score": 4,
    "evidence": "Last month I went to a Taylor Swift concert with my sister."
  },
  "disengagement_avoidance": {
    "score": 3,
    "evidence": "(neither avoiding nor extending)"
  },
  "interpersonal_warmth": {
    "score": 4,
    "evidence": "My mom is great—she always knows when I need a call."
  },
  "laughter_context": "positive"
}
```

---

## Reliability Requirements

Before using LLM scores in analysis:

1. **Run each response 3-5 times** at temperature=0
2. **Compute intra-LLM consistency** (ICC or Krippendorff's alpha)
3. **Minimum acceptable ICC**: 0.70 for fair reliability, 0.80+ preferred
4. **If low reliability**: Refine anchor examples or simplify dimension

---

## Target Questions for LLM Scoring

Only apply LLM scoring to open-ended questions where elaboration is expected:

| Question Pattern | Primary Dimensions to Score |
|-----------------|---------------------------|
| "How would you describe yourself?" | detail_richness, narrative_coherence |
| "How would your best friend describe you?" | social_connectedness, interpersonal_warmth, detail_richness |
| "How close are you to your family?" | social_connectedness, interpersonal_warmth, emotional_valence |
| "When was the last time you felt really happy?" | episodic_specificity, detail_richness, emotional_valence |
| "What do you do for fun?" | engagement_level, emotional_valence, episodic_specificity |
| "Is there anything you regret?" | narrative_coherence, emotional_valence, episodic_specificity |

---

## API Configuration

**Placeholder** - Configure with your preferred provider:

### OpenAI GPT-4
```python
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)
result = json.loads(response.choices[0].message.content)
```

### Anthropic Claude
```python
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}]
)
result = json.loads(response.content[0].text)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-06 | Initial rubric design (5 dimensions) |
| 2.0 | 2026-04-06 | Added episodic_specificity, disengagement_avoidance, interpersonal_warmth, laughter_context |

---

*Maintained by Team Olivier (Prof. Claude & Prof. Codex)*
