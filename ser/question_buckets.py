#!/usr/bin/env python3
"""Group the 161 Ellie question ids into 4 buckets, for question-aligned SER pooling.

Chosen 2026-09-14: 4 buckets plus contrasts against the neutral one. The point of the
neutral bucket is to cancel each speaker's own baseline -- individual differences in
voice and habitual affect are large, and averaging a whole session leaves them in,
which is the most likely reason the speaker-mean embedding failed.

FOLLOWUP ids ("can you tell me about that", "why") carry no topic of their own; turns
under them inherit the bucket of the question before. BACKCHANNEL ids are Ellie's
acknowledgements, which do not change the topic either -- several of these are id-tagged
in the transcript, so they cannot be filtered by text alone.

Assignments marked AMBIGUOUS below are judgment calls, listed so they can be overridden.
"""

SYMPTOM = """
feel_lately easy_sleep sleep_affects behavior_changes control_temper mad_makeyou
what_do_when_annoyed depression_diagnosed ptsd_diagnosed when_diagnosed symptoms_what
symptoms_cope bouts_symptoms suspect_problem feel_down disturbing_thoughts why_seek_help
therapy_changes therapy_going therapy_useful therapist_affect therapist_useful how_know
feelguilty feelbadly trigger avoid stop_going uncomfortable_talkv effectb
""".split()

NEGATIVE = """
last_argument argument_about situation_handled hard_decision hard_decisionb difficult
too_hard been_hard regret memory_erase self_change landed_trouble change_directions
combat advice_back
""".split()

POSITIVE = """
happy_lasttime memorableb influence_positive bf_describe family_relationship family_roleb
parent_best relax_fishtank do_fun ideal_weekendc happy_didthat dream_job shyoutgoing
how_close still_doing_x still_working_on_x
ellie17dec2012_07 ellie17dec2012_08 ellie17dec2012_09 ellie17dec2012_10
""".split()

NEUTRAL = """
where_originally like_about_la dont_like_la why_la when_la from_la2 compares_la adapted_la
where_live living_situation roommates study travel_trips travel_changed travel_shoes
often_backb job_virtually how_doingv todays_kids kids_elaborate easy_parent parent_hardest
parent_differences very_young long_time military why_enlist civilian_life old trade_offs
""".split()

FOLLOWUP = """
tell_about_that why2 more mind like_what give_example elaborate tell_me_about
tell_me_morev2 describe_felt see_mean after
""".split()

BACKCHANNEL = """
mhm mm yeah2 yeah3 yes right2 uh_huh aw awb wow oh oh_no oh_my_gosh uh_oh im_sorry
sorry_hear that_sucks understand makes_sense nice good_job thats_great thats_good awesome
good_hear cool3 hmm1 hmmb hmm_downer isee_downer yeah_downer downer me_too really
sounds_interesting great_situation great_thanks thank_you love_hear talk_later back_later
asked_everything bye okay_confirm introv4confirmation appreciate_open name_ellie
wild_laughter3 wild_laughter5 wild_laughter7big
ellie17dec2012_02 ellie17dec2012_03 ellie17dec2012_04 ellie17dec2012_06
""".split()

# Judgment calls worth a second look -- each is defensible either way.
AMBIGUOUS = {
    "advice_back": "NEGATIVE; 'advice to your younger self' is reflective, often regretful, "
                   "but plenty of answers are warm -- could be POSITIVE or its own bucket",
    "combat": "NEGATIVE; combat experience, but for some veterans it is told neutrally",
    "still_working_on_x": "POSITIVE; 'still working on X' can equally be a struggle",
    "dream_job": "POSITIVE; aspirational, but arguably just biographical (NEUTRAL)",
    "shyoutgoing": "POSITIVE; a trait question, not really valenced",
    "uncomfortable_talkv": "SYMPTOM; about the interview itself rather than the person",
    "trade_offs": "NEUTRAL; military trade-offs, borderline NEGATIVE",
    "old": "NEUTRAL; ambiguous id, likely an age question",
}

BUCKETS = {"symptom": SYMPTOM, "negative": NEGATIVE,
           "positive": POSITIVE, "neutral": NEUTRAL}
QID2BUCKET = {q: b for b, ids in BUCKETS.items() for q in ids}
FOLLOWUP = set(FOLLOWUP)
BACKCHANNEL = set(BACKCHANNEL)


def bucket_of(qid):
    """Returns a bucket name, or 'followup'/'backchannel'/'other' for the special cases."""
    if qid in BACKCHANNEL:
        return "backchannel"
    if qid in FOLLOWUP:
        return "followup"
    return QID2BUCKET.get(qid, "other")
