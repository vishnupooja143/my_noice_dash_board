"""
speaker_lock.py — CUSTOMER VOICE LOCK (v4.4)
  08-Oct: NOISE_REJECT — words STT made out of a steady fan / AC / traffic or a
  beep / dial tone on the caller's turn are now dropped ("other") instead of
  "unknown". See NOISE-ONLY TURNS in the config.
speaker_lock3.py — CUSTOMER VOICE LOCK (v4.3)
  06-Oct (V0061629030000047900): a person beside the caller scored 0.30 while
  the caller's own next words scored 0.20 — no threshold can separate that.
    * RIVAL: a voice that was already dropped as "not the customer" is
      remembered (its own voice print). A later turn that is clearly CLOSER to
      such a remembered voice than to the customer is that person again, even
      when its score against the customer is above the threshold.
      TCVL_RIVAL = shadow (default: only logs what it WOULD drop, behaviour
      unchanged) | on (drops) | off.
speaker_lock3.py — CUSTOMER VOICE LOCK (v4.2)
  06-Oct (V0061536040000047857): the caller answered "Hello." and "తెలుగు." —
  two words — yet 4 s of "speech" were collected, most of it line sound that was
  not words. The profile matched that sound, and the caller's own question
  ("రేట్ ఆఫ్ ఇంట్రెస్ట్ ఎంత?", three times) scored 0.10-0.16 and was dropped.
    * ENROLL_LOUD: only the LOUD part of the answers (the voice at the phone)
      counts towards the 4 s and goes into the profile.
    * ENROLL_MIN_WORDS: the first lock waits until STT has heard this many words
      from the caller (the agent reports them with note_enroll_words()).
    * FAST_RELOCK: a dropped turn that answers the bot and is as loud as the
      locked voice is remembered; two of them, after two different bot lines,
      with no clearly-matching turn in between, move the lock to that voice.
speaker_lock3.py — CUSTOMER VOICE LOCK (v4.1, low-latency)
  01-Oct: probation, two-answer check, short-word scoring (pitch), line activity,
  and latency work: the turn check is pre-computed while LiveKit waits for the
  end of turn (prepare_turn), never blocks the audio thread, judges at most
  TURN_JUDGE_MAX_S of audio; while the bot speaks a decision is made every
  LOCKED_EVAL_EVERY_HOPS hops; torch uses TORCH_THREADS threads per call.
  01-Oct (V0011301020000043983): the profile learns only from turns that clearly
  match ONE voice (BANK_ADD_TURN_SIM), and a turn in which the customer AND
  someone else spoke is detected piece by piece (voice print + pitch) and
  reported in last_turn_info["mixed"] so the agent can tell the LLM.
  03-Oct (V0031539040000045744): vad_should_mute() / customer_active() let the
  agent keep another voice from pausing or interrupting the bot; a hand-over
  re-lock now needs the other voice to answer DIFFERENT bot lines, so someone
  talking on beside the caller can no longer take the lock.
  03-Oct (V0031550420000045410): gray zone — a weak-scoring turn (0.15-0.30)
  that is as loud as the customer, answers the bot within a few seconds and is
  not a clearly different pitch is the customer on a noisy line; it is kept.

speaker_lock.py — CUSTOMER VOICE LOCK (v4.0)
=============================================
Only the customer's voice reaches STT. Everything else becomes digital silence.

PIPELINE
    raw call audio ─► 100 ms hops ─► 300 ms look-ahead delay line ─► decision ─► STT

PHASE 1 — ENROLLING (before lock)
    * Audio passes through untouched (the language answer must still be heard).
    * Customer speech is collected ONLY while the bot is silent, i.e. the voice
      that ANSWERS the bot. That is how "who is the customer" is decided.
    * Once ENROLL_MIN_SECONDS of speech is collected a profile is built in a
      background thread (audio keeps flowing meanwhile).
    * If the enrollment audio contains more than one voice (overlap / someone
      else talking), SepFormer splits every utterance into sources, the
      near-mic (loudest) source is taken as the customer, and one refinement
      pass re-picks per utterance the source closest to that profile.
      Only the separated customer voice goes into the profile.

PHASE 2 — LOCKED
    Every hop gets an ECAPA similarity computed on an 800 ms window CENTRED on
    that hop (300 ms of look-ahead makes this possible), so the decision is
    about the audio actually being released — not audio from 1.5 s earlier,
    which is what v3 did.
        not speech-like (TV music, traffic, fan, silence) ........ silence
        sim >= clean threshold (customer alone) .................. passed raw
        threshold <= sim < clean (customer + something else) ..... TSE: only the
                                                                    customer's
                                                                    voice is kept
        hard_reject <= sim < threshold (overlap / unsure) ........ TSE, else silence
        sim < hard_reject (another person) ....................... silence
    Short fades at every gate edge so the cut itself never produces a click that
    STT could transcribe as a syllable.

This module is pure numpy + optional torch models. It has no LiveKit dependency;
the agent converts hops back to rtc.AudioFrame (see stt_node in the agent).
"""
from __future__ import annotations

import collections
import enum
import logging
import os
import threading
import time
import uuid
import wave
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("speaker_lock_sdk.speaker_lock")


# Every setting of this module is read ONLY under the TCVL_ prefix
# (Tally Capital Voice Lock). The shared .env also serves other projects that use
# names like ENROLL_MIN_SECONDS / MIN_CALLER_P5 / HARD_REJECT_COSINE for their own
# speaker lock — those are ignored here, so the projects can't override each other.
ENV_PREFIX = "TCVL_"


def _env(name: str) -> Optional[str]:
    key = ENV_PREFIX + (name[3:] if name.startswith("SL_") else name)
    return os.getenv(key)


def _f(name: str, default: float) -> float:
    v = _env(name)
    return float(v) if v not in (None, "") else float(default)


def _i(name: str, default: int) -> int:
    v = _env(name)
    return int(float(v)) if v not in (None, "") else int(default)


def _b(name: str, default: bool) -> bool:
    v = _env(name)
    return default if v in (None, "") else v.strip().lower() == "true"


def _s(name: str, default: str) -> str:
    v = _env(name)
    return default if v in (None, "") else v.strip().lower()


# ════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════
HOP_MS              = _i("SL_HOP_MS", 100)          # decision granularity
LOOKAHEAD_HOPS      = _i("SL_LOOKAHEAD_HOPS", 3)    # 300 ms delay → centred decisions
WINDOW_BACK_HOPS    = _i("SL_WINDOW_BACK_HOPS", 8)  # window = 8 + 1 + 3 hops = 1.2 s
# (800 ms was too short for ECAPA on 8 kHz phone audio: the caller's own voice
#  scored anywhere from 0.12 to 0.57 and got cut mid-sentence.)
# Once a stretch of speech has been accepted as the customer, it stays open while
# the similarity stays above this much below the threshold — a word that scores
# badly on its own is not a different person.
KEEP_OPEN_DROP      = _f("SL_KEEP_OPEN_DROP", 0.10)
RUN_GAP_HOPS        = _i("SL_RUN_GAP_HOPS", 3)      # silence that ends a speech run
FADE_MS             = _i("SL_FADE_MS", 8)

# speech-likeness (cheap pre-filter; anything that passes still has to match the voice)
ABS_SPEECH_RMS      = _f("SL_ABS_SPEECH_RMS", 60.0)
SPEECH_OVER_BG      = _f("SL_SPEECH_OVER_BG", 1.3)
SFM_MIN             = _f("SFM_MUSIC_CEILING", 0.06)   # below → tonal (music / beeps)
SFM_MAX             = _f("SFM_NOISE_FLOOR", 0.88)     # above → broadband noise
BG_HISTORY_S        = _f("SL_BG_HISTORY_S", 10.0)
BG_PERCENTILE       = _f("SL_BG_PERCENTILE", 10.0)

# enrollment
ENROLL_MIN_SECONDS  = _f("ENROLL_MIN_SECONDS", 4.0)
ENROLL_RETRY_EXTRA_S = _f("ENROLL_RETRY_EXTRA_S", 1.5)
ENROLL_MAX_ATTEMPTS = _i("ENROLL_MAX_ATTEMPTS", 3)
ENROLL_BUFFER_MAX_S = _f("ENROLL_BUFFER_MAX_S", 15.0)
ENROLL_UTT_GAP_HOPS = _i("ENROLL_UTT_GAP_HOPS", 3)
POST_BOT_GRACE_S    = _f("POST_BOT_GRACE_S", 0.25)
MIN_CALLER_P5       = _f("MIN_CALLER_P5", 0.40)
MIN_CALLER_P5_LAST_TRY = _f("MIN_CALLER_P5_LAST_TRY", 0.28)
ENROLL_OUTLIER_ITERATIONS = _i("ENROLL_OUTLIER_ITERATIONS", 3)
ENROLL_OUTLIER_PERCENTILE = _f("ENROLL_OUTLIER_PERCENTILE", 30.0)
MULTI_SPEAKER_SIM   = _f("ENROLL_MULTI_SPEAKER_SIM", 0.40)
MULTI_SPEAKER_FRACTION = _f("ENROLL_MULTI_SPEAKER_FRACTION", 0.20)
SEPARATOR_PIECE_S   = _f("SEPARATOR_MAX_MS", 6000.0) / 1000.0

# locked decisions
STRANGER_BAND_MAX   = _f("STRANGER_BAND_MAX", 0.30)
HARD_REJECT_COSINE  = _f("HARD_REJECT_COSINE", 0.12)
THRESHOLD_MIN       = _f("SL_THRESHOLD_MIN", 0.25)
THRESHOLD_MAX       = _f("SL_THRESHOLD_MAX", 0.35)
OVERLAP_MARGIN      = _f("SL_OVERLAP_MARGIN", 0.10)
# [fix 25] keep the "unsure" start of a sentence spoken over the bot and release it
# when the sentence is recognised as the customer's
BACKFILL            = _b("BACKFILL", True)
BACKFILL_MAX_HOPS   = _i("BACKFILL_MAX_HOPS", 15)
# What to do with "unsure" hops when TSE is unavailable/failed: silence | pass
UNCERTAIN_POLICY    = _s("SL_UNCERTAIN_POLICY", "silence")

# target speaker extraction after lock (overlap handling)
TSE_WINDOW_HOPS     = _i("SL_TSE_WINDOW_HOPS", 15)   # 1.5 s mixture per TSE run
TSE_REF_SECONDS     = _f("SL_TSE_REF_SECONDS", 4.0)
TSE_ACCEPT_DROP     = _f("SL_TSE_ACCEPT_DROP", 0.05)
TSE_OUTPUT_LEVEL    = _f("SL_TSE_OUTPUT_LEVEL", 0.7)

# slow profile tracking (line/channel drift) — only on very confident customer audio
PROFILE_ADAPT       = _b("SPEAKER_PROFILE_EMA", True)
PROFILE_ADAPT_ALPHA = _f("SPEAKER_PROFILE_EMA_ALPHA", 0.02)
PROFILE_ANCHOR_MIN  = _f("SL_PROFILE_ANCHOR_MIN", 0.85)

STATS_EVERY_HOPS    = _i("SL_STATS_EVERY_HOPS", 100)
# DEBUG: when set (e.g. /tmp/voicelock_wav), every call writes two WAV files:
#   <call_id>_in.wav  = the audio the voice lock received (what the caller said)
#   <call_id>_out.wav = the audio sent on to Sarvam STT
# Leave empty in production — about 2 MB per minute of call.
DEBUG_WAV_DIR       = os.getenv("TCVL_DEBUG_WAV_DIR", "").strip()
USE_SEPFORMER       = _b("USE_SEPFORMER", True)     # pre-lock overlap separation
USE_TSE             = _b("USE_TSE", True)           # post-lock overlap extraction

# the customer is the voice that ANSWERS the bot: only speech this soon after the
# bot stops talking is used to (re-)identify them
ANSWER_WINDOW_S     = _f("SL_ANSWER_WINDOW_S", 10.0)
ENROLL_MIN_WINDOWS  = _i("ENROLL_MIN_WINDOWS", 5)

# DEAF PROTECTION — if the caller keeps answering the bot and almost nothing of it
# passes, the lock is on the wrong voice (bot echo, a TV, a bad enrollment).
# Re-identify the customer from those very answers instead of staying deaf.
RELOCK_SPEECH_S     = _f("SL_RELOCK_SPEECH_S", 1.2)   # judged per answer, not across turns
RELOCK_PASS_MAX     = _f("SL_RELOCK_PASS_MAX", 0.25)
RELOCK_MAX          = _i("SL_RELOCK_MAX", 3)

# SENTENCE-LEVEL SCORING. On an 8 kHz phone line a 1 s slice of the SAME person
# scored anywhere from 0.05 to 0.60 (real calls V9241736…, V9241752…). A voice
# print over the whole sentence so far is far steadier, so each 100 ms slice is
# judged by the sentence it belongs to (up to LONG_BACK_HOPS back), and a short
# window is used only to notice a DIFFERENT voice taking over mid-sentence.
LONG_BACK_HOPS      = _i("SL_LONG_BACK_HOPS", 25)       # up to 2.5 s of the sentence
KEEP_OPEN_SIM       = _f("SL_KEEP_OPEN_SIM", 0.20)      # sentence stays open above this
SWITCH_HOPS         = _i("SL_SWITCH_HOPS", 3)           # 300 ms of a foreign voice ends it

# PROFILE BANK. Every answer the customer gives the bot is added to their voice
# profile, so it covers how they actually sound on this line (loud, soft, near,
# far) instead of the first 4 s of the call only.
BANK_MAX            = _i("SL_BANK_MAX", 12)
# SELF-CLEANING PROFILE. The first entry is the enrollment print, and if other
# people were talking while it was learned (V9251606030000040818: 20 s of
# chatter before the lock) it is part-someone-else. Kept forever, it made the
# caller's own turns score 0.13-0.31 and let voices resembling it through.
# Once the bank has BANK_CLEAN_MIN entries, the entry that agrees least with the
# rest is dropped when it is clearly off (below BANK_OUTLIER_SIM, or
# BANK_OUTLIER_MARGIN under the others' typical agreement).
BANK_CLEAN_MIN      = _i("SL_BANK_CLEAN_MIN", 4)
BANK_OUTLIER_SIM    = _f("SL_BANK_OUTLIER_SIM", 0.30)
BANK_OUTLIER_MARGIN = _f("SL_BANK_OUTLIER_MARGIN", 0.15)
BANK_ADD_PASS_RATIO = _f("SL_BANK_ADD_PASS_RATIO", 0.8)
BANK_ADD_MIN_S      = _f("SL_BANK_ADD_MIN_S", 0.8)
# A rejected ANSWER that still resembles the profile this much is the customer on
# a bad moment → widen the profile. Below it, it's a different voice → re-enroll.
WIDEN_MIN_SIM       = _f("SL_WIDEN_MIN_SIM", 0.30)
# Below this the answering voice is clearly a DIFFERENT person (wife, colleague,
# TV). They stay blocked — the lock is only moved to them if they are the only one
# answering the bot for FOREIGN_ANSWERS_TO_RELOCK answers in a row (phone handed over).
FOREIGN_SIM         = _f("SL_FOREIGN_SIM", 0.12)
FOREIGN_ANSWERS_TO_RELOCK = _i("SL_FOREIGN_ANSWERS_TO_RELOCK", 2)
# While the customer is ANSWERING the bot, "unsure" speech (between the hard-reject
# and accept thresholds) is passed instead of silenced. Call V9241816… lost
# "అరవై మూడు లక్షల" this way: the customer's own answer scored 0.15–0.35 and only
# 8 of 23 slices reached STT, so the bot heard "around … three thousand".
# Clearly different voices (below HARD_REJECT_COSINE, or a detected voice switch)
# are still blocked, and outside answers the strict rule still applies.
ANSWER_LENIENT      = _b("ANSWER_LENIENT", True)

# ── MODE ─────────────────────────────────────────────────────────────────────
#  "verify" (default): while the bot is SILENT the caller's audio goes to STT
#     untouched — no word is ever chopped — and each finished TURN is checked
#     against the customer's voice print as a whole (1–3 s, far more reliable
#     than 100 ms slices). A turn from a different voice is dropped before the
#     LLM sees it. While the BOT is speaking, the strict per-slice gate still runs,
#     so TV / other people / echo can't barge in on the bot.
#  "gate": the old behaviour — every 100 ms slice is gated, always.
#  Why verify is the default: on 8 kHz calls the customer's own 1 s slices scored
#  0.05–0.60 (calls V9241736…, V9241752…, V9241908…); slice gating kept cutting
#  their words or going deaf. Whole-turn scores were 0.37–0.74.
MODE                = _s("MODE", "verify")
TURN_REJECT_SIM     = _f("TURN_REJECT_SIM", 0.15)   # whole turn below this = another voice
# Once the profile has learned from a few answers (bank >= MATURE_BANK) the
# customer's own turns score 0.35-0.85 (41 turns, calls 25-Sep). Other voices
# scored 0.03 / 0.18, and an "Okay" at 0.15 cut the bot off mid-answer on
# V9251232020000040392. From then on the stricter cut-off applies.
TURN_REJECT_SIM_MATURE = _f("TURN_REJECT_SIM_MATURE", 0.30)
# (0.25 -> 0.30 after V9251332020000040408: people talking beside the caller
#  scored 0.15 / 0.16 / 0.25 and got through; "ಸಾಹೇಬ್ರು ಇಲ್ಲ ರೀ" at 0.25 even
#  cancelled the hand-over. Customers' own turns: 0.34-0.85 over ~100 turns.)
# In verify mode, how many turns in a row from one other (near-phone) voice
# before we assume the phone was handed over and re-identify the customer.
# 2 was too eager: two remarks from someone standing next to the caller would
# move the lock onto them.
TURN_RELOCK_COUNT   = _i("TURN_RELOCK_COUNT", 3)
# Until ONE turn after the lock has been verified as the customer, the lock itself
# is unproven. A wrong lock (V0011101050000043817: learned from AMD-time audio)
# then drops the real customer — so re-identify after 2 such turns, not 3.
TURN_RELOCK_COUNT_UNPROVEN = _i("TURN_RELOCK_COUNT_UNPROVEN", 2)
MATURE_BANK         = _i("MATURE_BANK", 3)
TURN_MIN_S          = _f("TURN_MIN_S", 0.5)         # below this the voice print is unreliable
# SHORT TURNS ("hello", "haan", "ok"). Call V9251111… let two short "Hello"s from
# someone else through unchecked because 0.2–0.4 s is too short for a voice print.
# They are now checked two cheaper ways:
#   * loudness — the customer holds the phone; a person across the room is far
#     quieter. Quieter than QUIET_RATIO × the customer's usual level → dropped.
#   * a very lenient voice print — only a near-zero match is dropped, so the
#     customer's own "haan"/"yes" still goes through.
QUIET_RATIO         = _f("QUIET_RATIO", 0.35)
SHORT_TURN_REJECT_SIM = _f("SHORT_TURN_REJECT_SIM", 0.05)
AMBIGUOUS_SIM       = _f("AMBIGUOUS_SIM", 0.25)       # long turn this weak + quiet → other
# Mixed turns: the caller and someone beside them in ONE turn. The whole-turn
# voice print is then a blend and scores low, so the caller's words were thrown
# away with the other person's and the bot went silent. Each piece of the turn
# (split at pauses) is now also checked; one piece that is clearly the caller
# keeps the turn.
SEG_ACCEPT_SIM      = _f("SEG_ACCEPT_SIM", 0.40)      # a piece this close = the caller spoke in it
SEG_MIN_S           = _f("SEG_MIN_S", 0.8)            # shortest piece worth a voice print
SEG_MAX_S           = _f("SEG_MAX_S", 2.5)            # longer stretches are cut into pieces
SEG_MIN_TURN_S      = _f("SEG_MIN_TURN_S", 1.2)       # shorter turns are judged whole only
# Less captured audio than this cannot be judged by voice at all. Happens when the
# caller started talking just BEFORE the lock was made: only the last 0.2 s of
# "Two crore" was recorded, scored -0.17, and the caller's answer was thrown away
# (V9291256030000042411). Such a turn is let through unless it is also quiet.
MIN_JUDGE_S         = _f("MIN_JUDGE_S", 0.35)
# Right after locking the profile rests on a few seconds of short answers, so the
# caller's own next words can score low (V9301552590000043722: "SAP" scored 0.12 and
# was dropped). Until the profile has learned from a second turn, only a turn that
# is clearly someone else (below this) is dropped.
YOUNG_REJECT_SIM    = _f("YOUNG_REJECT_SIM", 0.05)

# PROBATION (01-Oct). A new lock is not trusted until one of the caller's turns
# matches it clearly (>= PROBATION_PROVE_SIM over >= PROBATION_PROVE_S). Until
# then NO turn is dropped: a turn that doesn't match is passed and remembered,
# and two such turns in a row move the lock onto that voice. On V0011101050000043817
# the lock was wrong and 4 of the caller's turns were thrown away before it was
# corrected; with probation they would all have reached the bot.
PROBATION           = _b("PROBATION", True)
PROBATION_PROVE_SIM = _f("PROBATION_PROVE_SIM", 0.40)
PROBATION_PROVE_S   = _f("PROBATION_PROVE_S", 0.8)
PROBATION_MISMATCH_SIM = _f("PROBATION_MISMATCH_SIM", 0.20)  # 2 in a row below → re-lock
PROBATION_MAX_TURNS = _i("PROBATION_MAX_TURNS", 4)     # never confirmed by then → rebuild

# TWO-ANSWER CHECK. The first lock is built from the caller's answers to at least
# two different bot lines, and those answers must sound like the same person. An
# answer that doesn't match the others (someone else answered, AMD noise, a TV) is
# left out of the profile. One long answer (>= ENROLL_SINGLE_ANSWER_S) is enough.
ENROLL_TWO_ANSWERS  = _b("ENROLL_TWO_ANSWERS", True)
# [fix 28] a voice that keeps talking while the bot talks is a TV / video, not the caller
BGM_GUARD           = _b("BGM_GUARD", True)
BGM_SKIP_HOPS       = _i("BGM_SKIP_HOPS", 15)        # first 1.5 s of bot speech: the caller may still be finishing
BGM_MIN_S           = _f("BGM_MIN_S", 3.0)           # this much speech over the bot …
BGM_MIN_FRACTION    = _f("BGM_MIN_FRACTION", 0.35)   # … and this share of the bot's speaking time
BGM_MATCH_SIM       = _f("BGM_MATCH_SIM", 0.40)      # an answer this close to that voice is left out
BGM_BUFFER_S        = _f("BGM_BUFFER_S", 10.0)
BGM_GIVE_UP_S       = _f("BGM_GIVE_UP_S", 25.0)
ENROLL_SINGLE_ANSWER_S = _f("ENROLL_SINGLE_ANSWER_S", 6.0)
ANSWER_AGREE_SIM    = _f("ANSWER_AGREE_SIM", 0.25)     # two answers, same person
ANSWER_CHECK_MIN_S  = _f("ANSWER_CHECK_MIN_S", 1.0)    # shorter answers aren't judged

# SHORT WORDS ("ఉమ్", "అవును", "Okay", < TURN_MIN_S). A voice print of 0.3 s is
# unreliable, so four clues are added up instead of one:
#   voice print   >= 0.40 +2 | < 0.15 -1 | < 0.05 -2
#   loudness vs the customer's usual level  >= 0.6x +1 | < 0.35x -2 | < 0.2x -3
#   pitch vs the customer's usual pitch  within SHORT_PITCH_SAME_ST +1 |
#                       more than SHORT_PITCH_DIFF_ST -2 | more than SHORT_PITCH_FAR_ST -3
#   timing: an answer right after the bot asked something  +1
# total <= SHORT_DROP_SCORE → another voice, dropped; otherwise kept.
SHORT_PITCH_SAME_ST = _f("SHORT_PITCH_SAME_ST", 3.0)   # semitones
SHORT_PITCH_DIFF_ST = _f("SHORT_PITCH_DIFF_ST", 6.0)   # -2
SHORT_PITCH_FAR_ST  = _f("SHORT_PITCH_FAR_ST", 9.0)    # -3 (e.g. man vs woman)
SHORT_DROP_SCORE    = _i("SHORT_DROP_SCORE", -2)

# ── LATENCY ─────────────────────────────────────────────────────────────────
# The voice print of a turn is taken over its LAST this-many seconds only (it was
# up to 10 s: ~2.5x the CPU, no better decision).
TURN_JUDGE_MAX_S    = _f("TURN_JUDGE_MAX_S", 4.0)
# While the bot speaks every 100 ms slice was voice-printed (twice). Now a fresh
# decision every N slices of the same sentence; the slices between reuse it.
LOCKED_EVAL_EVERY_HOPS = _i("LOCKED_EVAL_EVERY_HOPS", 2)
# CPU threads torch may use in this call's process. Every call is its own process
# (JobExecutorType.PROCESS); with torch's default (all cores) 5 calls fight over
# the CPU and every call's audio falls behind. 0 = leave torch's default.
TORCH_THREADS       = _i("TORCH_THREADS", 2)

# ── FIX 1: STRICT PROFILE LEARNING (V0011301020000043983) ───────────────────
# Every turn scoring >= the lock threshold (0.35) was added to the voice profile.
# Turns that held the caller AND a person beside them were learned too, the
# profile slowly became a blend of both voices, and both then scored 0.38-0.77.
# Only a turn that clearly matches (>= this) and is NOT mixed is learned now.
BANK_ADD_TURN_SIM   = _f("BANK_ADD_TURN_SIM", 0.50)

# ── FIX 2: MIXED TURNS ──────────────────────────────────────────────────────
# A long turn is cut into pieces (at pauses, max SEG_MAX_S). A piece scoring
# >= MIXED_CUST_SIM is the customer; a piece is another person when it scores
# < MIXED_OTHER_SIM, or < MIXED_WEAK_SIM AND its pitch is far from the customer's
# (> SHORT_PITCH_DIFF_ST semitones) or it is much quieter (QUIET_RATIO). Both in
# one turn → "mixed": the turn is kept (the customer spoke), never learned, and
# last_turn_info tells the agent which part was whom.
MIXED_CUST_SIM      = _f("MIXED_CUST_SIM", 0.40)
MIXED_OTHER_SIM     = _f("MIXED_OTHER_SIM", 0.15)
MIXED_WEAK_SIM      = _f("MIXED_WEAK_SIM", 0.30)
MIXED_MAX_PIECES    = _i("MIXED_MAX_PIECES", 6)

# ── BOT MUST NOT STOP FOR ANOTHER VOICE (V0031539040000045744) ──────────────
# While the bot speaks, the agent mutes LiveKit's VAD unless the lock has passed
# the CUSTOMER's voice within this many seconds — so a person beside the caller
# cannot pause or interrupt the bot, only the customer can.
CUSTOMER_ACTIVE_S   = _f("CUSTOMER_ACTIVE_S", 0.8)

# ── GRAY ZONE (V0031550420000045410) ────────────────────────────────────────
# The bot asked "annual turnover?", the caller answered "1.2 CR" LOUDER than their
# usual level, 1 s after the question — and the turn scored 0.26 (a TV was on;
# their own turns scored 0.27-0.53 all call). It was dropped, the bot stayed
# silent and the caller hung up. A turn scoring between GRAY_MIN_SIM and the
# reject line is now KEPT when all of these hold:
#   * it is at least GRAY_LOUD_RATIO x the customer's usual level (near the phone)
#   * it starts within GRAY_ANSWER_S after the bot stopped speaking (an answer)
#   * its pitch is not clearly different from the customer's
# It is never learned from. Below GRAY_MIN_SIM a turn is still another voice.
GRAY_ZONE           = _b("GRAY_ZONE", True)
GRAY_MIN_SIM        = _f("GRAY_MIN_SIM", 0.15)
GRAY_LOUD_RATIO     = _f("GRAY_LOUD_RATIO", 0.7)
GRAY_ANSWER_S       = _f("GRAY_ANSWER_S", 6.0)

# ── ENROLLMENT QUALITY (V0061536040000047857) ───────────────────────────────
# A 100 ms slice counts as "speech" from level 60 up, and the caller speaks at
# 2000-3000: breath, room sound and the bot's echo all counted towards the 4 s.
# Only slices at least ENROLL_LOUD_RATIO x the loud level of the answers so far
# (their ENROLL_LOUD_PERCENTILE-th percentile) are used now.
ENROLL_LOUD         = _b("ENROLL_LOUD", True)
ENROLL_LOUD_RATIO   = _f("ENROLL_LOUD_RATIO", 0.35)
ENROLL_LOUD_PERCENTILE = _f("ENROLL_LOUD_PERCENTILE", 90.0)
# The first lock waits for this many transcribed words. Only applies once the
# agent has called note_enroll_words() at least once; 0 switches it off.
ENROLL_MIN_WORDS    = _i("ENROLL_MIN_WORDS", 4)

# ── FAST RE-LOCK (V0061536040000047857) ─────────────────────────────────────
# The lock was on the wrong sound and the caller's question was dropped three
# times; the 3-in-a-row rule never fired because weak "matches" in between reset
# it. A dropped turn that (a) lasts >= FAST_RELOCK_MIN_S, (b) starts within
# GRAY_ANSWER_S after the bot stopped and (c) is >= FAST_RELOCK_LOUD_RATIO x the
# locked voice's level is the person holding the phone answering the bot.
# FAST_RELOCK_COUNT of them after different bot lines, with no turn in between
# that clearly matches the lock (>= PROBATION_PROVE_SIM), move the lock.
FAST_RELOCK         = _b("FAST_RELOCK", True)
FAST_RELOCK_COUNT   = _i("FAST_RELOCK_COUNT", 2)
FAST_RELOCK_MIN_S   = _f("FAST_RELOCK_MIN_S", 1.0)
FAST_RELOCK_LOUD_RATIO = _f("FAST_RELOCK_LOUD_RATIO", 0.8)

# ── RIVAL VOICES (V0061629030000047900) ─────────────────────────────────────
# On 8 kHz phone audio one person's score against their own profile moves
# between 0.2 and 0.6, so a bystander at 0.30-0.47 cannot be told from the
# caller by the customer profile alone. But the bystander's turns also move:
# sooner or later one of them is dropped (or marked OTHER inside a mixed turn).
# That audio is certainly not the customer, so its voice print is kept as a
# "rival". From then on a turn is the rival's when ALL of these hold:
#   * it matches a rival at least RIVAL_MIN_SIM
#   * it matches that rival at least RIVAL_MARGIN better than the customer
#   * its customer score is below RIVAL_CUST_SAFE (a clear customer match is
#     never overruled)
#   * it lasts at least RIVAL_MIN_S and the lock is confirmed (not on probation)
# Safety: a rival is never learned from audio that resembles the customer
# (>= RIVAL_LEARN_MAX_CUST_SIM), and a rival that a clearly-verified customer
# turn matches (>= RIVAL_PURGE_SIM) is forgotten. A re-lock forgets all rivals.
RIVAL               = _s("RIVAL", "shadow")          # shadow | on | off
RIVAL_MIN_SIM       = _f("RIVAL_MIN_SIM", 0.45)
RIVAL_MARGIN        = _f("RIVAL_MARGIN", 0.15)
RIVAL_CUST_SAFE     = _f("RIVAL_CUST_SAFE", 0.50)
RIVAL_MIN_S         = _f("RIVAL_MIN_S", 1.0)
RIVAL_LEARN_MIN_S   = _f("RIVAL_LEARN_MIN_S", 1.0)
RIVAL_LEARN_MAX_CUST_SIM = _f("RIVAL_LEARN_MAX_CUST_SIM", 0.25)
RIVAL_MERGE_SIM     = _f("RIVAL_MERGE_SIM", 0.50)
RIVAL_PURGE_SIM     = _f("RIVAL_PURGE_SIM", 0.45)
RIVAL_MAX           = _i("RIVAL_MAX", 4)

# ── NOISE-ONLY TURNS (08-Oct) ───────────────────────────────────────────────
# While the bot is silent all audio goes to STT untouched. A steady fan / AC /
# traffic or a beep / dial tone is not speech-like, so nothing was captured for
# the voice check and judge_turn() answered "unknown" — the agent then passed
# whatever word STT invented from the noise. Now: STT produced words, but the
# line carried NO speech-like sound for the last NOISE_ONLY_S seconds (all of
# them with the bot silent, and the customer did not just talk over the bot)
# → the words came from noise → "other".  TCVL_NOISE_REJECT=false switches it off.
NOISE_REJECT        = _b("NOISE_REJECT", True)
NOISE_ONLY_S        = _f("NOISE_ONLY_S", 2.5)


# ════════════════════════════════════════════════════════════════════════════
#  SIGNAL HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _rms(x: np.ndarray) -> float:
    if x is None or len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))


def _sfm(x: np.ndarray) -> float:
    if len(x) == 0:
        return 1.0
    mag = np.abs(np.fft.rfft(x.astype(np.float32) / 32768.0, n=max(256, len(x)))) + 1e-10
    return float(np.clip(np.exp(np.mean(np.log(mag))) / np.mean(mag), 0.0, 1.0))


def _pitch_hz(x: np.ndarray, sr: int) -> Optional[float]:
    """Median pitch (F0) of the voiced parts of x, by autocorrelation (70-400 Hz).
    None if fewer than 3 voiced 40 ms frames. Cheap: ~1 ms per second of audio."""
    if x is None or len(x) < int(0.12 * sr):
        return None
    x = x.astype(np.float32)
    frame, step = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / 400), int(sr / 70)
    if frame <= hi + 2:
        return None
    level = float(np.sqrt(np.mean(x ** 2))) + 1e-6
    f0s = []
    for s in range(0, len(x) - frame + 1, step):
        f = x[s:s + frame]
        f = f - f.mean()
        e = float(np.dot(f, f))
        if e < (0.3 * level) ** 2 * frame:
            continue
        ac = np.correlate(f, f, mode="full")[frame - 1:]
        seg = ac[lo:hi + 1]
        k = int(np.argmax(seg))
        if seg[k] / (ac[0] + 1e-9) < 0.45:
            continue
        lag = lo + k
        # prefer the shortest lag that is nearly as strong (avoid octave-down errors)
        for div in (3, 2):
            l2 = int(round(lag / div))
            if l2 >= lo and ac[l2] >= 0.85 * ac[lag]:
                lag = l2
                break
        f0s.append(sr / lag)
    if len(f0s) < 3:
        return None
    return float(np.median(f0s))


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-9 else v.astype(np.float32)


def _windows(x: np.ndarray, win: int, step: int) -> List[np.ndarray]:
    if len(x) < win // 2:
        return []
    if len(x) <= win:
        return [x]
    return [x[s:s + win] for s in range(0, len(x) - win + 1, step)]


def _split(x: np.ndarray, max_len: int) -> List[np.ndarray]:
    return [x[s:s + max_len] for s in range(0, len(x), max_len)]


# ════════════════════════════════════════════════════════════════════════════
#  ECAPA EMBEDDING
#  (Not embedding_backend.ecapa_embedding: that one early-exits on the FIRST
#   300 ms window, which would ignore most of an 800 ms decision window.)
# ════════════════════════════════════════════════════════════════════════════
_ECAPA = None
_ECAPA_FAILED = False
_ECAPA_LOAD_LOCK = threading.Lock()
_ECAPA_INFER_LOCK = threading.Lock()


def _load_ecapa():
    global _ECAPA, _ECAPA_FAILED
    if _ECAPA is not None or _ECAPA_FAILED:
        return _ECAPA
    with _ECAPA_LOAD_LOCK:
        if _ECAPA is not None or _ECAPA_FAILED:
            return _ECAPA
        try:
            import torch
            if TORCH_THREADS > 0:
                try:
                    torch.set_num_threads(TORCH_THREADS)
                except Exception:
                    pass
            from speechbrain.inference import EncoderClassifier
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _ECAPA = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/ecapa",
                run_opts={"device": device},
            )
            _ECAPA.eval()
            logger.info(f"✅ ECAPA loaded on {device.upper()}")
        except Exception as e:
            _ECAPA_FAILED = True
            logger.error(f"❌ ECAPA load failed — voice lock cannot work: {e}")
    return _ECAPA


def ecapa_embed(pcm: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    model = _load_ecapa()
    if model is None or pcm is None or len(pcm) == 0:
        return None
    x = pcm.astype(np.float32) / 32768.0
    if float(np.max(np.abs(x))) < 50.0 / 32768.0:
        return None
    import torch
    t = torch.from_numpy(x).unsqueeze(0)
    if sample_rate != 16000:
        try:
            import torchaudio.functional as AF
            t = AF.resample(t, orig_freq=sample_rate, new_freq=16000)
        except Exception:
            n_out = int(len(x) * 16000 / sample_rate)
            y = np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)
            t = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
    if t.shape[-1] < 8000:                      # ECAPA needs ≥ 0.5 s
        t = torch.nn.functional.pad(t, (0, 8000 - t.shape[-1]))
    device = getattr(model, "device", "cpu")
    with _ECAPA_INFER_LOCK, torch.no_grad():
        emb = model.encode_batch(t.to(device))
    return _norm(emb.squeeze().detach().cpu().numpy())


def _make_separator(sample_rate: int):
    try:
        if not USE_SEPFORMER:
            return None
        from source_separator import SourceSeparator
        return SourceSeparator(sample_rate=sample_rate)
    except Exception as e:
        logger.warning(f"SepFormer unavailable: {e}")
        return None


def _make_extractor(sample_rate: int):
    try:
        if not USE_TSE:
            return None
        from target_extractor1 import TargetExtractor
        return TargetExtractor(sample_rate=sample_rate)
    except Exception as e:
        logger.warning(f"TSE unavailable: {e}")
        return None


def preload_models(sample_rate: int = 8000) -> None:
    """Call from the worker prewarm so the first call doesn't pay model load time."""
    _load_ecapa()
    for maker in (_make_separator, _make_extractor):
        obj = maker(sample_rate)
        if obj is not None:
            try:
                obj.available()
            except Exception as e:
                logger.warning(f"preload failed for {type(obj).__name__}: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  VOICE LOCK
# ════════════════════════════════════════════════════════════════════════════
class LockState(enum.Enum):
    ENROLLING = "ENROLLING"
    LOCKED = "LOCKED"
    FAILED_OPEN = "FAILED_OPEN"   # could not identify one clean voice — passes audio


class _Hop:
    __slots__ = ("idx", "pcm", "rms", "speech", "answer", "bot")

    def __init__(self, idx: int, pcm: np.ndarray, rms: float, speech: bool, answer: bool,
                 bot: bool = False):
        self.idx, self.pcm, self.rms, self.speech = idx, pcm, rms, speech
        self.answer = answer   # speech while the bot is silent, shortly after it spoke
        self.bot = bot         # the bot was talking when this audio arrived


class _NoHop:
    bot = False


_NO_HOP = _NoHop()


EmbedFn = Callable[[np.ndarray, int], Optional[np.ndarray]]


class CustomerVoiceLock:
    def __init__(
        self,
        sample_rate: int = 8000,
        embed_fn: Optional[EmbedFn] = None,
        separator="auto",
        extractor="auto",
        call_id: Optional[str] = None,
    ):
        self.sample_rate = sample_rate
        self.hop_len = max(1, int(sample_rate * HOP_MS / 1000))
        self._hop_s = self.hop_len / sample_rate
        self._fade = max(1, int(sample_rate * FADE_MS / 1000))
        self._win_len = (WINDOW_BACK_HOPS + 1 + LOOKAHEAD_HOPS) * self.hop_len
        self._embed: EmbedFn = embed_fn or ecapa_embed
        self._separator = _make_separator(sample_rate) if separator == "auto" else separator
        self._extractor = _make_extractor(sample_rate) if extractor == "auto" else extractor
        self._extractor_ok = False
        self.call_id = call_id or uuid.uuid4().hex[:8]
        self._mutex = threading.RLock()
        self._wav_in = self._wav_out = None
        if DEBUG_WAV_DIR:
            try:
                os.makedirs(DEBUG_WAV_DIR, exist_ok=True)
                self._wav_in = self._open_wav(f"{self.call_id}_in.wav")
                self._wav_out = self._open_wav(f"{self.call_id}_out.wav")
            except Exception as e:
                logger.warning(f"[{self.call_id}] debug WAV disabled: {e}")
                self._wav_in = self._wav_out = None

        # streaming
        self._pending = np.zeros(0, dtype=np.int16)
        ring_len = max(TSE_WINDOW_HOPS, LONG_BACK_HOPS, WINDOW_BACK_HOPS) + 2 * LOOKAHEAD_HOPS + 6
        self._ring: Deque[_Hop] = collections.deque(maxlen=ring_len)
        self._next_idx = 0
        self._next_release = 0
        self._prev_open = True
        self._rms_hist: Deque[float] = collections.deque(maxlen=max(10, int(BG_HISTORY_S / self._hop_s)))
        self._bg = 5.0
        # what arrived from the phone line, before any gating (1 entry per hop):
        # (rms, speech-like). Lets the agent tell "caller silent / no audio on the
        # line" apart from "caller talking but STT returns nothing".
        self._line_hist: Deque[Tuple[float, bool]] = collections.deque(
            maxlen=max(10, int(60.0 / self._hop_s)))
        self._line_speech_total = 0
        self._last_line_speech_idx = -10 ** 9   # last speech-like hop while the bot was silent

        # bot state (written from the event loop, read here — plain attributes)
        self._bot_speaking = False
        self._bot_end_ts = 0.0
        self._bot_end_idx = 0

        # enrollment
        self.state = LockState.ENROLLING
        self._utts: List[List[np.ndarray]] = []
        self._utt_aids: List[int] = []      # which bot line each utterance answered
        self._answer_id = 0                 # +1 every time the bot starts talking
        self._two_answer_logged = False
        self._turn_cache = None             # (key, profile, sim, emb, part) from prepare_turn
        self._last_eval: Optional[Tuple[int, bool]] = None   # (hop idx, open?)
        self._proc_s = 0.0                  # time spent in process()
        self._proc_audio_s = 0.0            # audio handled by process()
        self._proc_max_ms = 0.0
        self._judge_ms: Deque[float] = collections.deque(maxlen=50)
        # what the last judge_turn() found — read by the agent
        self.last_turn_info: Dict = {"mixed": False}
        self._last_cust_pass_idx = -10 ** 9   # last hop of the customer passed while the bot spoke
        self._foreign_aids: List[int] = []    # which bot line each foreign turn followed
        self._last_enroll_idx = -10 ** 9
        self._enroll_speech_s = 0.0
        self._attempts = 0
        self._building = False
        self._best: Optional[Dict] = None

        # profile
        self._profile: Optional[np.ndarray] = None
        self._anchor: Optional[np.ndarray] = None
        self._thr = 0.42
        self._thr_clean = 0.52
        self._tse_ref: Optional[np.ndarray] = None
        self._tse_cache: Optional[Tuple[int, int, Optional[np.ndarray]]] = None

        self.stats: collections.Counter = collections.Counter()
        self._last_sim = 0.0
        self._bot_seen = False
        # Enrollment can be held until the bot has greeted (see hold_enrollment).
        self._enroll_open = True
        self._turns_since_lock = 0
        self._verified_since_lock = 0
        self._proven = False
        self._probation_turns: List[np.ndarray] = []
        self._cust_f0: Deque[float] = collections.deque(maxlen=20)
        self._wait_until_s = 0.0
        self._sims: Deque[float] = collections.deque(maxlen=200)
        self._answer_hist: Deque[Tuple[bool, np.ndarray]] = collections.deque(
            maxlen=max(10, int(2 * RELOCK_SPEECH_S / self._hop_s)))
        self._relocks = 0
        self._need_more = False
        self._run_open = False
        self._last_cand_idx = -10 ** 9
        self._run_start_idx = 0
        self._switch_streak = 0
        self._run_foreign = False
        self._bank: List[np.ndarray] = []
        self._pending_answer: Optional[List[Tuple[bool, np.ndarray]]] = None
        self._foreign_answers = 0
        self._turn_audio: Deque[Tuple[int, np.ndarray]] = collections.deque(maxlen=150)   # 15 s
        # Untouched audio of the caller's current answer (silences included,
        # 0.5 s before the first word), for a second STT listen when the first
        # one missed a number or time. Reset whenever the bot starts talking.
        self._raw_turn: Deque[np.ndarray] = collections.deque(maxlen=120)    # 12 s
        self._raw_preroll: Deque[np.ndarray] = collections.deque(maxlen=5)   # 0.5 s
        self._raw_started = False
        self._cust_rms: Deque[float] = collections.deque(maxlen=20)
        self._foreign_turns: List[np.ndarray] = []
        self._loud_foreign: List[Tuple[int, np.ndarray]] = []   # FAST_RELOCK: (bot line, audio)
        self._rivals: List[np.ndarray] = []     # voice prints of voices known NOT to be the customer
        self._enroll_words = 0              # words STT heard from the caller so far
        self._enroll_words_seen = False     # the agent reports words at all
        self._words_wait_logged = False
        self._last_err_ts = 0.0
        logger.info(f"[{self.call_id}] 🔒 CustomerVoiceLock init sr={sample_rate} hop={HOP_MS}ms "
                    f"lookahead={LOOKAHEAD_HOPS * HOP_MS}ms sep={self._separator is not None} "
                    f"tse={self._extractor is not None}")

    # ── public API ────────────────────────────────────────────────────────
    @property
    def locked(self) -> bool:
        return self.state is LockState.LOCKED

    def set_bot_speaking(self, speaking: bool) -> None:
        speaking = bool(speaking)
        if speaking != self._bot_speaking:
            logger.debug(f"[{self.call_id}] bot_speaking={speaking}")
        if speaking and not self._bot_speaking and self._answer_hist:
            n = len(self._answer_hist)
            ok = sum(1 for good, _ in self._answer_hist if good)
            logger.info(f"[{self.call_id}] 🎧 last answer: {n * self._hop_s:.1f}s of speech, "
                        f"{ok}/{n} hops reached STT")
            # embedding it is done on the audio thread, not the event loop
            self._pending_answer = list(self._answer_hist)
            self._answer_hist.clear()   # each answer is judged on its own
        if speaking and not self._bot_speaking:
            self._raw_turn.clear()
            self._raw_started = False
        if speaking and not self._bot_speaking:
            self._answer_id += 1
            self._bgm_bot_start = self._next_idx
        if speaking:
            self._bot_seen = True
        if self._bot_speaking and not speaking:
            self._bot_end_ts = time.monotonic()
            self._bot_end_idx = self._next_idx     # audio clock, not wall clock
        self._bot_speaking = speaking

    def hold_enrollment(self) -> None:
        """Do not learn a voice yet. For agents that listen BEFORE the bot speaks
        (e.g. AMD before the greeting): until the bot has talked, every sound would
        count as the customer's 'answer' — ringback, IVR, a bystander's 'hello?'."""
        with self._mutex:
            self._enroll_open = False

    def open_enrollment(self) -> None:
        """The bot has greeted — learn the voice that answers it from now on."""
        with self._mutex:
            if not self._enroll_open:
                self._enroll_open = True
                logger.info(f"[{self.call_id}] 🎙️ enrollment opened (bot has greeted)")

    def note_enroll_words(self, n: int) -> None:
        """The agent heard `n` words from the caller (a final transcript, not
        spoken over the bot). See ENROLL_MIN_WORDS."""
        with self._mutex:
            self._enroll_words_seen = True
            if self.state is not LockState.ENROLLING or not self._enroll_open:
                return
            self._enroll_words += max(0, int(n))
            self._maybe_build()

    def _is_answer_time(self) -> bool:
        if self._bot_speaking:
            return False
        if not self._bot_seen:          # no bot events wired → can't window, allow
            return True
        since = (self._next_idx - self._bot_end_idx) * self._hop_s
        return POST_BOT_GRACE_S <= since <= ANSWER_WINDOW_S

    def reset_stream(self) -> None:
        """A new STT stream started — drop half-processed audio, keep the profile."""
        with self._mutex:
            self._pending = np.zeros(0, dtype=np.int16)
            self._ring.clear()
            self._next_release = self._next_idx
            self._tse_cache = None
            self._prev_open = not self.locked

    def process(self, pcm: np.ndarray) -> List[np.ndarray]:
        """Feed int16 mono audio of any length; returns released 100 ms hops (int16)."""
        out: List[np.ndarray] = []
        t0 = time.monotonic()
        with self._mutex:
            if self._pending_answer is not None:
                done, self._pending_answer = self._pending_answer, None
                self._learn_from_answer(done)
            if pcm is not None and len(pcm):
                pcm = np.asarray(pcm, dtype=np.int16)
                self._pending = np.concatenate([self._pending, pcm])
                if self._wav_in is not None:
                    self._wav_in.writeframes(pcm.tobytes())
            while len(self._pending) >= self.hop_len:
                raw = self._pending[:self.hop_len].copy()
                self._pending = self._pending[self.hop_len:]
                self._push(raw)
                # The 300 ms look-ahead is only needed to judge audio heard WHILE THE
                # BOT SPEAKS. In verify mode everything else goes to STT untouched, so
                # holding it back only added 0.3 s to every reply. Release at once.
                need = (0 if (MODE == "verify" and not self._bot_speaking
                              and not (self._get(self._next_release) or _NO_HOP).bot)
                        else LOOKAHEAD_HOPS)
                while self._next_idx - 1 - self._next_release >= need:
                    hop = self._get(self._next_release)
                    self._next_release += 1
                    if hop is not None:
                        out.append(self._release(hop))
            if self._wav_out is not None and out:
                self._wav_out.writeframes(np.concatenate(out).astype(np.int16).tobytes())
            dt = time.monotonic() - t0
            self._proc_s += dt
            if pcm is not None:
                self._proc_audio_s += len(pcm) / self.sample_rate
            self._proc_max_ms = max(self._proc_max_ms, dt * 1000.0)
        return out

    def timing(self) -> Dict[str, float]:
        """Cost of the voice lock: share of real time spent in process(), worst
        single call, typical / worst turn check (ms)."""
        j = list(self._judge_ms)
        return {
            "cpu_pct": 100.0 * self._proc_s / max(1e-6, self._proc_audio_s),
            "proc_max_ms": self._proc_max_ms,
            "judge_ms_med": float(np.median(j)) if j else 0.0,
            "judge_ms_max": max(j) if j else 0.0,
        }

    def _open_wav(self, name: str):
        w = wave.open(os.path.join(DEBUG_WAV_DIR, name), "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(self.sample_rate)
        return w

    def summary(self) -> str:
        s = self.stats
        if self._sims:
            q = np.percentile(np.array(self._sims), [10, 50, 90])
            dist = f" sims p10/p50/p90={q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}"
        else:
            dist = ""
        return (f"[{self.call_id}] VoiceLock state={self.state.value} thr={self._thr:.2f}/"
                f"{self._thr_clean:.2f} pass={s['pass']} tse={s['tse_ok']} "
                f"tse_fail={s['tse_fail']} soft={s['soft_pass']} hold={s['hold_open']} "
                f"backfill={s['backfill']} other={s['other']} "
                f"unsure={s['uncertain']} noise/silence={s['silence']} last_sim={self._last_sim:.2f}"
                f"{dist} bank={len(self._bank)} cleaned={s['bank_cleaned']} widened={s['widened']} switch={s['switch']} "
                f"foreign_blocked={s['foreign_blocked']} answer_pass={s['answer_pass']} "
                f"mode={MODE} turns_ok={s['turn_customer']} turns_dropped={s['turn_other']} "
                f"relocks={self._relocks} bot_events={'yes' if self._bot_seen else 'NO'} "
                f"proven={'yes' if self._proven else 'no'} "
                f"mixed={s['turn_mixed']} bank_skipped={s['bank_skipped']} gray_kept={s['gray_kept']} "
                f"fast_relock={s['fast_relock']} noise_dropped={s['turn_noise']} words={self._enroll_words} "
                f"rival={RIVAL}:{len(self._rivals)}known/{s['rival_drop']}dropped/{s['rival_shadow']}shadow "
                f"pitch={np.median(self._cust_f0) if self._cust_f0 else 0:.0f}Hz "
                f"cpu={100.0 * self._proc_s / max(1e-6, self._proc_audio_s):.0f}% "
                f"proc_max={self._proc_max_ms:.0f}ms "
                f"judge_ms={np.median(self._judge_ms) if self._judge_ms else 0:.0f}/"
                f"{max(self._judge_ms) if self._judge_ms else 0:.0f} "
                f"line_speech={self._line_speech_total * self._hop_s:.1f}s "
                f"line_max={max((r for r, _ in self._line_hist), default=0.0):.0f}")

    def line_activity(self, seconds: float = 20.0) -> Dict[str, float]:
        """Raw phone-line audio over the last `seconds` (max 60), BEFORE the voice
        lock: loudest 100 ms level, seconds of speech-like audio, seconds received."""
        with self._mutex:
            n = max(1, int(round(min(seconds, 60.0) / self._hop_s)))
            recent = list(self._line_hist)[-n:]
            return {
                "max_rms": max((r for r, _ in recent), default=0.0),
                "speech_s": sum(1 for _, sp in recent if sp) * self._hop_s,
                "received_s": len(recent) * self._hop_s,
            }

    def _judge_hops(self) -> int:
        return max(5, int(round(TURN_JUDGE_MAX_S / self._hop_s)))

    @staticmethod
    def _turn_key(audio) -> Tuple[int, int, int]:
        return (audio[0][0], audio[-1][0], len(audio))

    def prepare_turn(self) -> None:
        """Call when the caller STOPS speaking (VAD), from a worker thread. The
        voice print of the turn so far is computed now, while LiveKit is still
        waiting to see if the caller goes on (0.5-1.8 s), so judge_turn() is
        instant. If they do go on, judge_turn() simply recomputes."""
        with self._mutex:
            if self.state is not LockState.LOCKED or MODE != "verify" or not self._turn_audio:
                return
            audio = list(self._turn_audio)
            prof = self._profile
        key = self._turn_key(audio)
        pcm = np.concatenate([p for _, p in audio[-self._judge_hops():]])
        sim, emb = self._similarity(pcm)
        pieces = (self._piece_scores(audio)
                  if len(audio) * self._hop_s >= SEG_MIN_TURN_S else [])
        with self._mutex:
            if self._profile is prof:
                self._turn_cache = (key, prof, sim, emb, pieces)

    def judge_turn(self) -> Tuple[str, Optional[float]]:
        """Call when a user turn is committed. Returns ("customer" | "other" |
        "unknown", similarity). "other" = drop this transcript.
        The voice print is computed OUTSIDE the lock's mutex, so the audio path
        (process) never waits for it — and usually it was already computed by
        prepare_turn() during the end-of-turn wait."""
        t0 = time.monotonic()
        with self._mutex:
            audio = list(self._turn_audio)
            self._turn_audio.clear()
            cache, self._turn_cache = self._turn_cache, None
            if (NOISE_REJECT and self.state is LockState.LOCKED and MODE == "verify"
                    and not audio and self._noise_only()):
                self.stats["turn_noise"] += 1
                self.last_turn_info = {"mixed": False, "noise": True}
                logger.warning(f"[{self.call_id}] 🔇 words from NOISE dropped — no speech-like "
                               f"sound on the line for the last {NOISE_ONLY_S:.1f}s "
                               f"(fan / traffic / beep / tone)")
                return "other", None
            if self.state is not LockState.LOCKED or MODE != "verify" or not audio:
                return "unknown", None
            prof = self._profile
        key = self._turn_key(audio)
        if cache is not None and cache[0] == key and cache[1] is prof:
            _, _, sim, emb, pieces = cache
            self.stats["judge_cached"] += 1
        else:
            sim, emb = self._similarity(
                np.concatenate([p for _, p in audio[-self._judge_hops():]]))
            pieces = (self._piece_scores(audio)
                      if len(audio) * self._hop_s >= SEG_MIN_TURN_S else [])

        def part_fn():
            sims = [p["sim"] for p in pieces if p["sim"] is not None]
            return max(sims) if sims else None

        with self._mutex:
            self.last_turn_info = {"mixed": False}
            if self.state is not LockState.LOCKED:
                return "unknown", sim
            res = self._judge_locked(audio, sim, emb, part_fn, pieces)
        self._judge_ms.append((time.monotonic() - t0) * 1000.0)
        return res

    def _noise_only(self) -> bool:
        """See NOISE_REJECT. True when the last NOISE_ONLY_S seconds of audio were
        all heard with the bot silent and none of it was speech-like."""
        if self._bot_speaking:
            return False                        # the agent handles words over the bot
        now = self._next_idx
        if self._bot_seen and (now - self._bot_end_idx) * self._hop_s < NOISE_ONLY_S:
            return False                        # may be the end of a barge-in
        if (now - self._last_cust_pass_idx) * self._hop_s < NOISE_ONLY_S:
            return False                        # the customer just spoke over the bot
        return (now - self._last_line_speech_idx) * self._hop_s >= NOISE_ONLY_S

    def _judge_locked(self, audio, sim, emb, part_fn, pieces=()) -> Tuple[str, Optional[float]]:
        dur = len(audio) * self._hop_s
        pcm = np.concatenate([p for _, p in audio[-100:]])
        loud = _rms(pcm)
        typical = float(np.median(self._cust_rms)) if len(self._cust_rms) >= 2 else None
        quiet = typical is not None and loud < QUIET_RATIO * typical
        if sim is not None:
            self._last_sim = sim
            self._sims.append(sim)
        loud_txt = (f"level {loud:.0f} vs customer {typical:.0f}" if typical
                    else f"level {loud:.0f}")
        mix = self._mixed_analysis(list(pieces), typical) if pieces else {"mixed": False}
        mixed = bool(mix.get("mixed"))
        if mixed:
            self.stats["turn_mixed"] += 1
            logger.warning(f"[{self.call_id}] 👥 MIXED turn — the customer and someone else "
                           f"both spoke ({dur:.1f}s, whole sim {sim if sim is not None else float('nan'):.2f}): "
                           f"{mix.get('timeline')} — kept, NOT learned")
        self.last_turn_info = mix
        # [fix 17] read by the agent's soft drop
        self.last_judge = {"level": float(loud), "typical": typical,
                           "quiet": bool(quiet), "dur": dur}

        if dur < MIN_JUDGE_S and not quiet:
            logger.info(f"[{self.call_id}] 🗣️ turn too short to judge by voice "
                        f"({dur:.1f}s captured, {loud_txt}) — let through")
            return "unknown", sim
        if dur < TURN_MIN_S:
            score, why = self._short_turn_score(pcm, sim, loud, typical, audio[0][0])
            if PROBATION and not self._proven:
                logger.info(f"[{self.call_id}] 🟡 probation: short turn passed "
                            f"({dur:.1f}s, score {score:+d}: {why})")
                return "customer", sim
            if score <= SHORT_DROP_SCORE:
                self.stats["turn_other"] += 1
                logger.warning(f"[{self.call_id}] 🚫 short turn from another voice dropped "
                               f"({dur:.1f}s, score {score:+d}: {why})")
                return "other", sim
            logger.info(f"[{self.call_id}] 🗣️ short turn accepted ({dur:.1f}s, "
                        f"score {score:+d}: {why})")
            return "customer", sim

        if sim is None:
            return "unknown", None

        reject_below = (TURN_REJECT_SIM_MATURE if len(self._bank) >= MATURE_BANK
                        else TURN_REJECT_SIM
                        if (len(self._bank) >= 2 or self._turns_since_lock > 3)
                        else YOUNG_REJECT_SIM)
        self._turns_since_lock += 1
        if PROBATION and not self._proven:
            return self._probation_turn(pcm, sim, emb, dur, loud, quiet, loud_txt, mixed)
        is_other = sim < reject_below or (quiet and sim < AMBIGUOUS_SIM)
        rival_hit = False
        if (RIVAL != "off" and not is_other and self._rivals and emb is not None
                and dur >= RIVAL_MIN_S and sim < RIVAL_CUST_SAFE):
            rsim = self._rival_match(emb)
            if rsim >= RIVAL_MIN_SIM and rsim - sim >= RIVAL_MARGIN:
                if RIVAL == "on":
                    rival_hit = is_other = True
                    self.stats["rival_drop"] += 1
                    self.last_judge["rival"] = True
                    logger.warning(f"[{self.call_id}] 👤 this is a voice already known as NOT the "
                                   f"customer (other-voice match {rsim:.2f} vs customer {sim:.2f}, "
                                   f"{dur:.1f}s, {loud_txt})")
                else:
                    self.stats["rival_shadow"] += 1
                    logger.warning(f"[{self.call_id}] 👤 [shadow] WOULD drop — matches a known other "
                                   f"voice {rsim:.2f} vs customer {sim:.2f} ({dur:.1f}s, {loud_txt}); "
                                   f"kept, set TCVL_RIVAL=on to drop")
        if is_other and not rival_hit and dur >= SEG_MIN_TURN_S:
            part = part_fn()
            if part is not None and part >= SEG_ACCEPT_SIM:
                # the caller spoke in this turn, with someone else before/after
                # them. Keep it — silence here is what made callers hang up.
                # Not learned from: part of it is another voice.
                self._foreign_turns = []
                if not mixed:
                    self.stats["turn_mixed"] += 1
                    self.last_turn_info = {"mixed": True, "timeline": "(piece match only)"}
                logger.info(f"[{self.call_id}] 🗣️ mixed turn kept — the customer's voice is "
                            f"in part of it (part sim={part:.2f}, whole {sim:.2f}, {dur:.1f}s, "
                            f"{loud_txt})")
                return "customer", part
        if not is_other:
            self._foreign_turns = []
            if sim >= PROBATION_PROVE_SIM and not mixed:
                self._loud_foreign = []     # the lock clearly fits the caller
            if sim >= BANK_ADD_TURN_SIM and not mixed:
                self._rival_purge(emb)
            self._verified_since_lock += 1
            if dur >= TURN_MIN_S and not mixed:
                f0 = _pitch_hz(pcm, self.sample_rate)
                if f0 is not None:
                    self._cust_f0.append(f0)
            self.stats["turn_customer"] += 1
            self._cust_rms.append(loud)
            if sim >= BANK_ADD_TURN_SIM and emb is not None and not mixed:
                self._bank_add(emb)
            elif emb is not None and sim >= self._thr:
                self.stats["bank_skipped"] += 1
            logger.info(f"[{self.call_id}] 🗣️ turn verified as customer "
                        f"(sim={sim:.2f}, {dur:.1f}s, {loud_txt}, bank={len(self._bank)})")
            return "customer", sim

        if GRAY_ZONE and sim >= GRAY_MIN_SIM and not quiet and not rival_hit:
            keep, why = self._gray_zone_keep(pcm, loud, typical, audio[0][0])
            if keep:
                self._foreign_turns = []
                self.stats["gray_kept"] += 1
                logger.warning(f"[{self.call_id}] 🟠 weak voice match KEPT — near-phone voice "
                               f"answering the bot (sim={sim:.2f}, {dur:.1f}s, {why}); not learned")
                return "customer", sim

        self.stats["turn_other"] += 1
        if not rival_hit and not quiet:
            self._rival_learn(emb, dur, "dropped turn")
        # A known other voice does not count towards the FAST re-lock (it would hand
        # the lock to the bystander). The slower rule below still applies: only
        # that voice answering TURN_RELOCK_COUNT bot lines in a row, with no
        # customer turn in between, moves the lock.
        if (FAST_RELOCK and not rival_hit
                and self._answers_bot_loudly(loud, typical, audio[0][0], dur)):
            self._loud_foreign.append((self._answer_id, pcm))
            del self._loud_foreign[:-4]
            lines = len({a for a, _ in self._loud_foreign})
            if lines >= FAST_RELOCK_COUNT:
                utts = [u for _, u in self._loud_foreign]
                self._loud_foreign = []
                self._foreign_turns = []
                self.stats["fast_relock"] += 1
                logger.error(f"[{self.call_id}] 🔁 {len(utts)} loud answers to the bot were "
                             f"dropped as another voice (last sim={sim:.2f}, {loud_txt}) — "
                             f"the lock is on the wrong voice; re-identifying the customer "
                             f"from those answers, this turn is kept")
                self._relock_from(utts)
                return "customer", sim
            logger.warning(f"[{self.call_id}] ⚠️ dropped turn is loud and answers the bot "
                           f"({lines}/{FAST_RELOCK_COUNT} before the lock is moved)")
        if not quiet:
            self._foreign_add(pcm)
        need_foreign = (TURN_RELOCK_COUNT_UNPROVEN if self._verified_since_lock == 0
                        else TURN_RELOCK_COUNT)
        if self._foreign_answers_distinct() >= need_foreign:
            # the only (loud, near-phone) voice talking to the bot for several
            # turns → the phone changed hands; follow the person on the line
            utts = self._foreign_turns
            self._foreign_turns = []
            logger.error(f"[{self.call_id}] 🔁 {len(utts)} turns in a row from a different "
                         f"voice (last sim={sim:.2f}) — re-identifying the customer")
            self._relock_from(utts)
            return "customer", sim
        logger.warning(f"[{self.call_id}] 🚫 turn from a DIFFERENT voice dropped "
                       f"(sim={sim:.2f}, {dur:.1f}s, {loud_txt}{', too quiet' if quiet else ''})")
        return "other", sim

    def _gray_zone_keep(self, pcm: np.ndarray, loud: float, typical: Optional[float],
                        first_idx: int) -> Tuple[bool, str]:
        """A weak-scoring turn that still looks like the person holding the phone
        answering the bot. See GRAY_ZONE."""
        why = []
        if typical:
            r = loud / typical
            why.append(f"level {r:.2f}x")
            if r < GRAY_LOUD_RATIO:
                return False, ", ".join(why)
        else:
            why.append("level n/a")
        if not self._bot_seen or self._bot_end_idx <= 0 or self._bot_speaking:
            return False, ", ".join(why + ["no bot line to answer"])
        since = (first_idx - self._bot_end_idx) * self._hop_s
        why.append(f"{since:.1f}s after the bot")
        if not (0.0 <= since <= GRAY_ANSWER_S):
            return False, ", ".join(why)
        f0 = _pitch_hz(pcm, self.sample_rate)
        if f0 is not None and self._cust_f0:
            ref = float(np.median(self._cust_f0))
            st = abs(12.0 * np.log2(f0 / ref))
            why.append(f"pitch {f0:.0f}/{ref:.0f}Hz")
            if st > SHORT_PITCH_DIFF_ST:
                return False, ", ".join(why)
        return True, ", ".join(why)

    def _rival_learn(self, emb: Optional[np.ndarray], secs: float, why: str) -> None:
        """Remember the voice print of audio that was judged NOT the customer."""
        if RIVAL == "off" or emb is None or self._profile is None or secs < RIVAL_LEARN_MIN_S:
            return
        if float(np.dot(emb, self._profile)) >= RIVAL_LEARN_MAX_CUST_SIM:
            return                                   # too close to the customer to be sure
        e = emb.astype(np.float32)
        if self._rivals:
            sims = [float(np.dot(e, r)) for r in self._rivals]
            k = int(np.argmax(sims))
            if sims[k] >= RIVAL_MERGE_SIM:           # the same other person again
                self._rivals[k] = _norm(self._rivals[k] + e)
                return
        self._rivals.append(e)
        del self._rivals[:-RIVAL_MAX]
        self.stats["rival_learned"] += 1
        logger.info(f"[{self.call_id}] 👤 another voice remembered ({why}, {secs:.1f}s) — "
                    f"{len(self._rivals)} known other voice(s)")

    def _rival_match(self, emb: Optional[np.ndarray]) -> float:
        if emb is None or not self._rivals:
            return -1.0
        return max(float(np.dot(emb, r)) for r in self._rivals)

    def _rival_purge(self, emb: Optional[np.ndarray]) -> None:
        """A clearly-verified customer turn matches a 'rival' → it was the customer."""
        if emb is None or not self._rivals:
            return
        keep = [r for r in self._rivals if float(np.dot(emb, r)) < RIVAL_PURGE_SIM]
        if len(keep) != len(self._rivals):
            logger.warning(f"[{self.call_id}] 👤 {len(self._rivals) - len(keep)} remembered "
                           f"'other' voice(s) match the customer — forgotten")
            self._rivals = keep

    def _answers_bot_loudly(self, loud: float, typical: Optional[float],
                            first_idx: int, dur: float) -> bool:
        """See FAST_RELOCK."""
        if dur < FAST_RELOCK_MIN_S:
            return False
        ref = typical or (max(self._cust_rms) if self._cust_rms else None)
        if not ref or loud < FAST_RELOCK_LOUD_RATIO * ref:
            return False
        if not self._bot_seen or self._bot_end_idx <= 0 or self._bot_speaking:
            return False
        since = (first_idx - self._bot_end_idx) * self._hop_s
        return 0.0 <= since <= GRAY_ANSWER_S

    def _foreign_add(self, pcm: np.ndarray) -> None:
        self._foreign_turns.append(pcm)
        self._foreign_aids.append(self._answer_id)
        del self._foreign_aids[:-50]

    def _foreign_answers_distinct(self) -> int:
        """How many DIFFERENT bot lines the current run of foreign turns answered.
        Someone talking on and on beside the caller is one 'answer', not three —
        only a new person who keeps answering the bot takes the lock over. Without
        bot events every turn counts."""
        n = len(self._foreign_turns)
        if n == 0:
            return 0
        if not self._bot_seen:
            return n
        return len(set(self._foreign_aids[-n:]))

    # ── for the agent: who may interrupt the bot ───────────────────────────
    @property
    def bot_speaking(self) -> bool:
        return self._bot_speaking

    def customer_active(self, within_s: float = CUSTOMER_ACTIVE_S) -> bool:
        """True if the customer's own voice was passed (while the bot was
        speaking) within the last `within_s` seconds of audio."""
        return (self._next_idx - self._last_cust_pass_idx) * self._hop_s <= within_s

    def vad_should_mute(self) -> bool:
        """The agent should feed its VAD silence right now: the bot is speaking,
        the customer is locked, and the customer is not the one talking."""
        return (self.state is LockState.LOCKED and self._bot_speaking
                and not self.customer_active())

    def _relock_from(self, utts: List[np.ndarray]) -> None:
        self._relocks += 1
        self.state = LockState.ENROLLING
        self._profile = None
        self._bank = []
        self._cust_rms.clear()
        self._attempts = 0
        self._best = None
        self._probation_turns = []
        self._loud_foreign = []
        self._rivals = []
        self._utts = [[u] for u in utts]
        self._utt_aids = list(range(-len(utts), 0))   # separate turns
        self._enroll_speech_s = sum(len(u) for u in utts) / self.sample_rate
        self._maybe_build()

    def _probation_turn(self, pcm, sim, emb, dur, loud, quiet, loud_txt,
                        mixed: bool = False) -> Tuple[str, float]:
        """A long turn while the lock is not yet confirmed. Never dropped."""
        if sim >= PROBATION_PROVE_SIM and dur >= PROBATION_PROVE_S:
            self._proven = True
            self._probation_turns = []
            self._foreign_turns = []
            self._verified_since_lock += 1
            self.stats["turn_customer"] += 1
            self._cust_rms.append(loud)
            if emb is not None and sim >= BANK_ADD_TURN_SIM and not mixed:
                self._bank_add(emb)
            f0 = _pitch_hz(pcm, self.sample_rate)
            if f0 is not None:
                self._cust_f0.append(f0)
            logger.info(f"[{self.call_id}] ✅ lock confirmed — this turn clearly matches it "
                        f"(sim={sim:.2f}, {dur:.1f}s, {loud_txt}); normal filtering from now on")
            return "customer", sim
        self.stats["probation_pass"] += 1
        if not quiet:
            self._probation_turns.append(pcm)
        if sim < PROBATION_MISMATCH_SIM and not quiet:
            self._foreign_add(pcm)
        else:
            self._foreign_turns = []
        logger.warning(f"[{self.call_id}] 🟡 probation: turn passed (sim={sim:.2f}, {dur:.1f}s, "
                       f"{loud_txt}) — lock not confirmed yet"
                       f"{f', mismatch {len(self._foreign_turns)}/{TURN_RELOCK_COUNT_UNPROVEN}' if self._foreign_turns else ''}")
        if self._foreign_answers_distinct() >= TURN_RELOCK_COUNT_UNPROVEN:
            utts = self._foreign_turns
            self._foreign_turns = []
            logger.error(f"[{self.call_id}] 🔁 the new lock doesn't match the caller "
                         f"({len(utts)} turns in a row, last sim={sim:.2f}) — re-identifying, "
                         f"nothing was dropped")
            self._relock_from(utts)
        elif len(self._probation_turns) >= PROBATION_MAX_TURNS:
            utts = self._probation_turns
            logger.error(f"[{self.call_id}] 🔁 lock never confirmed after {len(utts)} turns — "
                         f"rebuilding it from the voice that is actually talking to the bot")
            self._relock_from(utts)
        return "customer", sim

    def _short_turn_score(self, pcm: np.ndarray, sim: Optional[float], loud: float,
                          typical: Optional[float], first_idx: int) -> Tuple[int, str]:
        score, why = 0, []
        if sim is not None:
            d = 2 if sim >= 0.40 else -2 if sim < SHORT_TURN_REJECT_SIM else -1 if sim < 0.15 else 0
            score += d
            why.append(f"voice {sim:.2f}({d:+d})")
        if typical:
            r = loud / typical
            d = 1 if r >= 0.6 else -3 if r < 0.2 else -2 if r < QUIET_RATIO else 0
            score += d
            why.append(f"level {r:.2f}x({d:+d})")
        f0 = _pitch_hz(pcm, self.sample_rate)
        if f0 is not None and self._cust_f0:
            ref = float(np.median(self._cust_f0))
            st = abs(12.0 * np.log2(f0 / ref))
            d = (1 if st <= SHORT_PITCH_SAME_ST else -3 if st > SHORT_PITCH_FAR_ST
                 else -2 if st > SHORT_PITCH_DIFF_ST else 0)
            score += d
            why.append(f"pitch {f0:.0f}/{ref:.0f}Hz({d:+d})")
        if self._bot_seen and self._bot_end_idx > 0:
            since = (first_idx - self._bot_end_idx) * self._hop_s
            if POST_BOT_GRACE_S <= since <= 4.0:
                score += 1
                why.append(f"answer {since:.1f}s after bot(+1)")
        self.stats["short_scored"] += 1
        return score, ", ".join(why) or "no clues"

    def take_turn_raw(self) -> np.ndarray:
        """The caller's latest answer exactly as it came off the phone line
        (int16, self.sample_rate), then forgotten. Empty if nothing was said."""
        with self._mutex:
            pcm = (np.concatenate(list(self._raw_turn)) if self._raw_turn
                   else np.zeros(0, dtype=np.int16))
            self._raw_turn.clear()
            self._raw_started = False
            return pcm

    def _turn_pieces(self, audio: List[Tuple[int, np.ndarray]]) -> List[Tuple[int, int, np.ndarray]]:
        """The turn cut at pauses (long stretches at SEG_MAX_S); a too-short piece
        joins the one before. Returns (first hop idx, last hop idx, pcm)."""
        min_h = max(1, int(round(SEG_MIN_S / self._hop_s)))
        max_h = max(min_h, int(round(SEG_MAX_S / self._hop_s)))
        pieces: List[List[Tuple[int, np.ndarray]]] = []
        cur: List[Tuple[int, np.ndarray]] = []
        last = None
        for idx, p in audio:
            if cur and (idx - last > RUN_GAP_HOPS + 1 or len(cur) >= max_h):
                pieces.append(cur)
                cur = []
            cur.append((idx, p))
            last = idx
        if cur:
            pieces.append(cur)
        merged: List[List[Tuple[int, np.ndarray]]] = []
        for pc in pieces:
            if merged and len(merged[-1]) < min_h:
                merged[-1] = merged[-1] + pc
            else:
                merged.append(pc)
        if len(merged) > 1 and len(merged[-1]) < min_h:
            tail = merged.pop()
            merged[-1] = merged[-1] + tail
        return [(pc[0][0], pc[-1][0], np.concatenate([x for _, x in pc]))
                for pc in merged if len(pc) >= min_h]

    def _piece_scores(self, audio: List[Tuple[int, np.ndarray]]) -> List[Dict]:
        """Voice print + pitch + level of each piece (last MIXED_MAX_PIECES).
        Expensive (one voice print per piece) — call OUTSIDE the mutex."""
        pcs = self._turn_pieces(audio)
        if len(pcs) < 2:
            return []
        t0 = audio[0][0]
        out = []
        for a, b, pcm in pcs[-MIXED_MAX_PIECES:]:
            sim, _pemb = self._similarity(pcm)
            out.append({"t0": (a - t0) * self._hop_s, "t1": (b - t0 + 1) * self._hop_s,
                        "sim": sim, "f0": _pitch_hz(pcm, self.sample_rate), "rms": _rms(pcm),
                        "pcm": pcm, "emb": _pemb})
        return out

    def _best_part_sim(self, audio: List[Tuple[int, np.ndarray]]) -> Optional[float]:
        sims = [p["sim"] for p in self._piece_scores(audio) if p["sim"] is not None]
        return max(sims) if sims else None

    def _mixed_analysis(self, pieces: List[Dict], typical: Optional[float]) -> Dict:
        """Did the customer AND someone else speak in this turn?"""
        info: Dict = {"mixed": False}
        scored = [p for p in pieces if p["sim"] is not None]
        if len(scored) < 2:
            return info
        ref_f0 = float(np.median(self._cust_f0)) if self._cust_f0 else None
        cust_s = other_s = 0.0
        parts = []
        _keep = []                          # [fix 18] audio of the pieces that are not OTHER
        n_cust = n_other = 0
        for p in scored:
            sim, f0 = p["sim"], p["f0"]
            far = (ref_f0 is not None and f0 is not None
                   and abs(12.0 * np.log2(f0 / ref_f0)) > SHORT_PITCH_DIFF_ST)
            quiet = typical is not None and p["rms"] < QUIET_RATIO * typical
            if sim >= MIXED_CUST_SIM:
                who = "customer"
                n_cust += 1
                cust_s += p["t1"] - p["t0"]
            elif sim < MIXED_OTHER_SIM or (sim < MIXED_WEAK_SIM and (far or quiet)):
                who = "OTHER"
                n_other += 1
                other_s += p["t1"] - p["t0"]
            else:
                who = "unsure"
            if who != "OTHER" and p.get("pcm") is not None:
                _keep.append(p["pcm"])
            if who == "OTHER" and not quiet and self._proven:
                self._rival_learn(p.get("emb"), p["t1"] - p["t0"], "other voice inside a turn")
            extra = f", {f0:.0f}Hz" if f0 is not None else ""
            parts.append(f"{p['t0']:.1f}-{p['t1']:.1f}s {who} ({sim:.2f}{extra})")
        info.update({
            "mixed": n_cust > 0 and n_other > 0,
            "customer_s": round(cust_s, 1),
            "other_s": round(other_s, 1),
            "timeline": " | ".join(parts),
            "customer_pitch": round(ref_f0) if ref_f0 else None,
        })
        if info["mixed"] and _keep:
            info["customer_pcm"] = np.concatenate(_keep)     # [fix 18]
        return info

    def close(self) -> None:
        logger.info(self.summary())
        with self._mutex:
            for w in (self._wav_in, self._wav_out):
                if w is not None:
                    try:
                        w.close()
                    except Exception:
                        pass
            if self._wav_in is not None:
                logger.info(f"[{self.call_id}] 🎧 debug audio saved: "
                            f"{os.path.join(DEBUG_WAV_DIR, self.call_id)}_in.wav / _out.wav")
            self._wav_in = self._wav_out = None

    # ── streaming internals ───────────────────────────────────────────────
    def _get(self, idx: int) -> Optional[_Hop]:
        if not self._ring:
            return None
        k = idx - self._ring[0].idx
        return self._ring[k] if 0 <= k < len(self._ring) else None

    def _span(self, lo: int, hi: int) -> np.ndarray:
        parts = [h.pcm for h in self._ring if lo <= h.idx <= hi]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)

    def _voice_span(self, center: int, back: int = WINDOW_BACK_HOPS) -> np.ndarray:
        """Speech hops around `center`, from the SAME sentence only: never across a
        pause, and never before the point where another voice took over."""
        def walk(step: int, limit: int) -> List[_Hop]:
            got, gap, i = [], 0, center + step
            while abs(i - center) <= limit:
                if step < 0 and i < self._run_start_idx:
                    break
                h = self._get(i)
                if h is None:
                    break
                if h.speech:
                    got.append(h)
                    gap = 0
                else:
                    gap += 1
                    if gap > RUN_GAP_HOPS:
                        break
                i += step
            return got
        me = self._get(center)
        back_h = list(reversed(walk(-1, back)))
        fwd = walk(+1, LOOKAHEAD_HOPS)
        parts = [h.pcm for h in back_h] + ([me.pcm] if me is not None else []) + [h.pcm for h in fwd]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)

    def _push(self, raw: np.ndarray) -> None:
        rms = _rms(raw)
        self._rms_hist.append(rms)
        self._bg = max(5.0, float(np.percentile(self._rms_hist, BG_PERCENTILE)))
        speech = (rms >= max(ABS_SPEECH_RMS, SPEECH_OVER_BG * self._bg)
                  and SFM_MIN <= _sfm(raw) <= SFM_MAX)
        hop = _Hop(self._next_idx, raw, rms, speech, speech and self._is_answer_time(),
                   self._bot_speaking)
        self._line_hist.append((rms, bool(speech and not self._bot_speaking)))
        if speech and not self._bot_speaking:
            self._line_speech_total += 1
            self._last_line_speech_idx = self._next_idx
        self._ring.append(hop)
        self._next_idx += 1
        if BGM_GUARD and self._bot_speaking and self.state is LockState.ENROLLING:
            _st = getattr(self, "_bgm_bot_start", None)
            if _st is not None and hop.idx - _st >= BGM_SKIP_HOPS:
                self._bgm_bot_hops = getattr(self, "_bgm_bot_hops", 0) + 1
                if speech:
                    if getattr(self, "_bgm_buf", None) is None:
                        self._bgm_buf = collections.deque(
                            maxlen=max(10, int(BGM_BUFFER_S / self._hop_s)))
                    self._bgm_buf.append(raw)
                    self._bgm_speech_hops = getattr(self, "_bgm_speech_hops", 0) + 1
        if speech and not self._raw_started:
            self._raw_started = True
            self._raw_turn.extend(self._raw_preroll)
        if self._raw_started:
            self._raw_turn.append(raw)
        else:
            self._raw_preroll.append(raw)
        if self.state is LockState.ENROLLING:
            self._enroll_observe(hop)
        if self._next_idx % STATS_EVERY_HOPS == 0:
            logger.info(self.summary())

    def _release(self, hop: _Hop) -> np.ndarray:
        if self.state is not LockState.LOCKED:
            return self._shape(hop.pcm, hop.pcm)
        if MODE == "verify" and not hop.bot:
            # bot is silent: the caller's words go to STT untouched; the turn is
            # verified as a whole in judge_turn()
            if hop.speech:
                self._turn_audio.append((hop.idx, hop.pcm))
            self.stats["raw_turn"] += 1
            return self._shape(hop.pcm, hop.pcm)
        _bf_other = self.stats["other"]
        try:
            seg = self._locked_decision(hop)
        except Exception as e:
            now = time.monotonic()
            if now - self._last_err_ts > 5.0:
                self._last_err_ts = now
                logger.error(f"[{self.call_id}] voice lock decision error (hop silenced): {e}")
            seg = None
        if seg is not None:
            self._last_cust_pass_idx = hop.idx
        if BACKFILL:
            try:
                seg = self._backfill(hop, seg, _bf_other)
            except Exception as e:
                logger.error(f"[{self.call_id}] back-fill error (ignored): {e}")
        if hop.answer:
            self._answer_hist.append((seg is not None, hop.pcm))
            self._check_relock()
        return self._shape(hop.pcm, seg)

    def _backfill(self, hop: _Hop, seg: Optional[np.ndarray],
                  other_before: int) -> Optional[np.ndarray]:
        """[fix 25] See BACKFILL. Returns seg, possibly with the blocked start of
        this sentence in front of it."""
        buf = getattr(self, "_bf", None)
        if buf is None:
            buf = self._bf = []
            self._bf_run = None
        run = getattr(self, "_run_start_idx", None)
        if self._bf_run != run:                     # a new sentence
            buf.clear()
            self._bf_run = run
        if seg is None:
            if self.stats["other"] != other_before or getattr(self, "_run_foreign", False):
                buf.clear()                         # clearly somebody else: keep nothing
            elif hop.speech:
                buf.append((hop.idx, hop.pcm))      # unsure: keep it for now
                del buf[:-BACKFILL_MAX_HOPS]
            return None
        if (buf and seg is hop.pcm and self._run_open
                and hop.idx - buf[-1][0] <= RUN_GAP_HOPS + 1):
            n = len(buf)
            self.stats["backfill"] += n
            if not getattr(self, "_bf_logged", False) or n >= 5:
                self._bf_logged = True
                logger.info(f"[{self.call_id}] ⏪ barge-in start restored: {n * self._hop_s:.1f}s "
                            f"spoken over the bot is sent to STT with the rest of the sentence")
            seg = np.concatenate([p for _, p in buf] + [hop.pcm])
        buf.clear()
        return seg

    def _check_relock(self) -> None:
        n = len(self._answer_hist)
        if n * self._hop_s < RELOCK_SPEECH_S:
            return
        passed = sum(1 for ok, _ in self._answer_hist if ok)
        if passed / n > RELOCK_PASS_MAX:
            return
        rejected = [pcm for ok, pcm in self._answer_hist if not ok]
        self._answer_hist.clear()

        # The customer is answering the bot and we're blocking them. First
        # question: is this still THEIR voice, just sounding different right now?
        sim, emb = self._similarity(np.concatenate(rejected[-40:]))
        if sim is not None and sim >= WIDEN_MIN_SIM:
            self._bank_add(emb)
            self.stats["widened"] += 1
            self._run_open = True       # let the rest of this sentence through
            logger.warning(f"[{self.call_id}] 🔧 Answer was being blocked but matches the "
                           f"customer (sim={sim:.2f}) — voice profile widened "
                           f"(bank={len(self._bank)}), rest of the answer passes")
            return

        if sim is not None and sim < FOREIGN_SIM:
            self._foreign_answers += 1
            if self._foreign_answers < FOREIGN_ANSWERS_TO_RELOCK:
                self.stats["foreign_blocked"] += 1
                logger.warning(f"[{self.call_id}] 🚫 A different voice answered (sim={sim:.2f}) "
                               f"— kept blocked ({self._foreign_answers}/"
                               f"{FOREIGN_ANSWERS_TO_RELOCK} before assuming the phone changed hands)")
                return
        self._foreign_answers = 0

        self._relocks += 1
        if self._relocks > RELOCK_MAX:
            self.state = LockState.FAILED_OPEN
            logger.error(f"[{self.call_id}] ❌ Caller answers still rejected after "
                         f"{RELOCK_MAX} re-locks — audio passes unfiltered so the call isn't deaf")
            return
        logger.error(f"[{self.call_id}] 🔁 WRONG LOCK: the voice answering the bot does not "
                     f"match (sim={sim if sim is not None else float('nan'):.2f}, "
                     f"{passed}/{n} hops passed) — re-identifying the customer "
                     f"(relock {self._relocks}/{RELOCK_MAX})")
        self.state = LockState.ENROLLING
        self._profile = None
        self._bank = []
        self._tse_cache = None
        self._rivals = []
        self._attempts = 0
        self._best = None
        self._utts = [rejected]
        self._utt_aids = [self._answer_id]
        self._enroll_speech_s = len(rejected) * self._hop_s
        self._last_enroll_idx = self._next_idx
        self._prev_open = True
        self._maybe_build()

    def _learn_from_answer(self, hist: List[Tuple[bool, np.ndarray]]) -> None:
        """A finished answer that clearly got through is the customer's voice on
        this line right now — add it to the profile bank."""
        if self.state is not LockState.LOCKED or not hist:
            return
        ok = [pcm for good, pcm in hist if good]
        if len(ok) / len(hist) >= 0.5:
            self._foreign_answers = 0        # the customer answered normally
        if (len(ok) / len(hist) < BANK_ADD_PASS_RATIO
                or len(ok) * self._hop_s < BANK_ADD_MIN_S):
            return
        sim, emb = self._similarity(np.concatenate(ok[-40:]))
        if emb is not None and sim is not None and sim >= max(self._thr, BANK_ADD_TURN_SIM):
            self._bank_add(emb)
            logger.info(f"[{self.call_id}] ➕ answer added to voice profile "
                        f"(sim={sim:.2f}, bank={len(self._bank)})")

    def _bank_agreement(self) -> np.ndarray:
        B = np.vstack(self._bank)
        S = B @ B.T
        n = len(self._bank)
        return (S.sum(axis=1) - np.diag(S)) / max(1, n - 1)

    def _bank_add(self, emb: np.ndarray) -> None:
        self._bank.append(emb.astype(np.float32))
        if len(self._bank) >= BANK_CLEAN_MIN:
            agree = self._bank_agreement()
            worst = int(np.argmin(agree))
            others = np.delete(agree, worst)
            if agree[worst] < BANK_OUTLIER_SIM or agree[worst] < float(np.median(others)) - BANK_OUTLIER_MARGIN:
                self._bank.pop(worst)
                self.stats["bank_cleaned"] += 1
                logger.warning(f"[{self.call_id}] 🧽 dropped a profile entry that doesn't match the "
                               f"customer's other answers (agreement {agree[worst]:.2f} vs "
                               f"{float(np.median(others)):.2f}{', the enrollment print' if worst == 0 else ''})")
        if len(self._bank) > BANK_MAX:
            self._bank.pop(int(np.argmin(self._bank_agreement())))   # least typical goes
        self._profile = _norm(np.mean(np.vstack(self._bank), axis=0))

    def _shape(self, raw: np.ndarray, seg: Optional[np.ndarray]) -> np.ndarray:
        """Apply the gate with short fades so edges never click."""
        f = min(self._fade, len(raw))
        if seg is None:
            out = np.zeros(len(raw), dtype=np.int16)
            if self._prev_open:   # fade the tail out instead of a hard cut
                ramp = np.linspace(1.0, 0.0, f, dtype=np.float32)
                out[:f] = (raw[:f].astype(np.float32) * ramp).astype(np.int16)
            self._prev_open = False
            return out
        out = np.asarray(seg, dtype=np.int16).copy()
        if not self._prev_open:
            ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
            out[:f] = (out[:f].astype(np.float32) * ramp).astype(np.int16)
        self._prev_open = True
        return out

    # ── LOCKED ────────────────────────────────────────────────────────────
    def _is_candidate(self, hop: _Hop) -> bool:
        if hop.speech:
            return True
        # quiet consonant just before / after a speech hop: judge it too, so word
        # onsets and tails aren't chopped off.
        if hop.rms < max(ABS_SPEECH_RMS * 0.5, 1.1 * self._bg):
            return False
        for n in (hop.idx - 1, hop.idx + 1):
            nb = self._get(n)
            if nb is not None and nb.speech:
                return True
        return False

    def _similarity(self, pcm: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
        if pcm is None or len(pcm) == 0:
            return None, None
        emb = self._embed(pcm, self.sample_rate)
        if emb is None or self._profile is None:
            return None, None
        sim = float(np.dot(emb, self._profile))
        # While the profile is young, the best single entry may be used (helps the
        # caller when they sound different). Once it has learned from several
        # answers, ONLY the averaged profile counts: "best single entry" let anyone
        # resembling a part-contaminated enrollment print through
        # (V9251606030000040818) — in tests the gap between caller and bystander
        # widened from 0.16 to 0.56 with no loss for the caller.
        if self._bank and len(self._bank) < MATURE_BANK:
            sim = max(sim, max(float(np.dot(emb, b)) for b in self._bank))
        return sim, emb

    def _locked_decision(self, hop: _Hop) -> Optional[np.ndarray]:
        if not self._is_candidate(hop):
            self.stats["silence"] += 1
            return None

        if hop.idx - self._last_cand_idx > RUN_GAP_HOPS:     # a new sentence
            self._run_open = False
            self._run_start_idx = hop.idx
            self._switch_streak = 0
            self._run_foreign = False
            self._last_eval = None
        self._last_cand_idx = hop.idx

        # same sentence, decided very recently → reuse that decision (the voice
        # print is the expensive part, and 100 ms barely changes it)
        le = self._last_eval
        if le is not None and 0 < hop.idx - le[0] < LOCKED_EVAL_EVERY_HOPS:
            self.stats["reused"] += 1
            return hop.pcm if le[1] else None
        seg = self._locked_decision_eval(hop)
        self._last_eval = ((hop.idx, seg is hop.pcm)
                           if (seg is None or seg is hop.pcm) else None)
        return seg

    def _locked_decision_eval(self, hop: _Hop) -> Optional[np.ndarray]:

        # judged on the sentence so far (steady), not on this 100 ms alone
        sim, _ = self._similarity(self._voice_span(hop.idx, LONG_BACK_HOPS))
        if sim is None:
            self.stats["silence"] += 1
            return None
        self._last_sim = sim
        self._sims.append(sim)

        # someone else cutting in mid-sentence: the short window notices it
        if self._run_open:
            short, _ = self._similarity(self._voice_span(hop.idx, WINDOW_BACK_HOPS))
            if short is not None and short < HARD_REJECT_COSINE:
                self._switch_streak += 1
            else:
                self._switch_streak = 0
            if self._switch_streak >= SWITCH_HOPS:
                self._run_open = False
                self._run_start_idx = hop.idx        # new voice → new sentence
                self._switch_streak = 0
                self._run_foreign = True
                self.stats["switch"] += 1
                self.stats["other"] += 1
                return None

        if sim >= self._thr_clean:                    # clearly the customer
            self.stats["pass"] += 1
            self._run_open = True
            return hop.pcm

        if sim >= self._thr:                          # the customer, maybe with something under it
            # passed RAW — the separation model changes words on a phone line
            self.stats["soft_pass"] += 1
            self._run_open = True
            return hop.pcm

        if self._run_open and sim >= KEEP_OPEN_SIM:   # same sentence, weaker patch
            self.stats["hold_open"] += 1
            return hop.pcm

        self._run_open = False
        if sim < HARD_REJECT_COSINE:                  # clearly somebody else
            self.stats["other"] += 1
            return None

        if ANSWER_LENIENT and hop.answer and not self._run_foreign:
            # the customer answering the bot, on a weak-scoring stretch: keep the words
            self.stats["answer_pass"] += 1
            return hop.pcm

        seg = self._tse(hop)                          # overlap / unsure → extract customer
        if seg is not None:
            return seg
        self.stats["uncertain"] += 1
        return hop.pcm if UNCERTAIN_POLICY == "pass" else None

    def _tse(self, hop: _Hop) -> Optional[np.ndarray]:
        if self._extractor is None or not self._extractor_ok or self._tse_ref is None:
            return None
        c = self._tse_cache
        if c is None or not (c[0] <= hop.idx <= c[1]):
            newest = self._next_idx - 1
            start = max(self._ring[0].idx, newest - TSE_WINDOW_HOPS + 1)
            mix = self._span(start, newest)
            y = None
            try:
                t0 = time.monotonic()
                y = self._extractor.extract(mix, self._tse_ref)
                self.stats["tse_ms"] = int((time.monotonic() - t0) * 1000)
            except Exception as e:
                logger.warning(f"[{self.call_id}] TSE error: {e}")
            # target_extractor returns the INPUT object when it gives up —
            # that is the mixture, i.e. exactly what must not pass.
            if y is None or y is mix or len(y) < len(mix):
                y = None
            else:
                y = np.asarray(y[:len(mix)], dtype=np.float32)
                target = TSE_OUTPUT_LEVEL * _rms(mix)
                cur = _rms(y)
                if cur > 1e-6:
                    y = y * (target / cur)
                y = np.clip(y, -32768, 32767).astype(np.int16)
            self._tse_cache = (start, newest, y)
            c = self._tse_cache

        start, end, y = c
        if y is None:
            self.stats["tse_fail"] += 1
            return None
        off = (hop.idx - start) * self.hop_len
        seg = y[off:off + self.hop_len]
        if len(seg) != self.hop_len:
            self.stats["tse_fail"] += 1
            return None

        lo = max(start, hop.idx - WINDOW_BACK_HOPS)
        hi = min(end, hop.idx + LOOKAHEAD_HOPS)
        check = y[(lo - start) * self.hop_len:(hi - start + 1) * self.hop_len]
        sim, _ = self._similarity(check)
        if sim is None or sim < self._thr - TSE_ACCEPT_DROP:
            self.stats["tse_fail"] += 1
            return None
        self.stats["tse_ok"] += 1
        return seg

    def _adapt(self, emb: Optional[np.ndarray], sim: float) -> None:
        if not PROFILE_ADAPT or emb is None or sim < self._thr_clean + 0.05:
            return
        cand = _norm((1.0 - PROFILE_ADAPT_ALPHA) * self._profile + PROFILE_ADAPT_ALPHA * emb)
        if float(np.dot(cand, self._anchor)) >= PROFILE_ANCHOR_MIN:
            self._profile = cand

    # ── ENROLLING ─────────────────────────────────────────────────────────
    def _enroll_observe(self, hop: _Hop) -> None:
        # First enrollment: only the voice that ANSWERS the bot. After a re-lock
        # that rule can deadlock — V9251438240000040543 re-locked, then the people
        # around the caller kept interrupting the bot so it never finished a line,
        # no speech ever counted as an "answer", and the lock stayed open for the
        # rest of the call. So a RE-enrollment takes any speech heard while the bot
        # is quiet (SepFormer still picks the near-mic voice if several are mixed).
        if not self._enroll_open:
            return
        if not hop.answer:
            if not (self._relocks > 0 and hop.speech and not self._bot_speaking):
                return
        if (self._utts and hop.idx - self._last_enroll_idx <= ENROLL_UTT_GAP_HOPS
                and (not self._utt_aids or self._utt_aids[-1] == self._answer_id)):
            self._utts[-1].append(hop.pcm)
        else:
            self._utts.append([hop.pcm])
            self._utt_aids.append(self._answer_id)
        self._last_enroll_idx = hop.idx
        self._enroll_speech_s += self._hop_s
        while self._enroll_speech_s > ENROLL_BUFFER_MAX_S and len(self._utts) > 1:
            dropped = self._utts.pop(0)
            if self._utt_aids:
                self._utt_aids.pop(0)
            self._enroll_speech_s -= len(dropped) * self._hop_s
        self._maybe_build()

    def _loud_utts(self) -> Tuple[List[List[np.ndarray]], float]:
        """The enrollment answers with their quiet slices left out (ENROLL_LOUD),
        and how many seconds are left. An answer of whole turns (after a re-lock)
        is kept as it is."""
        hop = self.hop_len
        levels = [_rms(h) for u in self._utts for h in u if len(h) == hop]
        if not ENROLL_LOUD or len(levels) < 5:
            return self._utts, self._enroll_speech_s
        floor = ENROLL_LOUD_RATIO * float(np.percentile(levels, ENROLL_LOUD_PERCENTILE))
        out: List[List[np.ndarray]] = []
        n = 0
        for u in self._utts:
            keep = [h for h in u if len(h) != hop or _rms(h) >= floor]
            n += sum(len(h) for h in keep)
            out.append(keep)
        return out, n / self.sample_rate

    def _maybe_build(self) -> None:
        if self._building or self.state is not LockState.ENROLLING:
            return
        need = max(ENROLL_MIN_SECONDS + self._attempts * ENROLL_RETRY_EXTRA_S, self._wait_until_s)
        if self._enroll_speech_s < need:
            return
        loud_utts, loud_s = self._loud_utts()
        # _wait_until_s is measured on everything heard; the loud part must reach
        # the plain minimum
        if loud_s < ENROLL_MIN_SECONDS + self._attempts * ENROLL_RETRY_EXTRA_S:
            return
        if (ENROLL_MIN_WORDS > 0 and self._relocks == 0 and self._enroll_words_seen
                and self._enroll_words < ENROLL_MIN_WORDS):
            if not self._words_wait_logged:
                self._words_wait_logged = True
                logger.info(f"[{self.call_id}] enrollment: {loud_s:.1f}s of sound but only "
                            f"{self._enroll_words} word(s) heard — waiting for "
                            f"{ENROLL_MIN_WORDS} before trusting it as the caller's voice")
            return
        aids = [a for lu, a in zip(loud_utts, self._utt_aids) if lu]
        if (ENROLL_TWO_ANSWERS and self._relocks == 0 and len(set(aids)) < 2
                and loud_s < ENROLL_SINGLE_ANSWER_S):
            if not self._two_answer_logged:
                self._two_answer_logged = True
                logger.info(f"[{self.call_id}] enrollment: {loud_s:.1f}s heard, all "
                            f"in ONE answer — waiting for a second answer to confirm the voice")
            return
        # aids were taken from the full answers; keep each with its loud part
        pairs = [(np.concatenate(lu), a)
                 for u, lu, a in zip(self._utts, loud_utts, self._utt_aids) if u and lu]
        if len(self._utt_aids) != len(self._utts):
            pairs = [(np.concatenate(lu), i) for i, lu in enumerate(loud_utts) if lu]
        utts = [x for x, _ in pairs]
        aids = [a for _, a in pairs]
        if not utts:
            return
        self._building = True
        threading.Thread(target=self._build_worker, args=(utts, aids),
                         name=f"voicelock-enroll-{self.call_id}", daemon=True).start()

    def _build_worker(self, utts: List[np.ndarray], aids: Optional[List[int]] = None) -> None:
        res = None
        self._need_more = False
        try:
            if self._extractor is not None:
                self._extractor_ok = bool(self._extractor.available())
            if BGM_GUARD and utts:
                utts, aids = self._bgm_filter(utts, aids)
            if aids is not None and utts:
                utts = self._agreeing_answers(utts, aids)
            res = self._build_profile(utts) if utts else None
        except Exception as e:
            logger.error(f"[{self.call_id}] enrollment build failed: {e}")
        with self._mutex:
            self._building = False
            if res is None and self._need_more:
                # not a failed attempt — just not enough speech yet; try again
                # after one more second of answers, not on every new hop
                self._wait_until_s = self._enroll_speech_s + 1.0
                return
            self._attempts += 1
            if res is not None and (self._best is None or res["p5"] > self._best["p5"]):
                self._best = res
            last = self._attempts >= ENROLL_MAX_ATTEMPTS
            if res is not None and res["p5"] >= MIN_CALLER_P5:
                self._lock_on(res)
            elif last and self._best is not None and self._best["p5"] >= MIN_CALLER_P5_LAST_TRY:
                self._lock_on(self._best)
            elif last:
                self.state = LockState.FAILED_OPEN
                self._utts = []
                self._utt_aids = []
                logger.error(f"[{self.call_id}] ❌ Could not isolate one customer voice after "
                             f"{self._attempts} attempts (best p5="
                             f"{self._best['p5'] if self._best else 'n/a'}) — audio passes unfiltered")
            else:
                logger.warning(f"[{self.call_id}] ⚠️ Enrollment attempt {self._attempts} not clean "
                               f"(p5={res['p5'] if res else 'n/a'}) — collecting more speech")

    def _bgm_filter(self, utts: List[np.ndarray], aids: Optional[List[int]]):
        """[fix 28] Drop the answers spoken by a voice that kept talking over the bot."""
        if getattr(self, "_bgm_off", False):
            return utts, aids
        sp = getattr(self, "_bgm_speech_hops", 0)
        bh = getattr(self, "_bgm_bot_hops", 0)
        buf = getattr(self, "_bgm_buf", None)
        if not buf or sp * self._hop_s < BGM_MIN_S or sp < BGM_MIN_FRACTION * max(1, bh):
            return utts, aids
        cached = getattr(self, "_bgm_prof_at", None)
        if cached is None or cached[0] != sp:
            embs, _ = self._embed_pieces([np.concatenate(list(buf))])
            prof = _norm(np.median(np.vstack(embs), axis=0)) if len(embs) >= 4 else None
            self._bgm_prof_at = (sp, prof)
        prof = self._bgm_prof_at[1]
        if prof is None:
            return utts, aids
        min_len = int(0.8 * self.sample_rate)
        piece = int(1.0 * self.sample_rate)
        keep, dropped, cut_s = [], [], 0.0
        utts = list(utts)
        for i, u in enumerate(utts):
            if len(u) < min_len:
                keep.append(i)                      # too short to tell ("హా", "Hello")
                continue
            # judged second by second: the caller's answer and the video often
            # run into each other without a pause
            parts = [u[j:j + piece] for j in range(0, len(u), piece)]
            if len(parts) > 1 and len(parts[-1]) < piece // 2:
                parts[-2] = np.concatenate([parts[-2], parts[-1]])
                parts.pop()
            good = []
            for part in parts:
                e = self._embed(part, self.sample_rate)
                if e is not None and float(np.dot(e, prof)) >= BGM_MATCH_SIM:
                    cut_s += len(part) / self.sample_rate
                else:
                    good.append(part)
            if len(good) == len(parts):
                keep.append(i)
            elif good and sum(len(g) for g in good) >= min_len // 2:
                utts[i] = np.concatenate(good)
                keep.append(i)
            else:
                dropped.append(i)
        if not dropped and cut_s == 0.0:
            return utts, aids
        now = time.monotonic()
        if not any(len(utts[i]) >= min_len for i in keep):
            first = getattr(self, "_bgm_wait_since", None)
            if first is None:
                self._bgm_wait_since = first = now
            if now - first > BGM_GIVE_UP_S:
                self._bgm_off = True
                logger.warning(f"[{self.call_id}] 📺 only the voice that talks over the bot was heard "
                               f"for {BGM_GIVE_UP_S:.0f}s — background guard off for this call")
                return utts, aids
        else:
            self._bgm_wait_since = None
        if getattr(self, "_bgm_logged", 0) < 3:
            self._bgm_logged = getattr(self, "_bgm_logged", 0) + 1
            logger.warning(f"[{self.call_id}] 📺 a voice kept talking while the bot was speaking "
                           f"({sp * self._hop_s:.1f}s, {100 * sp // max(1, bh)}% of the time) — "
                           f"TV / video, not the caller: {cut_s + sum(len(utts[i]) for i in dropped) / self.sample_rate:.1f}s "
                           f"of it left out of the voice profile")
        # forget those answers, so the caller's short answers are not pushed out of the buffer
        sig = {(len(utts[i]), utts[i][:64].tobytes()) for i in dropped}
        with self._mutex:
            nu, na = [], []
            for hops, a in zip(self._utts, self._utt_aids):
                if hops and (sum(len(h) for h in hops), hops[0][:64].tobytes()) in sig:
                    self._enroll_speech_s = max(0.0, self._enroll_speech_s - len(hops) * self._hop_s)
                    continue
                nu.append(hops)
                na.append(a)
            self._utts, self._utt_aids = nu, na
        utts2 = [utts[i] for i in keep]
        aids2 = [aids[i] for i in keep] if aids is not None and len(aids) == len(utts) else None
        if not utts2:
            self._need_more = True
        return utts2, aids2

    def _embed_pieces(self, pieces: List[np.ndarray]):
        step = max(1, self._win_len // 3)
        embs, wins = [], []
        for p in pieces:
            for w in _windows(p, self._win_len, step):
                e = self._embed(w, self.sample_rate)
                if e is not None:
                    embs.append(e)
                    wins.append(w)
        return embs, wins

    def _mean_sim(self, x: np.ndarray, prof: np.ndarray) -> float:
        embs, _ = self._embed_pieces([x])
        return float(np.mean([np.dot(e, prof) for e in embs])) if embs else -1.0

    def _agreeing_answers(self, utts: List[np.ndarray], aids: List[int]) -> List[np.ndarray]:
        """Keep only the answers that sound like the same person. Answers shorter
        than ANSWER_CHECK_MIN_S are kept unjudged. If two long answers disagree and
        there is no third to break the tie, wait for one (returns [] + need_more)."""
        groups: Dict[int, List[np.ndarray]] = {}
        for u, a in zip(utts, aids):
            groups.setdefault(a, []).append(u)
        if len(groups) < 2:
            return utts
        embs: Dict[int, np.ndarray] = {}
        for a, us in groups.items():
            x = np.concatenate(us)
            if len(x) >= ANSWER_CHECK_MIN_S * self.sample_rate:
                e = self._embed(x[-int(6 * self.sample_rate):], self.sample_rate)
                if e is not None:
                    embs[a] = e
        if len(embs) < 2:
            return utts
        keys = list(embs)
        agree = {a: [] for a in keys}
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                s = float(np.dot(embs[a], embs[b]))
                agree[a].append(s)
                agree[b].append(s)
        best = {a: max(v) for a, v in agree.items()}
        odd = [a for a in keys if best[a] < ANSWER_AGREE_SIM]
        if not odd:
            logger.info(f"[{self.call_id}] ✅ {len(keys)} answers sound like the same person "
                        f"(agreement {min(best.values()):.2f}–{max(best.values()):.2f})")
            return utts
        if len(odd) == len(keys):
            if len(keys) == 2 and self._enroll_speech_s < ENROLL_BUFFER_MAX_S - 1.0:
                logger.warning(f"[{self.call_id}] 👥 the two answers sound like DIFFERENT people "
                               f"(agreement {best[keys[0]]:.2f}) — waiting for a third answer")
                self._need_more = True
                return []
            return utts       # no two agree: let SepFormer sort it out
        logger.warning(f"[{self.call_id}] 👥 {len(odd)} answer(s) don't match the others "
                       f"(agreement {', '.join(f'{best[a]:.2f}' for a in odd)}) — left out of "
                       f"the voice profile")
        return [u for u, a in zip(utts, aids) if a not in odd]

    def _build_profile(self, utts: List[np.ndarray]) -> Optional[Dict]:
        # Short answers ("తెలుగు", "హా") are each shorter than one window, so window
        # the JOINED speech — otherwise a 3 s enrollment yields only ~3 windows.
        embs, wins = self._embed_pieces([np.concatenate(utts)])
        if len(embs) < ENROLL_MIN_WINDOWS:
            logger.info(f"[{self.call_id}] enrollment: only {len(embs)} windows, need "
                        f"{ENROLL_MIN_WINDOWS} — waiting for more speech")
            self._need_more = True
            return None
        E = np.vstack(embs)
        prof = _norm(np.median(E, axis=0))
        sims = E @ prof
        multi = float(np.mean(sims < MULTI_SPEAKER_SIM)) > MULTI_SPEAKER_FRACTION
        separated = False

        if multi and self._separator is not None and self._separator.available():
            logger.warning(f"[{self.call_id}] 👥 More than one voice in enrollment audio "
                           f"({np.mean(sims < MULTI_SPEAKER_SIM):.0%} of windows off-profile) "
                           f"— separating with SepFormer")
            max_len = int(SEPARATOR_PIECE_S * self.sample_rate)
            alts: List[List[np.ndarray]] = []
            for u in utts:
                for p in _split(u, max_len):
                    if len(p) < self._win_len // 2:
                        continue
                    streams = [np.asarray(s[:len(p)], dtype=np.int16)
                               for s in self._separator.separate(p)]
                    streams = [s for s in streams if len(s) == len(p)]
                    alts.append(streams or [p])
            # 1st pass: the near-mic (loudest) source is the one answering the bot
            pieces = [max(a, key=_rms) for a in alts]
            embs, wins = self._embed_pieces(pieces)
            if len(embs) < ENROLL_MIN_WINDOWS:
                # 3 windows locked V0011101050000043817 onto the wrong voice —
                # too few to trust. Wait for more answers instead.
                logger.info(f"[{self.call_id}] enrollment: only {len(embs)} separated windows, "
                            f"need {ENROLL_MIN_WINDOWS} — waiting for more speech")
                self._need_more = True
                return None
            prof = _norm(np.median(np.vstack(embs), axis=0))
            # 2nd pass: per utterance, the source that matches that voice best
            pieces = [max(a, key=lambda s: self._mean_sim(s, prof)) for a in alts]
            embs, wins = self._embed_pieces(pieces)
            if len(embs) < ENROLL_MIN_WINDOWS:
                self._need_more = True
                return None
            E = np.vstack(embs)
            prof = _norm(np.median(E, axis=0))
            separated = True

        keep = np.ones(len(E), dtype=bool)
        for _ in range(ENROLL_OUTLIER_ITERATIONS):
            s = E @ prof
            cutoff = max(float(np.percentile(s, ENROLL_OUTLIER_PERCENTILE)),
                         float(s.mean() - s.std()))
            k = s >= cutoff
            if k.sum() < 3:
                break
            keep = k
            prof = _norm(np.median(E[keep], axis=0))

        kept_sims = E[keep] @ prof
        p5 = float(np.percentile(kept_sims, 5))

        order = np.argsort(-(E @ prof))
        ref, total = [], 0
        for j in order:
            if not keep[j]:
                continue
            ref.append(wins[j])
            total += len(wins[j])
            if total >= TSE_REF_SECONDS * self.sample_rate:
                break
        return {"profile": prof, "p5": p5, "ref": np.concatenate(ref),
                "separated": separated, "windows": int(keep.sum())}

    def _lock_on(self, res: Dict) -> None:
        p5 = res["p5"]
        self._profile = res["profile"].copy()
        self._anchor = res["profile"].copy()
        self._bank = [res["profile"].copy()]
        self._run_open = False
        self._switch_streak = 0
        self._thr = float(np.clip((p5 + STRANGER_BAND_MAX) / 2.0,
                                  max(HARD_REJECT_COSINE + 0.05, THRESHOLD_MIN), THRESHOLD_MAX))
        self._thr_clean = min(self._thr + OVERLAP_MARGIN, 0.95)
        self._tse_ref = res["ref"]
        self._tse_cache = None
        self._cust_f0 = collections.deque(maxlen=20)
        f0 = _pitch_hz(res["ref"], self.sample_rate)
        if f0 is not None:
            self._cust_f0.append(f0)
        self._utts = []
        self._utt_aids = []
        self._loud_foreign = []
        self.state = LockState.LOCKED
        self._turns_since_lock = 0
        self._verified_since_lock = 0
        self._proven = False
        self._probation_turns: List[np.ndarray] = []
        self._wait_until_s = 0.0
        logger.warning(
            f"[{self.call_id}] 🔒 CUSTOMER VOICE LOCKED | p5={p5:.3f} thr={self._thr:.2f} "
            f"clean={self._thr_clean:.2f} windows={res['windows']} "
            f"separated={res['separated']} tse={'on' if self._extractor_ok else 'off'} "
            f"— from here only this voice reaches STT"
        )

# [FIXES17-2026-10-05 applied]

# [FIXES18-2026-10-05 applied]

# [FIXES25-2026-10-05 applied]

# [FIXES28-2026-10-05 applied]
# [FIXES31-2026-10-06 applied]

# [FIXES33-2026-10-06 applied]