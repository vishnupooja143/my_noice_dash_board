"""
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
        from target_extractor import TargetExtractor
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

        # streaming
        self._pending = np.zeros(0, dtype=np.int16)
        ring_len = max(TSE_WINDOW_HOPS, LONG_BACK_HOPS, WINDOW_BACK_HOPS) + 2 * LOOKAHEAD_HOPS + 6
        self._ring: Deque[_Hop] = collections.deque(maxlen=ring_len)
        self._next_idx = 0
        self._next_release = 0
        self._prev_open = True
        self._rms_hist: Deque[float] = collections.deque(maxlen=max(10, int(BG_HISTORY_S / self._hop_s)))
        self._bg = 5.0

        # bot state (written from the event loop, read here — plain attributes)
        self._bot_speaking = False
        self._bot_end_ts = 0.0
        self._bot_end_idx = 0

        # enrollment
        self.state = LockState.ENROLLING
        self._utts: List[List[np.ndarray]] = []
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
        self._turn_audio: Deque[np.ndarray] = collections.deque(maxlen=80)   # 8 s
        self._cust_rms: Deque[float] = collections.deque(maxlen=20)
        self._foreign_turns: List[np.ndarray] = []
        self._last_err_ts = 0.0
        self._enroll_hold = False            # True = don't learn any voice yet (AMD phase)
        self._cust_pass_idx = -10 ** 9       # last hop where the customer passed OVER the bot
        logger.info(f"[{self.call_id}] 🔒 CustomerVoiceLock init sr={sample_rate} hop={HOP_MS}ms "
                    f"lookahead={LOOKAHEAD_HOPS * HOP_MS}ms sep={self._separator is not None} "
                    f"tse={self._extractor is not None}")

    # ── public API ────────────────────────────────────────────────────────
    @property
    def locked(self) -> bool:
        return self.state is LockState.LOCKED

    @property
    def bot_speaking(self) -> bool:
        return self._bot_speaking

    def hold_enrollment(self) -> None:
        """Nothing heard from now on is learned as the customer (AMD listens
        before the greeting)."""
        with self._mutex:
            self._enroll_hold = True

    def open_enrollment(self) -> None:
        """Greeting finished - start learning the voice that answers the bot."""
        with self._mutex:
            self._enroll_hold = False
            self._utts = []
            self._enroll_speech_s = 0.0
            self._last_enroll_idx = -10 ** 9
        logger.info(f"[{self.call_id}] 🎙️ enrollment open — learning the voice that answers the bot")

    def customer_active(self, window_s: float = 1.0) -> bool:
        """True if the lock passed the customer's voice over the bot within window_s."""
        return (self._next_idx - self._cust_pass_idx) * self._hop_s <= window_s

    def vad_should_mute(self) -> bool:
        """True = feed the VAD silence: the bot is talking and the customer is not.
        No mutex here - it is called per frame from the event loop."""
        return (self.state is LockState.LOCKED and self._bot_speaking
                and not self.customer_active(1.0))

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
        if speaking:
            self._bot_seen = True
        if self._bot_speaking and not speaking:
            self._bot_end_ts = time.monotonic()
            self._bot_end_idx = self._next_idx     # audio clock, not wall clock
        self._bot_speaking = speaking

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
        with self._mutex:
            if self._pending_answer is not None:
                done, self._pending_answer = self._pending_answer, None
                self._learn_from_answer(done)
            if pcm is not None and len(pcm):
                self._pending = np.concatenate([self._pending, np.asarray(pcm, dtype=np.int16)])
            while len(self._pending) >= self.hop_len:
                raw = self._pending[:self.hop_len].copy()
                self._pending = self._pending[self.hop_len:]
                self._push(raw)
                while self._next_idx - 1 - self._next_release >= LOOKAHEAD_HOPS:
                    hop = self._get(self._next_release)
                    self._next_release += 1
                    if hop is not None:
                        out.append(self._release(hop))
        return out

    def summary(self) -> str:
        s = self.stats
        if self._sims:
            q = np.percentile(np.array(self._sims), [10, 50, 90])
            dist = f" sims p10/p50/p90={q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}"
        else:
            dist = ""
        return (f"[{self.call_id}] VoiceLock state={self.state.value} thr={self._thr:.2f}/"
                f"{self._thr_clean:.2f} pass={s['pass']} tse={s['tse_ok']} "
                f"tse_fail={s['tse_fail']} soft={s['soft_pass']} hold={s['hold_open']} other={s['other']} "
                f"unsure={s['uncertain']} noise/silence={s['silence']} last_sim={self._last_sim:.2f}"
                f"{dist} bank={len(self._bank)} widened={s['widened']} switch={s['switch']} "
                f"foreign_blocked={s['foreign_blocked']} answer_pass={s['answer_pass']} "
                f"mode={MODE} turns_ok={s['turn_customer']} turns_dropped={s['turn_other']} "
                f"relocks={self._relocks} bot_events={'yes' if self._bot_seen else 'NO'}")

    def judge_turn(self) -> Tuple[str, Optional[float]]:
        """Call when a user turn is committed. Returns ("customer" | "other" |
        "unknown", similarity). "other" = drop this transcript."""
        with self._mutex:
            audio = list(self._turn_audio)
            self._turn_audio.clear()
            if self.state is not LockState.LOCKED or MODE != "verify" or not audio:
                return "unknown", None
            dur = len(audio) * self._hop_s
            pcm = np.concatenate(audio[-60:])
            loud = _rms(pcm)
            typical = float(np.median(self._cust_rms)) if len(self._cust_rms) >= 2 else None
            quiet = typical is not None and loud < QUIET_RATIO * typical
            sim, emb = self._similarity(pcm)
            if sim is not None:
                self._last_sim = sim
                self._sims.append(sim)
            loud_txt = (f"level {loud:.0f} vs customer {typical:.0f}" if typical
                        else f"level {loud:.0f}")

            if dur < TURN_MIN_S:
                # quiet AND not clearly the customer's voice → someone away from the
                # phone. (A strong voice match that is quiet is the customer
                # speaking softly — kept.)
                weak = sim is None or sim < 0.40
                if (quiet and weak) or (sim is not None and sim < SHORT_TURN_REJECT_SIM):
                    self.stats["turn_other"] += 1
                    logger.warning(f"[{self.call_id}] 🚫 short turn from another voice dropped "
                                   f"({dur:.1f}s, sim={sim if sim is not None else float('nan'):.2f}, "
                                   f"{loud_txt}{', too quiet' if quiet else ''})")
                    return "other", sim
                logger.info(f"[{self.call_id}] 🗣️ short turn accepted ({dur:.1f}s, "
                            f"sim={sim if sim is not None else float('nan'):.2f}, {loud_txt})")
                return "customer", sim

            if sim is None:
                return "unknown", None

            is_other = sim < TURN_REJECT_SIM or (quiet and sim < AMBIGUOUS_SIM)
            if not is_other:
                self._foreign_turns = []
                self.stats["turn_customer"] += 1
                self._cust_rms.append(loud)
                if sim >= self._thr and emb is not None:
                    self._bank_add(emb)
                logger.info(f"[{self.call_id}] 🗣️ turn verified as customer "
                            f"(sim={sim:.2f}, {dur:.1f}s, {loud_txt}, bank={len(self._bank)})")
                return "customer", sim

            self.stats["turn_other"] += 1
            if not quiet:
                self._foreign_turns.append(pcm)
            if len(self._foreign_turns) >= FOREIGN_ANSWERS_TO_RELOCK:
                # the only (loud, near-phone) voice talking to the bot for several
                # turns → the phone changed hands; follow the person on the line
                utts = self._foreign_turns
                self._foreign_turns = []
                self._relocks += 1
                logger.error(f"[{self.call_id}] 🔁 {len(utts)} turns in a row from a different "
                             f"voice (last sim={sim:.2f}) — re-identifying the customer")
                self.state = LockState.ENROLLING
                self._profile = None
                self._bank = []
                self._cust_rms.clear()
                self._attempts = 0
                self._best = None
                self._utts = [[u] for u in utts]
                self._enroll_speech_s = sum(len(u) for u in utts) / self.sample_rate
                self._maybe_build()
                return "customer", sim
            logger.warning(f"[{self.call_id}] 🚫 turn from a DIFFERENT voice dropped "
                           f"(sim={sim:.2f}, {dur:.1f}s, {loud_txt}{', too quiet' if quiet else ''})")
            return "other", sim

    def close(self) -> None:
        logger.info(self.summary())

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
        self._ring.append(hop)
        self._next_idx += 1
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
                self._turn_audio.append(hop.pcm)
            self.stats["raw_turn"] += 1
            return self._shape(hop.pcm, hop.pcm)
        try:
            seg = self._locked_decision(hop)
        except Exception as e:
            now = time.monotonic()
            if now - self._last_err_ts > 5.0:
                self._last_err_ts = now
                logger.error(f"[{self.call_id}] voice lock decision error (hop silenced): {e}")
            seg = None
        if seg is not None and hop.bot:
            self._cust_pass_idx = hop.idx
        if hop.answer:
            self._answer_hist.append((seg is not None, hop.pcm))
            self._check_relock()
        return self._shape(hop.pcm, seg)

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
        self._attempts = 0
        self._best = None
        self._utts = [rejected]
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
        if emb is not None and sim is not None and sim >= self._thr:
            self._bank_add(emb)
            logger.info(f"[{self.call_id}] ➕ answer added to voice profile "
                        f"(sim={sim:.2f}, bank={len(self._bank)})")

    def _bank_add(self, emb: np.ndarray) -> None:
        self._bank.append(emb.astype(np.float32))
        if len(self._bank) > BANK_MAX:
            self._bank.pop(1)            # keep the original enrollment at [0]
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
        if self._bank:
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
        self._last_cand_idx = hop.idx

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
        if self._enroll_hold or not hop.answer:
            return
        if self._utts and hop.idx - self._last_enroll_idx <= ENROLL_UTT_GAP_HOPS:
            self._utts[-1].append(hop.pcm)
        else:
            self._utts.append([hop.pcm])
        self._last_enroll_idx = hop.idx
        self._enroll_speech_s += self._hop_s
        while self._enroll_speech_s > ENROLL_BUFFER_MAX_S and len(self._utts) > 1:
            dropped = self._utts.pop(0)
            self._enroll_speech_s -= len(dropped) * self._hop_s
        self._maybe_build()

    def _maybe_build(self) -> None:
        if self._building or self.state is not LockState.ENROLLING:
            return
        need = ENROLL_MIN_SECONDS + self._attempts * ENROLL_RETRY_EXTRA_S
        if self._enroll_speech_s < need:
            return
        utts = [np.concatenate(u) for u in self._utts if u]
        self._building = True
        threading.Thread(target=self._build_worker, args=(utts,),
                         name=f"voicelock-enroll-{self.call_id}", daemon=True).start()

    def _build_worker(self, utts: List[np.ndarray]) -> None:
        res = None
        self._need_more = False
        try:
            if self._extractor is not None:
                self._extractor_ok = bool(self._extractor.available())
            res = self._build_profile(utts)
        except Exception as e:
            logger.error(f"[{self.call_id}] enrollment build failed: {e}")
        with self._mutex:
            self._building = False
            if res is None and self._need_more:
                return          # not a failed attempt — just not enough speech yet
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
                logger.error(f"[{self.call_id}] ❌ Could not isolate one customer voice after "
                             f"{self._attempts} attempts (best p5="
                             f"{self._best['p5'] if self._best else 'n/a'}) — audio passes unfiltered")
            else:
                logger.warning(f"[{self.call_id}] ⚠️ Enrollment attempt {self._attempts} not clean "
                               f"(p5={res['p5'] if res else 'n/a'}) — collecting more speech")

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
            if len(embs) < 3:
                return None
            prof = _norm(np.median(np.vstack(embs), axis=0))
            # 2nd pass: per utterance, the source that matches that voice best
            pieces = [max(a, key=lambda s: self._mean_sim(s, prof)) for a in alts]
            embs, wins = self._embed_pieces(pieces)
            if len(embs) < 3:
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
        self._utts = []
        self.state = LockState.LOCKED
        logger.warning(
            f"[{self.call_id}] 🔒 CUSTOMER VOICE LOCKED | p5={p5:.3f} thr={self._thr:.2f} "
            f"clean={self._thr_clean:.2f} windows={res['windows']} "
            f"separated={res['separated']} tse={'on' if self._extractor_ok else 'off'} "
            f"— from here only this voice reaches STT"
        )