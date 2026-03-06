import os
import sys
import time

import numpy as np
import pretty_midi
import psutil
import scipy.signal as spsig
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi, sosfilt, sosfilt_zi

# Input / output (reference MIDIs from basic-pitch: in1_good_midi.mid etc.)
WAV_FILE = "audio_files/Input 1.wav"
OUTPUT_MIDI = "audio_files/Input 1_out.mid"
if len(sys.argv) >= 2 and str(sys.argv[1]).strip():
    WAV_FILE = str(sys.argv[1]).strip()
if len(sys.argv) >= 3 and str(sys.argv[2]).strip():
    OUTPUT_MIDI = str(sys.argv[2]).strip()
_midi_base = OUTPUT_MIDI.replace(".mid", "")
OUTPUT_WAV_FILTERED = _midi_base + "_filtered.wav"
OUTPUT_WAV_RESYNTH = _midi_base + "_resynth.wav"

# Assignment constraints
READ_SIZE = 2048
MAX_CHUNKS = 2048

# ── Analysis settings ──────────────────────────────────────────────────────────
# FRAME_SIZE raised 4096 → 8192: gives 5.4 Hz/bin instead of 10.8 Hz/bin.
# Reference MIDIs include F#1 (MIDI 30, 46 Hz) and E1 (MIDI 28, 41 Hz).
# At 4096, those two notes differ by less than 1 FFT bin — unresolvable.
# At 8192 they are ~2 bins apart and the salience interpolation can distinguish them.
FRAME_SIZE = 8192
# STFT hop: use a dedicated analysis hop for melody accuracy. Basic-pitch-style
# transcription uses fine time resolution (~5–20 ms). The I/O read_size (2048)
# was being reused as STFT hop, giving ~46 ms/frame and smearing the tune.
# 1024 samples @ 44.1 kHz ≈ 23 ms/frame (~43 frames/s) preserves note timing.
HOP_SIZE = 1024

MIDI_MIN = 28        # was 30; reference contains MIDI 28 (E1 = 41 Hz)
MIDI_MAX = 96
CAND_STEP = 0.5

# DET_LOW_HZ raised 55 → 33: the old value physically removed F#1 and E1 from
# the audio signal before any analysis, making them impossible to detect.
DET_LOW_HZ = 33.0
DET_HIGH_HZ = 5200.0
DBG_LOW_HZ = 80.0
DBG_HIGH_HZ = 2600.0

HUM_BASE_HZ = 60.0
HUM_HARMONICS = (1, 2, 3)
HUM_Q = 25.0

# ── Noise reduction (for radio / noisy inputs) ─────────────────────────────────
# Disabled by default: was too aggressive and killed melody + made filtered WAV harsh.
# To re-enable gently: set NOISE_GATE_RATIO high (e.g. 3.0) and NOISE_SUBTRACT_ALPHA low (e.g. 0.15).
NOISE_GATE_RATIO = None      # None = no gate. If float, gate blocks below this * noise_floor
NOISE_SUBTRACT_ALPHA = 0.0   # 0 = off. Spectral hiss subtraction (try 0.1–0.2 if needed)

# ── Polyphony tracking parameters ─────────────────────────────────────────────
# Basic-pitch (Spotify) uses min_note_len ~11 frames (~128 ms), energy_tol=11 frames
# (sustain through ~128 ms dips). We approximate: longer sustain = fewer fragments.
POLY_MAX_PER_FRAME    = 3    # candidates per frame
POLY_MIN_NOTE_FRAMES  = 4    # min frames before a voice becomes a note
POLY_MAX_GAP_FRAMES   = 5    # allow 5 frame misses before closing (~115 ms at 23 ms/frame);
                              # basic-pitch uses ~11 frames (~128 ms) of tolerance
POLY_MATCH_SEMITONES  = 1.5  # matching tolerance; bridges vibrato/jitter
POLY_START_ONSET      = 0.10 # onset needed to open a new voice
POLY_MAX_ACTIVE_VOICES = 4   # hard cap during tracking

# ── Post-processing cap ────────────────────────────────────────────────────────
MAX_OUTPUT_POLYPHONY = 3

# ── MIDI output (match reference: organ, smoother transitions) ──────────────────
# GM program: 0=piano, 16=Drawbar Organ, 19=Church Organ
MIDI_PROGRAM = 16
# Extend each note end by this many seconds so notes overlap slightly (legato).
LEGATO_EXTEND_S = 0.028
# Pitch-bend glide from previous note into current (semitones); only if interval <= this.
PITCH_BEND_MAX_SEMITONES = 2
PITCH_BEND_GLIDE_S = 0.045


def midi_to_hz(m):
    return 440.0 * (2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0))


def velocity_from_conf(conf):
    # Reference MIDIs span velocity 28-99 (mean ~55).
    # Old mapping was capped at 84 with a narrow range.
    return int(np.clip(20 + 78 * (max(0.0, conf) ** 0.65), 28, 100))


def build_notch_chain(sr):
    nyq = 0.5 * sr
    chain = []
    for h in HUM_HARMONICS:
        f0 = HUM_BASE_HZ * h
        if f0 >= nyq * 0.95:
            continue
        b, a = iirnotch(f0 / nyq, HUM_Q)
        chain.append({"b": b.astype(np.float64), "a": a.astype(np.float64), "zi": lfilter_zi(b, a) * 0.0})
    return chain


def apply_notch_chain(x, chain):
    y = x
    for stage in chain:
        y, zi_next = lfilter(stage["b"], stage["a"], y, zi=stage["zi"])
        stage["zi"] = zi_next
    return y


def align_length(x, n):
    y = np.zeros((n,), dtype=np.float32)
    m = min(n, len(x))
    if m > 0:
        y[:m] = np.asarray(x[:m], dtype=np.float32)
    return y


def stream_filter_audio(path, process):
    with sf.SoundFile(path) as finfo:
        sr = finfo.samplerate
        total_samples = len(finfo)
    read_size = max(READ_SIZE, int(np.ceil(float(total_samples) / float(MAX_CHUNKS))))

    nyq = 0.5 * sr
    sos_det = butter(4, [DET_LOW_HZ / nyq, min(DET_HIGH_HZ / nyq, 0.999)], btype="band", output="sos")
    sos_dbg = butter(4, [DBG_LOW_HZ / nyq, min(DBG_HIGH_HZ / nyq, 0.999)], btype="band", output="sos")
    zi_det = sosfilt_zi(sos_det) * 0.0
    zi_dbg = sosfilt_zi(sos_dbg) * 0.0
    notch_det = build_notch_chain(sr)
    notch_dbg = build_notch_chain(sr)

    det_chunks = []
    dbg_chunks = []
    chunk_count = 0
    peak_mem = process.memory_info().rss / (1024.0 * 1024.0)

    with sf.SoundFile(path) as f:
        while chunk_count < MAX_CHUNKS:
            chunk = f.read(read_size, dtype="float32")
            if len(chunk) == 0:
                break
            if chunk.ndim > 1:
                chunk = np.mean(chunk, axis=1)

            y_det, zi_det = sosfilt(sos_det, chunk, zi=zi_det)
            y_dbg, zi_dbg = sosfilt(sos_dbg, chunk, zi=zi_dbg)
            y_det = apply_notch_chain(y_det, notch_det)
            y_dbg = apply_notch_chain(y_dbg, notch_dbg)

            y_det = np.tanh(1.5 * y_det).astype(np.float32)
            y_dbg = np.asarray(y_dbg, dtype=np.float32)

            det_chunks.append(y_det)
            dbg_chunks.append(y_dbg)
            chunk_count += 1

            mem = process.memory_info().rss / (1024.0 * 1024.0)
            if mem > peak_mem:
                peak_mem = mem

    x_det = np.concatenate(det_chunks) if len(det_chunks) else np.zeros((0,), dtype=np.float32)
    x_dbg = np.concatenate(dbg_chunks) if len(dbg_chunks) else np.zeros((0,), dtype=np.float32)
    x_det = align_length(x_det, total_samples)
    x_dbg = align_length(x_dbg, total_samples)

    if NOISE_GATE_RATIO is not None and NOISE_GATE_RATIO > 0:
        block = max(READ_SIZE, 2048)
        n_blocks = max(1, len(x_det) // block)
        rms_per_block = np.array([
            np.sqrt(np.mean(x_det[i * block:(i + 1) * block] ** 2) + 1e-12)
            for i in range(n_blocks)
        ], dtype=np.float32)
        noise_floor = float(np.percentile(rms_per_block, 10))
        gate_thresh = NOISE_GATE_RATIO * noise_floor
        for i in range(n_blocks):
            start, end = i * block, min((i + 1) * block, len(x_det))
            rms = np.sqrt(np.mean(x_det[start:end] ** 2) + 1e-12)
            if rms < gate_thresh and rms > 1e-9:
                x_det[start:end] *= (rms / gate_thresh)
            elif rms <= 1e-9:
                x_det[start:end] = 0.0

    return sr, total_samples, read_size, x_det, x_dbg, chunk_count, peak_mem


def harmonic_stft(x, sr, hop_size):
    hop_size = int(np.clip(hop_size, 1, FRAME_SIZE - 1))
    if len(x) < FRAME_SIZE:
        f = np.fft.rfftfreq(FRAME_SIZE, 1.0 / sr)
        return f, np.zeros((0,), dtype=np.float64), np.zeros((len(f), 0), dtype=np.complex64), np.zeros_like(x)

    f, t, z = spsig.stft(
        x,
        fs=sr,
        window="hann",
        nperseg=FRAME_SIZE,
        noverlap=FRAME_SIZE - hop_size,
        boundary="zeros",
        padded=True,
    )
    mag = np.abs(z)

    if NOISE_SUBTRACT_ALPHA > 0:
        noise_spec = np.percentile(mag, 15, axis=1, keepdims=True)
        mag = np.maximum(mag - NOISE_SUBTRACT_ALPHA * noise_spec, 0.0)
        z = z * (mag / (np.abs(z) + 1e-12))

    # HPSS-style mask: harmonic content is smooth in time; percussive is smooth in freq.
    # With FRAME_SIZE doubled to 8192, Hz/bin halved to 5.4 Hz. The frequency-axis
    # kernel must be scaled from 17→31 bins to maintain the same ~167 Hz coverage
    # needed to reliably separate tonal peaks from broadband percussive content.
    med_time = median_filter(mag, size=(1, 17), mode="nearest")
    med_freq = median_filter(mag, size=(31, 1), mode="nearest")
    mask_h = med_time / (med_time + med_freq + 1e-9)
    z_h = z * mask_h

    _, y_h = spsig.istft(
        z_h,
        fs=sr,
        window="hann",
        nperseg=FRAME_SIZE,
        noverlap=FRAME_SIZE - hop_size,
        input_onesided=True,
        boundary=True,
    )
    y_h = align_length(y_h, len(x))
    return f, t, z_h, y_h


def compute_salience(z_h, freqs, sr):
    if z_h.size == 0:
        midi_grid = np.arange(MIDI_MIN, MIDI_MAX + 1e-9, CAND_STEP, dtype=np.float64)
        return midi_grid, np.zeros((0, len(midi_grid)), dtype=np.float32), np.zeros((0, len(midi_grid)), dtype=np.float32)

    mag = np.abs(z_h).T.astype(np.float64)  # T x F
    t_count, f_count = mag.shape

    # Per-frame whitening reduces influence of broad-band background.
    env = median_filter(mag, size=(1, 9), mode="nearest")
    mag_w = mag / (env + 1e-7)

    midi_grid = np.arange(MIDI_MIN, MIDI_MAX + 1e-9, CAND_STEP, dtype=np.float64)
    f0 = midi_to_hz(midi_grid)
    bin_hz = sr / FRAME_SIZE

    # Extended to 8 harmonics: extra partials strengthen salience for low notes.
    harm = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
    w_h  = np.asarray([1.0, 0.82, 0.62, 0.45, 0.30, 0.20, 0.13, 0.08], dtype=np.float64)
    pos = f0[:, None] * harm[None, :] / bin_hz
    k = np.floor(pos).astype(np.int64)
    a = pos - k
    valid = (k >= 1) & (k < (f_count - 1))

    sal = np.zeros((t_count, len(midi_grid)), dtype=np.float64)

    # Pre-compute octave suppression index pairs once
    oct1_steps = int(round(12.0 / CAND_STEP))
    oct2_steps = int(round(24.0 / CAND_STEP))
    n_bins = len(midi_grid)

    for ti in range(t_count):
        m = mag_w[ti]
        s = np.zeros((n_bins,), dtype=np.float64)
        for hi in range(len(harm)):
            vh = valid[:, hi]
            if not np.any(vh):
                continue
            kk = k[vh, hi]
            aa = a[vh, hi]
            val = (1.0 - aa) * m[kk] + aa * m[kk + 1]
            s[vh] += w_h[hi] * val

        # ── Octave salience suppression ────────────────────────────────────────
        for shift in (oct1_steps, oct2_steps):
            ia = np.arange(shift, n_bins)
            ib = ia - shift
            harmonic_mask = s[ia] < 0.60 * s[ib]
            s[ia[harmonic_mask]] *= 0.25

        sal[ti] = s

    p99 = float(np.percentile(sal, 99))
    if p99 > 1e-12:
        sal = sal / p99
    sal = np.clip(sal, 0.0, 1.0)

    d_sal = np.vstack([np.zeros((1, sal.shape[1])), np.maximum(sal[1:] - sal[:-1], 0.0)])
    flux_spec = np.vstack([np.zeros((1, mag_w.shape[1])), np.maximum(mag_w[1:] - mag_w[:-1], 0.0)])
    flux = np.mean(flux_spec, axis=1, keepdims=True)
    p95_flux = float(np.percentile(flux, 95))
    if p95_flux > 1e-12:
        flux = flux / p95_flux

    onset = np.clip(0.78 * d_sal + 0.22 * flux, 0.0, 1.0)
    return midi_grid, sal.astype(np.float32), onset.astype(np.float32)


def project_salience_to_midi(midi_grid, sal, onset):
    if sal.size == 0:
        midi_vals = np.arange(MIDI_MIN, MIDI_MAX + 1, dtype=np.int16)
        empty = np.zeros((0, len(midi_vals)), dtype=np.float32)
        return midi_vals, empty, empty

    midi_vals = np.arange(int(np.floor(midi_grid[0])), int(np.ceil(midi_grid[-1])) + 1, dtype=np.int16)
    t_count = sal.shape[0]
    pitch_sal = np.zeros((t_count, len(midi_vals)), dtype=np.float32)
    pitch_onset = np.zeros_like(pitch_sal)

    for mi, m in enumerate(midi_vals):
        idx = np.where(np.abs(midi_grid - float(m)) <= 0.55)[0]
        if idx.size == 0:
            idx = np.asarray([int(np.argmin(np.abs(midi_grid - float(m))))], dtype=np.int64)
        pitch_sal[:, mi] = np.max(sal[:, idx], axis=1)
        pitch_onset[:, mi] = np.max(onset[:, idx], axis=1)

    if pitch_sal.shape[1] >= 3:
        smooth = pitch_sal.copy()
        smooth[:, 1:-1] = 0.25 * pitch_sal[:, :-2] + 0.50 * pitch_sal[:, 1:-1] + 0.25 * pitch_sal[:, 2:]
        pitch_sal = smooth

    # Slightly stronger temporal smoothing (5 frames) so pitch sticks to the melody.
    pitch_sal = median_filter(pitch_sal, size=(5, 1), mode="nearest")
    pitch_onset = median_filter(pitch_onset, size=(5, 1), mode="nearest")
    return midi_vals, pitch_sal, pitch_onset


def select_frame_candidates(midi_vals, pitch_sal, pitch_onset):
    if pitch_sal.size == 0:
        return [], np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    t_count, n_pitch = pitch_sal.shape
    frame_candidates = []

    frame_ref   = np.percentile(pitch_sal, 86, axis=1).astype(np.float32)
    start_thr   = np.maximum(0.10,  0.62 * frame_ref)
    sustain_thr = np.maximum(0.07,  0.44 * frame_ref)

    for ti in range(t_count):
        s = pitch_sal[ti]
        o = pitch_onset[ti]

        peaks = np.ones((n_pitch,), dtype=bool)
        if n_pitch >= 2:
            peaks[0] = s[0] >= s[1]
            peaks[-1] = s[-1] >= s[-2]
        if n_pitch > 2:
            peaks[1:-1] = (s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:])

        valid = peaks & ((s >= start_thr[ti]) | (o >= POLY_START_ONSET))
        idx = np.where(valid)[0]

        if idx.size == 0:
            top_k = min(POLY_MAX_PER_FRAME, n_pitch)
            if top_k >= n_pitch:
                idx = np.arange(n_pitch, dtype=np.int64)
            else:
                idx = np.argpartition(s, -top_k)[-top_k:]

        scores = s[idx] + 0.38 * o[idx]
        order = np.argsort(-scores)
        selected = []

        for oi in order:
            pi = int(idx[oi])
            p = int(midi_vals[pi])
            amp = float(s[pi])
            ons = float(o[pi])

            harmonic = False
            for sj in selected:
                p0 = int(midi_vals[sj])
                d = abs(p - p0)
                if d in (12, 24):
                    lower_amp = float(s[sj])
                    if amp < 0.78 * lower_amp:
                        harmonic = True
                        break
            if harmonic:
                continue

            selected.append(pi)
            if len(selected) >= POLY_MAX_PER_FRAME:
                break

        frame_candidates.append(selected)

    return frame_candidates, start_thr, sustain_thr


def track_polyphonic_notes(midi_vals, pitch_sal, pitch_onset, frame_candidates, start_thr, sustain_thr, frame_times, hop_sec):
    if pitch_sal.size == 0:
        return []

    t_count = pitch_sal.shape[0]
    active = []
    events = []

    def close_voice(v, end_frame):
        if (end_frame - v["start"]) < POLY_MIN_NOTE_FRAMES:
            return
        weights = np.asarray(v["weights"], dtype=np.float64)
        pitches = np.asarray(v["pitches"], dtype=np.float64)
        midi_est = float(np.average(pitches, weights=weights)) if np.sum(weights) > 1e-9 else float(np.mean(pitches))
        pitch = int(np.clip(int(np.round(midi_est)), MIDI_MIN, MIDI_MAX))
        amp = float(np.mean(v["amps"]))
        t_start = float(frame_times[v["start"]]) if len(frame_times) else (v["start"] * hop_sec)
        t_end = float(frame_times[end_frame - 1] + hop_sec) if len(frame_times) else (end_frame * hop_sec)
        if t_end > t_start:
            events.append((t_start, t_end, pitch, amp))

    for ti in range(t_count):
        s = pitch_sal[ti]
        o = pitch_onset[ti]
        cand_idx = frame_candidates[ti]
        cands = [
            {"idx": ci, "pitch": float(midi_vals[ci]), "amp": float(s[ci]), "onset": float(o[ci])}
            for ci in cand_idx
        ]

        pairs = []
        for vi, v in enumerate(active):
            for ci, c in enumerate(cands):
                d = abs(c["pitch"] - v["last_pitch"])
                if d <= POLY_MATCH_SEMITONES:
                    pairs.append((d, -c["amp"], vi, ci))
        pairs.sort()

        matched_v = set()
        matched_c = set()
        for _, _, vi, ci in pairs:
            if vi in matched_v or ci in matched_c:
                continue
            matched_v.add(vi)
            matched_c.add(ci)
            c = cands[ci]
            v = active[vi]
            v["last_pitch"] = c["pitch"]
            v["last_frame"] = ti
            v["misses"] = 0
            v["pitches"].append(c["pitch"])
            v["amps"].append(c["amp"])
            v["weights"].append(c["amp"] + 0.35 * c["onset"] + 1e-6)

        closed = []
        for vi, v in enumerate(active):
            if vi in matched_v:
                continue
            v["misses"] += 1
            if v["misses"] > POLY_MAX_GAP_FRAMES:
                close_voice(v, v["last_frame"] + 1)
                closed.append(vi)

        if closed:
            closed_set = set(closed)
            active = [v for i, v in enumerate(active) if i not in closed_set]

        for ci, c in enumerate(cands):
            if ci in matched_c:
                continue
            if c["onset"] < POLY_START_ONSET and c["amp"] < float(start_thr[ti]):
                continue

            duplicate = False
            for v in active:
                if abs(v["last_pitch"] - c["pitch"]) <= 0.40 and (ti - v["last_frame"]) <= POLY_MAX_GAP_FRAMES:
                    duplicate = True
                    break
                if abs(v["last_pitch"] - c["pitch"]) <= 1.0 and (ti - v["last_frame"]) <= 2:
                    duplicate = True
                    break
            if duplicate:
                continue

            if len(active) >= POLY_MAX_ACTIVE_VOICES:
                weakest = int(np.argmin(
                    [np.mean(v["amps"][-min(len(v["amps"]), 3):]) if len(v["amps"]) else 0.0 for v in active]
                ))
                close_voice(active[weakest], active[weakest]["last_frame"] + 1)
                del active[weakest]

            active.append({
                "start": ti,
                "last_frame": ti,
                "last_pitch": c["pitch"],
                "misses": 0,
                "pitches": [c["pitch"]],
                "amps": [c["amp"]],
                "weights": [c["amp"] + 0.35 * c["onset"] + 1e-6],
            })

    for v in active:
        close_voice(v, v["last_frame"] + 1)

    if len(events) == 0:
        return events

    amps = np.asarray([e[3] for e in events], dtype=np.float64)
    amp_floor = max(0.08, float(np.percentile(amps, 22)) * 0.92)
    events = [e for e in events if e[3] >= amp_floor]
    if len(events) <= 1:
        return events

    events.sort(key=lambda x: (x[0], x[2]))
    merged = [events[0]]
    for s0, e0, p0, a0 in events[1:]:
        ps, pe, pp, pa = merged[-1]
        # Merge same-pitch segments separated by up to 130 ms (align with basic-pitch
        # sustain tolerance; reduces sporadic breaks in sustained notes).
        if p0 == pp and (s0 - pe) <= 0.130:
            merged[-1] = (ps, max(pe, e0), pp, float(0.5 * (pa + a0)))
        else:
            merged.append((s0, e0, p0, a0))
    return merged


def melody_extraction_filter(events, max_poly=MAX_OUTPUT_POLYPHONY):
    if not events:
        return events

    events = sorted(events, key=lambda x: x[0])
    n = len(events)

    # ── Pass 1: Octave-stack collapse ─────────────────────────────────────────
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        si, ei, pi, ai = events[i]
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            sj, ej, pj, aj = events[j]
            if sj >= ei:
                break

            overlap = min(ei, ej) - max(si, sj)
            if overlap < 0.060:
                continue

            diff = abs(pi - pj)
            is_octave = any(abs(diff - oct) <= 1 for oct in (12, 24, 36))
            if not is_octave:
                continue

            if ai >= aj:
                keep[j] = False
            else:
                keep[i] = False
                break

    events = [e for i, e in enumerate(events) if keep[i]]
    if not events:
        return events

    # ── Pass 1b: Trim output pitch range ─────────────────────────────────────
    # Our detection range is wide (28–96) to help salience computation, but the
    # OUTPUT should only contain melodically relevant pitches. Notes below 30
    # are sub-bass rumble; notes above 84 are almost always harmonics.
    OUTPUT_PITCH_MIN = 30
    OUTPUT_PITCH_MAX = 84
    events = [(s, e, p, a) for s, e, p, a in events
              if OUTPUT_PITCH_MIN <= p <= OUTPUT_PITCH_MAX]
    if not events:
        return events

    # ── Pass 2: Hard polyphony cap (top-voice priority) ───────────────────────
    # Melody is almost always the highest-pitched voice at any moment. Old logic
    # kept the LOUDEST notes — which tends to be bass/chords, dropping melody.
    # New logic: always keep the highest note, then fill remaining slots by amp.
    end_t = max(e for s, e, p, a in events)
    drop = set()
    t = 0.0
    while t <= end_t:
        active_i = [i for i, (s, e, p, a) in enumerate(events) if s <= t < e]
        if len(active_i) > max_poly:
            top_voice = max(active_i, key=lambda i: events[i][2])
            rest = [i for i in active_i if i != top_voice]
            rest_by_amp = sorted(rest, key=lambda i: -events[i][3])
            keep_rest = rest_by_amp[:max_poly - 1]
            for idx in rest:
                if idx not in keep_rest:
                    drop.add(idx)
        t += 0.020

    events = [e for i, e in enumerate(events) if i not in drop]

    # ── Pass 3: Minimum duration gate ─────────────────────────────────────────
    events = [(s, e, p, a) for s, e, p, a in events if (e - s) >= 0.128]

    # ── Pass 4: Top-voice velocity boost ──────────────────────────────────────
    # Make the melody (highest note at any moment) louder so it stands out.
    if events:
        events = sorted(events, key=lambda x: x[0])
        boosted = list(events)
        end_t = max(e for s, e, p, a in events)
        top_set = set()
        t = 0.0
        while t <= end_t:
            active_i = [i for i, (s, e, p, a) in enumerate(boosted) if s <= t < e]
            if active_i:
                top_i = max(active_i, key=lambda i: boosted[i][2])
                top_set.add(top_i)
            t += 0.020
        for i in top_set:
            s, e, p, a = boosted[i]
            boosted[i] = (s, e, p, min(1.0, a * 1.35))
        events = boosted

    return events


def synth_from_notes(note_events, sr, n_samples):
    y = np.zeros((n_samples,), dtype=np.float64)
    if len(note_events) == 0:
        return y.astype(np.float32)

    for s, e, p, amp in note_events:
        n0 = int(np.clip(np.floor(s * sr), 0, n_samples))
        n1 = int(np.clip(np.ceil(e * sr), 0, n_samples))
        if n1 <= n0 + 1:
            continue
        n = n1 - n0
        t = np.arange(n, dtype=np.float64) / sr
        f0 = float(midi_to_hz(float(p)))

        a = float(np.clip(0.16 + 0.36 * amp, 0.10, 0.55))
        sig = a * (0.86 * np.sin(2.0 * np.pi * f0 * t)
                 + 0.21 * np.sin(2.0 * np.pi * 2.0 * f0 * t)
                 + 0.09 * np.sin(2.0 * np.pi * 3.0 * f0 * t))

        attack = max(1, int(0.012 * sr))
        release = max(1, int(0.040 * sr))
        env = np.ones((n,), dtype=np.float64)
        env[:attack] *= np.linspace(0.0, 1.0, attack, endpoint=True)
        env[-release:] *= np.linspace(1.0, 0.0, release, endpoint=True)
        y[n0:n1] += sig * env

    return np.clip(y, -1.0, 1.0).astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────
process = psutil.Process(os.getpid())
start_time = time.time()

if not os.path.exists(WAV_FILE):
    raise FileNotFoundError(f"Input file not found: {WAV_FILE}")

sr, total_samples, read_size, x_det, x_dbg, chunk_count, peak_memory = stream_filter_audio(WAV_FILE, process)

freqs, frame_times, z_harm, y_harm = harmonic_stft(x_det, sr, HOP_SIZE)
midi_grid, sal, onset = compute_salience(z_harm, freqs, sr)

hop_sec = HOP_SIZE / sr
midi_vals, pitch_sal, pitch_onset = project_salience_to_midi(midi_grid, sal, onset)
frame_candidates, start_thr, sustain_thr = select_frame_candidates(midi_vals, pitch_sal, pitch_onset)
note_events = track_polyphonic_notes(
    midi_vals, pitch_sal, pitch_onset,
    frame_candidates, start_thr, sustain_thr,
    frame_times, hop_sec,
)

# Final melody extraction: collapse harmonic octave stacks, enforce polyphony cap.
note_events = melody_extraction_filter(note_events, max_poly=MAX_OUTPUT_POLYPHONY)

# Build MIDI with organ sound and smoother transitions (legato + optional pitch-bend glide).
pm = pretty_midi.PrettyMIDI()
instr = pretty_midi.Instrument(program=MIDI_PROGRAM)
# Sort by start time for legato and glide logic.
note_events = sorted(note_events, key=lambda x: (x[0], x[2]))
prev_pitch = None
for i, (s, e, p, amp) in enumerate(note_events):
    s = float(max(0.0, s))
    e = float(max(s + 1e-3, e))
    # Legato: extend end slightly so notes blend into the next (no hard gaps).
    if LEGATO_EXTEND_S > 0:
        next_start = note_events[i + 1][0] if (i + 1) < len(note_events) else None
        if next_start is not None and (next_start - s) < 2.0:
            e = min(e + LEGATO_EXTEND_S, next_start - 0.005)
        else:
            e = e + LEGATO_EXTEND_S
    e = max(e, s + 0.001)  # pretty_midi requires end > start
    instr.notes.append(pretty_midi.Note(
        velocity=velocity_from_conf(amp),
        pitch=int(p),
        start=s,
        end=e,
    ))
    # Smooth glide from previous note into this one (pitch bend at note start).
    if prev_pitch is not None and PITCH_BEND_GLIDE_S > 0 and PITCH_BEND_MAX_SEMITONES > 0:
        delta = prev_pitch - p
        if abs(delta) <= PITCH_BEND_MAX_SEMITONES:
            # MIDI pitch bend: ±8192 ≈ ±2 semitones → 1 semitone ≈ 4096
            bend_start = int(np.clip(round(delta * 4096), -8192, 8191))
            instr.pitch_bends.append(pretty_midi.PitchBend(bend_start, s))
            instr.pitch_bends.append(pretty_midi.PitchBend(0, s + PITCH_BEND_GLIDE_S))
    prev_pitch = p
pm.instruments.append(instr)
pm.write(OUTPUT_MIDI)

filt_out = np.clip(0.25 * x_dbg + 0.95 * y_harm, -1.0, 1.0).astype(np.float32)
sf.write(OUTPUT_WAV_FILTERED, filt_out, sr, subtype="PCM_16")

resynth = synth_from_notes(note_events, sr, total_samples)
sf.write(OUTPUT_WAV_RESYNTH, resynth, sr, subtype="PCM_16")

mem_now = process.memory_info().rss / (1024.0 * 1024.0)
peak_memory = max(peak_memory, mem_now)

print("\n--- Processing Results ---")
print(f"Input File:       {WAV_FILE}")
print(f"Output MIDI:      {OUTPUT_MIDI}")
print(f"Debug WAV (filtered signal): {OUTPUT_WAV_FILTERED}")
print(f"Debug WAV (resynth sines):   {OUTPUT_WAV_RESYNTH}")
print(f"Chunk Size:       {read_size} samples")
print(f"STFT Hop:         {HOP_SIZE} samples ({hop_sec*1000:.1f} ms)")
print(f"Chunks Read:      {chunk_count}")
print(f"Processing Time:  {time.time() - start_time:.2f}s")
print(f"Peak RAM Usage:   {peak_memory:.2f} MB")
print(f"Total Samples:    {total_samples}")
print(f"STFT Frames:      {sal.shape[0]}")
print(f"MIDI Notes:       {len(instr.notes)}")

# Optional: compare to reference MIDI (e.g. basic-pitch in1_good_midi.mid)
try:
    base = os.path.splitext(os.path.basename(WAV_FILE))[0]
    num = base.replace("Input ", "").strip()
    ref_midi = os.path.join(os.path.dirname(WAV_FILE), f"in{num}_good_midi.mid")
    if os.path.isfile(ref_midi):
        ref_pm = pretty_midi.PrettyMIDI(ref_midi)
        ref_notes = [n for inst in ref_pm.instruments for n in inst.notes]
        ref_pitches = [n.pitch for n in ref_notes] if ref_notes else []
        our_pitches = [n.pitch for n in instr.notes]
        print(f"\n--- vs reference {os.path.basename(ref_midi)} ---")
        print(f"Reference notes: {len(ref_notes)}  |  Ours: {len(instr.notes)}")
        if ref_pitches:
            print(f"Reference pitch range: {min(ref_pitches)}–{max(ref_pitches)}  |  Ours: {min(our_pitches) if our_pitches else '-'}–{max(our_pitches) if our_pitches else '-'}")
except Exception as e:
    pass
