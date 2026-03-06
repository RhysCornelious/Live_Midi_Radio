# import os
# import time
# import psutil
# import numpy as np
# import pretty_midi
# from basic_pitch.inference import predict
# from basic_pitch import ICASSP_2022_MODEL_PATH

# # Resolve all paths relative to this script's directory, not the working directory
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# def p(relative_path):
#     return os.path.join(SCRIPT_DIR, relative_path)

# # all 4 input/output pairs
# FILES = [
#     (p("audio_files/Input 1.wav"), p("audio_files/Input 1_out_spotify.mid")),
#     (p("audio_files/Input 2.wav"), p("audio_files/Input 2_out_spotify.mid")),
#     (p("audio_files/Input 3.wav"), p("audio_files/Input 3_out_spotify.mid")),
#     (p("audio_files/Input 4.wav"), p("audio_files/Input 4_out_spotify.mid")),
# ]

# process = psutil.Process(os.getpid())

# # Pre-load the model once before the loop so that the first file's
# # timer isn't inflated by TensorFlow's one-time model loading cost.
# print("Loading model...")
# from basic_pitch.inference import Model
# basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
# print("Model loaded.\n")

# for WAV_FILE, OUTPUT_MIDI in FILES:
#     start_time = time.time()
#     peak_memory = 0

#     # Run Basic Pitch transcription using the pre-loaded model.
#     # predict() returns: (model_output, midi_data, note_events)
#     # midi_data is a pretty_midi.PrettyMIDI object ready to use
#     model_output, midi_data, note_events = predict(
#         WAV_FILE,
#         basic_pitch_model,
#         # You can tune these thresholds if needed:
#         # onset_threshold=0.5,
#         # frame_threshold=0.3,
#         # minimum_note_length=58,   # ms
#         # minimum_frequency=32.7,   # Hz (C1)
#         # maximum_frequency=2093.0, # Hz (C7)
#         # multiple_pitch_bends=False,
#         # melodia_trick=True,
#     )

#     mem = process.memory_info().rss / (1024 * 1024)
#     if mem > peak_memory:
#         peak_memory = mem

#     # Basic Pitch returns a PrettyMIDI object. We'll grab the first instrument
#     # (it combines all voices into one) and apply the same post-processing as before.
#     if not midi_data.instruments:
#         print(f"{WAV_FILE}: No notes detected, skipping.")
#         continue

#     instr = midi_data.instruments[0]
#     instr.program = 0  # Grand piano

#     # -------------------------------------------------------------------------
#     # Post-processing (carried over from original script)
#     # -------------------------------------------------------------------------

#     # Remove notes outside the 30–90 MIDI range (almost certainly noise)
#     instr.notes = [n for n in instr.notes if 30 <= n.pitch <= 90]

#     # Detect the key of the song by weighting each pitch class by duration * velocity,
#     # then snap any off-key notes to the nearest in-key pitch.
#     if instr.notes:
#         pc_weights = np.zeros(12)
#         for n in instr.notes:
#             pc_weights[n.pitch % 12] += (n.end - n.start) * n.velocity

#         best_pcs, best_score = set(), -1.0
#         for root in range(12):
#             for _, intervals in [("major", {0, 2, 4, 5, 7, 9, 11}), ("minor", {0, 2, 3, 5, 7, 8, 10})]:
#                 pcs = frozenset((root + iv) % 12 for iv in intervals)
#                 score = sum(pc_weights[pc] for pc in pcs)
#                 if score > best_score:
#                     best_pcs, best_score = pcs, score

#         snap_map = {}
#         for pc in range(12):
#             if pc in best_pcs:
#                 snap_map[pc] = 0
#             else:
#                 for d in range(1, 7):
#                     if (pc + d) % 12 in best_pcs:
#                         snap_map[pc] = d
#                         break
#                     if (pc - d) % 12 in best_pcs:
#                         snap_map[pc] = -d
#                         break

#         for n in instr.notes:
#             shift = snap_map.get(n.pitch % 12, 0)
#             if shift != 0:
#                 n.pitch += shift

#     # Merge notes of the same pitch separated by less than 150ms
#     if instr.notes:
#         instr.notes.sort(key=lambda n: (n.pitch, n.start))
#         merged = []
#         prev = instr.notes[0]
#         for note in instr.notes[1:]:
#             if note.pitch == prev.pitch and note.start - prev.end < 0.150:
#                 prev = pretty_midi.Note(
#                     max(prev.velocity, note.velocity),
#                     prev.pitch,
#                     prev.start,
#                     max(prev.end, note.end)
#                 )
#             else:
#                 merged.append(prev)
#                 prev = note
#         merged.append(prev)
#         instr.notes = merged

#     # Remove notes shorter than 150ms (noise artifacts)
#     instr.notes = [n for n in instr.notes if (n.end - n.start) >= 0.150]

#     # Cap polyphony at 4 simultaneous notes, keeping highest priority (duration * velocity)
#     if instr.notes:
#         notes = instr.notes
#         starts = np.array([n.start for n in notes])
#         ends = np.array([n.end for n in notes])
#         durations = ends - starts
#         vels = np.array([n.velocity for n in notes])
#         priority = durations * vels
#         end_t = float(np.max(ends))
#         drop = set()
#         t = 0.0
#         while t <= end_t:
#             active_mask = (starts <= t) & (ends > t)
#             active_i = np.where(active_mask)[0]
#             if len(active_i) > 4:
#                 order = np.argsort(-priority[active_i])
#                 for idx in active_i[order[4:]]:
#                     drop.add(int(idx))
#             t += 0.050
#         instr.notes = [n for i, n in enumerate(notes) if i not in drop]

#     # Write MIDI output
#     pm = pretty_midi.PrettyMIDI()
#     pm.instruments.append(instr)
#     pm.write(OUTPUT_MIDI)

#     mem = process.memory_info().rss / (1024 * 1024)
#     if mem > peak_memory:
#         peak_memory = mem

#     elapsed = time.time() - start_time
#     note_count = len(instr.notes)
#     print(f"{WAV_FILE}")
#     print(f"  Time: {elapsed:.2f}s  |  Peak RAM: {peak_memory:.1f} MB  |  Notes Output: {note_count}")
#     print()

import os
import gc
import time
import psutil
import tempfile
import logging
import pathlib
import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

logging.getLogger().setLevel(logging.ERROR)

from basic_pitch.inference import predict, Model
from basic_pitch.constants import AUDIO_SAMPLE_RATE
import basic_pitch

# Use the ONNX model directly instead of TensorFlow to reduce RAM usage.
# Requires: pip install basic-pitch[onnx]
_bp_dir = pathlib.Path(basic_pitch.__file__).parent
ONNX_MODEL_PATH = _bp_dir / "saved_models" / "icassp_2022" / "nmp.onnx"

# 1 second worth of samples at the target rate per chunk
READ_CHUNK_SIZE = int(AUDIO_SAMPLE_RATE * 1.0)

# Basic Pitch needs a minimum amount of audio to work with — chunks shorter
# than this (e.g. the last chunk of a file) are skipped to avoid errors.
MIN_CHUNK_SAMPLES = int(AUDIO_SAMPLE_RATE * 0.1)

# Resolve all paths relative to this script's directory, not the working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def p(relative_path):
    return os.path.join(SCRIPT_DIR, relative_path)

# All 4 input/output pairs
FILES = [
    (p("audio_files/Input 1.wav"), p("audio_files/Input 1_out_spotify.mid")),
    (p("audio_files/Input 2.wav"), p("audio_files/Input 2_out_spotify.mid")),
    (p("audio_files/Input 3.wav"), p("audio_files/Input 3_out_spotify.mid")),
    (p("audio_files/Input 4.wav"), p("audio_files/Input 4_out_spotify.mid")),
]

process = psutil.Process(os.getpid())

# Pre-load the model once before the loop so that the first file's
# timer isn't inflated by model loading cost.
print("Loading model...")
basic_pitch_model = Model(ONNX_MODEL_PATH)
print("Model loaded.\n")

results = []

for WAV_FILE, OUTPUT_MIDI in FILES:
    start_time = time.time()
    peak_memory = 0

    all_notes = []
    chunk_count = 0
    skipped_chunks = 0
    time_offset = 0.0  # tracks where in the song each chunk starts, in seconds

    with sf.SoundFile(WAV_FILE) as f:
        native_sr = f.samplerate

        # Native samples per chunk (before resampling)
        native_chunk_size = int(READ_CHUNK_SIZE * native_sr / AUDIO_SAMPLE_RATE)

        # Integer ratio for resampling to Basic Pitch's required 22050 Hz
        g = gcd(AUDIO_SAMPLE_RATE, native_sr)
        up = AUDIO_SAMPLE_RATE // g
        down = native_sr // g

        while True:
            raw = f.read(native_chunk_size, dtype="float32")
            if len(raw) == 0:
                break

            # Mix down to mono if stereo
            if raw.ndim > 1:
                raw = np.mean(raw, axis=1)

            # Resample chunk to 22050 Hz if needed
            if up != down:
                chunk_audio = resample_poly(raw, up, down).astype(np.float32)
            else:
                chunk_audio = raw

            chunk_duration = len(raw) / native_sr

            # Skip chunks that are too short for Basic Pitch to process
            if len(chunk_audio) < MIN_CHUNK_SAMPLES:
                skipped_chunks += 1
                time_offset += chunk_duration
                continue

            # predict() only accepts a file path, so write the chunk to a temp wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, chunk_audio, AUDIO_SAMPLE_RATE)

            try:
                _, midi_data, _ = predict(tmp_path, basic_pitch_model)
            finally:
                os.remove(tmp_path)

            # Shift all predicted notes forward by the chunk's position in the song
            if midi_data and midi_data.instruments:
                for note in midi_data.instruments[0].notes:
                    note.start += time_offset
                    note.end += time_offset
                    all_notes.append(note)

            # Explicitly free chunk memory before next iteration
            del chunk_audio, raw, midi_data
            gc.collect()

            time_offset += chunk_duration
            chunk_count += 1

            mem = process.memory_info().rss / (1024 * 1024)
            if mem > peak_memory:
                peak_memory = mem

    # -------------------------------------------------------------------------
    # Post-processing (carried over from original script)
    # -------------------------------------------------------------------------

    # Remove notes outside the 30-90 MIDI range (almost certainly noise)
    all_notes = [n for n in all_notes if 30 <= n.pitch <= 90]

    # Detect the key of the song by weighting each pitch class by duration * velocity,
    # then snap any off-key notes to the nearest in-key pitch.
    if all_notes:
        pc_weights = np.zeros(12)
        for n in all_notes:
            pc_weights[n.pitch % 12] += (n.end - n.start) * n.velocity

        best_pcs, best_score = set(), -1.0
        for root in range(12):
            for _, intervals in [("major", {0, 2, 4, 5, 7, 9, 11}), ("minor", {0, 2, 3, 5, 7, 8, 10})]:
                pcs = frozenset((root + iv) % 12 for iv in intervals)
                score = sum(pc_weights[pc] for pc in pcs)
                if score > best_score:
                    best_pcs, best_score = pcs, score

        snap_map = {}
        for pc in range(12):
            if pc in best_pcs:
                snap_map[pc] = 0
            else:
                for d in range(1, 7):
                    if (pc + d) % 12 in best_pcs:
                        snap_map[pc] = d
                        break
                    if (pc - d) % 12 in best_pcs:
                        snap_map[pc] = -d
                        break

        for n in all_notes:
            shift = snap_map.get(n.pitch % 12, 0)
            if shift != 0:
                n.pitch += shift

    # Merge notes of the same pitch separated by less than 150ms
    if all_notes:
        all_notes.sort(key=lambda n: (n.pitch, n.start))
        merged = []
        prev = all_notes[0]
        for note in all_notes[1:]:
            if note.pitch == prev.pitch and note.start - prev.end < 0.150:
                prev = pretty_midi.Note(
                    max(prev.velocity, note.velocity),
                    prev.pitch,
                    prev.start,
                    max(prev.end, note.end)
                )
            else:
                merged.append(prev)
                prev = note
        merged.append(prev)
        all_notes = merged

    # Remove notes shorter than 150ms (noise artifacts)
    all_notes = [n for n in all_notes if (n.end - n.start) >= 0.150]

    # Cap polyphony at 4 simultaneous notes, keeping highest priority (duration * velocity)
    if all_notes:
        starts = np.array([n.start for n in all_notes])
        ends = np.array([n.end for n in all_notes])
        durations = ends - starts
        vels = np.array([n.velocity for n in all_notes])
        priority = durations * vels
        end_t = float(np.max(ends))
        drop = set()
        t = 0.0
        while t <= end_t:
            active_mask = (starts <= t) & (ends > t)
            active_i = np.where(active_mask)[0]
            if len(active_i) > 4:
                order = np.argsort(-priority[active_i])
                for idx in active_i[order[4:]]:
                    drop.add(int(idx))
            t += 0.050
        all_notes = [n for i, n in enumerate(all_notes) if i not in drop]

    # Write MIDI output
    pm = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(program=0)  # Grand piano
    instr.notes = all_notes
    pm.instruments.append(instr)
    pm.write(OUTPUT_MIDI)

    mem = process.memory_info().rss / (1024 * 1024)
    if mem > peak_memory:
        peak_memory = mem

    elapsed = time.time() - start_time
    results.append((os.path.basename(WAV_FILE), elapsed, peak_memory, len(all_notes)))

for name, elapsed, peak_mem, notes in results:
    print(f"Predicting MIDI for {name}")
    print(f"  Time: {elapsed:.2f}s  |  Peak RAM: {peak_mem:.1f} MB  |  Notes Output: {notes}")
    print()