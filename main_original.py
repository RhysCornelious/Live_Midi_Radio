import os
import soundfile as sf
import numpy as np
import pretty_midi
import time
import psutil
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, lfilter, lfilter_zi, find_peaks

# Function to convert a freq in Hz to a MIDI note number
def freq_to_midi(f):
    return 12 * np.log2(np.maximum(f, 1e-9) / 440.0) + 69

# Peak finding using scipy's find_peaks function
def find_peaks_scipy(fft_mag, freqs, relevant_indices, max_val):
    freq_at_bins = freqs[relevant_indices]
    height_arr = np.where(freq_at_bins < 100, max_val * 0.16, max_val * 0.19)
    peak_idx, _ = find_peaks(
        fft_mag,
        height=height_arr,
        prominence=max_val * 0.06,
        distance=4,
    )
    return list(peak_idx)

# all 4 input/output pairs
FILES = [
    ("audio_files/Input 1.wav", "audio_files/Input 1_out.mid"),
    ("audio_files/Input 2.wav", "audio_files/Input 2_out.mid"),
    ("audio_files/Input 3.wav", "audio_files/Input 3_out.mid"),
    ("audio_files/Input 4.wav", "audio_files/Input 4_out.mid"),
]

process = psutil.Process(os.getpid())

for WAV_FILE, OUTPUT_MIDI in FILES:
    start_time = time.time()
    peak_memory = 0

    # Setting up ideal chunk size and window for Fourier
    with sf.SoundFile(WAV_FILE) as f_info:
        # Finding length of file
        sr = f_info.samplerate # Should be 44100 but for robustness
        total_samples = len(f_info)
        
        # figure out the hop size required for each load to get 2048 chunks - processing
        # was not found to be by the limit so want to maximize resolution as much as possible
        HOP_SIZE = int(np.ceil(total_samples / 2048))

        # make sure that we are loading at least 8192 data points per chunk
        # larger chunks give better frequency resolution (~5.4 Hz/bin at 44.1k)
        # Still not 'dumping the file' but since ram is low and speed is high, loading more gives better resolution
        CHUNK_SIZE = max(HOP_SIZE, 8192)

        # Generate window at size of chunk for optimal spectal leakage. 
        # Different windows attempted, hanning was best through testing
        window = np.hanning(CHUNK_SIZE)

        # zero padding helps to increase precision of peak frequency estimates
        FFT_SIZE = CHUNK_SIZE * 2

        # Set up freq array to index fourier transform
        freqs = np.fft.rfftfreq(FFT_SIZE, 1 / sr)

    # Highest note on a keyboard (C8) is just under 4200Hz
    # This means we won't be using up memory for notes outside of piano's range
    relevant_indices = np.where(freqs <= 4200)[0]

    # Setting up the bandpass and notch filters. IIR filters are useful for minimizing
    # memory usage by carrying internal state from chunk to chunk

    # 4th order butterworth bandpass to keep 80-2500 Hz. wider upper limit lets
    # the HPS use more harmonics (a 500Hz note needs 1000, 1500, 2000 Hz).
    # the wiener mask handles suppressing HF noise so the bandpass doesnt need to
    nyquist_freq = 0.5 * sr
    bandpass_coeffs = butter(4,
        [80.0 / nyquist_freq, min(2500.0 / nyquist_freq, 0.999)],
        btype="band", output="sos")
    bandpass_state = sosfilt_zi(bandpass_coeffs) * 0.0

    # notch filters at 60, 120, and 180 Hz to remove power line hum in radio recordings
    # These have Q factor of 25 which is super high, they are meant to just cut out the hum
    notch_filters = []
    for hum_freq in [60.0, 120.0, 180.0]:
        notch_b, notch_a = iirnotch(hum_freq / nyquist_freq, 25.0)
        notch_filters.append([notch_b.astype(np.float64),
                                notch_a.astype(np.float64),
                                lfilter_zi(notch_b, notch_a) * 0.0])

    # Initialize midi file and then set instrument (0 is for grand piano)
    pm = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(program=0)

    # Used to track active notes, so we don't get them played every frame of the MIDI file
    active_notes = {}

    # Small buffer to prevent notes from being played rapidly when they disapear from one chunk
    # and then reappear in the next
    note_persistence = {}

    # Stores the magnitudes of notes, allowing you to play a note again if it was being
    # sustained and is played again louder
    prev_magnitudes = {}

    # This tracks when individual notes started, and is important for determining if we should
    # retrigger or update note velocity when it rises
    note_start_frame = {}
    frame_count = 0

    # These intervals relate to 2-8x the frequency (octaves)
    harmonic_intervals = [12, 19, 24, 28, 31, 34, 36]

    # this buffer holds one analysis window worth of filtered samples
    # after the first fill, we shift it left by HOP_SIZE and read new samples into the end
    analysis_buffer = np.zeros(CHUNK_SIZE, dtype=np.float32)

    # Used to track size of our sliding window as we move across the audio file
    buffer_valid = 0
    current_sample = 0

    # noise floor gets built up adaptively from the signal as we go
    noise_floor = None

    # circular buffer that stores the last 16 frames of cleaned magnitudes
    # we take the median across 16 most recent loaded chunks for use in Wiener mask
    hpss_buffer_size = 16
    num_relevant = len(relevant_indices)
    magnitude_history = np.zeros((hpss_buffer_size, num_relevant), dtype=np.float32)
    history_write_pos = 0

    audio_stream = sf.SoundFile(WAV_FILE)

    # here begins our actual processing of the audio file. Each time we increment our buffer
    # filter the audio, and process it into MIDI notes
    while True:
        # if buffer is full from the last iteration, shift it left to make room
        # we keep the overlap portion and only read new samples to fill the gap
        if buffer_valid >= CHUNK_SIZE:
            overlap = CHUNK_SIZE - HOP_SIZE
            analysis_buffer[:overlap] = analysis_buffer[HOP_SIZE:]
            buffer_valid = overlap

        # read raw audio. Some safety checks to make sure the file is not empty and that it is mono
        samples_needed = CHUNK_SIZE - buffer_valid
        raw = audio_stream.read(samples_needed, dtype="float32")
        if len(raw) == 0:
            break
        if raw.ndim > 1:
            raw = np.mean(raw, axis=1)

        # apply our bandpass then each notch filter in sequence
        filtered = raw.astype(np.float64)
        filtered, bandpass_state = sosfilt(bandpass_coeffs, filtered, zi=bandpass_state)
        for notch in notch_filters:
            filtered, notch[2] = lfilter(notch[0], notch[1], filtered, zi=notch[2])

        # write the filtered samples into the buffer and clean up
        n_new = len(filtered)
        analysis_buffer[buffer_valid:buffer_valid + n_new] = filtered[:n_new].astype(np.float32)
        buffer_valid += n_new
        del filtered, raw

        # Safety check to make sure there is enough data to be read
        # This will cut it off slightly early rather than attempting to process
        # a smaller chunk that cannot express lower tones effectively
        # While this may cut off processing early, it is better than some wild note at the end of the file
        if buffer_valid < CHUNK_SIZE:
            break

        # tracking our frame, used for noise floor initialization and logic for retriggering notes
        frame_count += 1

        # Taking the Fourier transform using our Hanning window designed prior and getting magnitude across spectrum
        full_fft = np.fft.rfft(analysis_buffer * window, n=FFT_SIZE)
        full_fft_mag = np.abs(full_fft)

        # narrow down to our relevant indices (Defined earlier to have max at C8 of piano aka 4200Hz-ish)
        fft_mag = full_fft_mag[relevant_indices]

        # adaptive noise floor gives a guess for the baseline background noise
        # We have a noise floor for each freq in the FFT, and update it with a moving
        # Average every time we come through here. Creates boolean array indicating freqs that are just noise
        if noise_floor is None:
            noise_floor = fft_mag.copy()
        else:
            quiet_bins = fft_mag < noise_floor
            noise_floor[quiet_bins] = (0.20 * fft_mag[quiet_bins]
                                       + 0.80 * noise_floor[quiet_bins])
            noise_floor[~quiet_bins] = (0.007 * fft_mag[~quiet_bins]
                                        + 0.993 * noise_floor[~quiet_bins])

        # here we deal with the noise from the radio
        # we skip the first 5 frames while the noise floor is still settling in
        if frame_count > 5:
            # subtract 2x the noise floor from each bin to remove the constant hiss/static
            # clamp to a small floor (0.01x noise) so we dont get zero-magnitude artifacts
            noise_subtracted = fft_mag - 2.0 * noise_floor
            spectral_floor = 0.01 * noise_floor
            np.maximum(noise_subtracted, spectral_floor, out=noise_subtracted)

            # recall the 16 frame rolling buffer where we take the median
            buf_idx = history_write_pos % hpss_buffer_size
            magnitude_history[buf_idx] = noise_subtracted
            history_write_pos += 1
            frames_available = min(history_write_pos, hpss_buffer_size)
            harmonic_magnitude = np.median(magnitude_history[:frames_available], axis=0)

            # wiener mask - creates a smooth 0-to-1 mask to keep real notes and mute the noise
            # Small adjustement for higher freqs (above 1000Hz) was helpful in stopping random high notes
            bin_frequencies = freqs[relevant_indices]
            hf_boost = 1.0 + np.clip(
                (bin_frequencies - 1000.0) / 1000.0, 0.0, None
            ) * 4.0
            effective_noise = noise_floor * hf_boost

            harmonic_power = np.power(harmonic_magnitude, 2.0)
            noise_power = np.power(effective_noise, 2.0)
            wiener_mask = harmonic_power / (harmonic_power + noise_power + 1e-10)
            np.minimum(wiener_mask, 1.0, out=wiener_mask)

            # apply the mask to our magnitudes
            fft_mag = (fft_mag * wiener_mask).astype(fft_mag.dtype)
            del noise_subtracted, harmonic_magnitude, wiener_mask

        # if the highest value in the spectrum is this low, there are no notes being played and it will incldue random noise
        # Just prevents trying to get a note from just noise
        max_val = np.max(fft_mag)
        if max_val < 0.006:
            current_sample += HOP_SIZE
            continue

        # HPS = harmonic product spectrum, In radio, real notes will have strong upper harmonics. By multiplying the magnitude
        # of f with them we get a good idea of which notes are actually notes
        hps = fft_mag.copy()
        for harmonic in [2, 3, 4]:
            stretched_indices = relevant_indices * harmonic
            valid = stretched_indices < len(full_fft_mag)
            hps_contrib = np.zeros_like(fft_mag)
            hps_contrib[valid] = full_fft_mag[stretched_indices[valid]]
            hps *= hps_contrib

        # normalize hps to a reasonable range so the threshold logic works
        hps_max = np.max(hps)
        if hps_max > 0:
            hps = hps / hps_max
        
        # we use the harmonic products for finding peaks to get rid of noisy peaks
        peaks = find_peaks_scipy(hps, freqs, relevant_indices, 1.0)

        # Defining time stamp for the midi file and making a set() to store the notes
        # as well as dict for their magnitudes
        t_start = current_sample / sr
        current_chunk_midi_notes = set()
        current_magnitudes = {}

        # Making a set that stores the freq and magnitude of the peaks and sorting by magnitude
        # This provides polyphonic detection with magnitude awareness
        peak_magnitudes = [(fft_mag[p], p) for p in peaks]
        peak_magnitudes.sort(key=lambda x: x[0], reverse=True)

        # Cycles through all the notes
        for mag, p in peak_magnitudes:
            # Using parabolic interpolation to prevent notes jumping back and forth
            # Provides a rudimentary way to much more precisely identify the peak frequency
            # that isn't limited by bin size. Only works if there is a valid bin to either side
            if 0 < p < len(fft_mag) - 1:
                # Taking the magnitudes of the peak freq and the bins to either side
                y1, y2, y3 = fft_mag[p-1], fft_mag[p], fft_mag[p+1]

                # Calculates the difference in magnitude from the peak and its neighbors
                denom = (2 * y2 - y1 - y3)
                
                if denom != 0:
                    # This determines wether the higher neighbor (y3) or lower neighbor (y1)
                    # has a greater magnitude
                    offset = (y3 - y1) / (2 * denom)
                    # This shifts our freq in the direction of the higher magnitude neighbor
                    # by a factor that is influenced by the magnitude of the different between the neighbors
                    # as well as the magnitude difference between our peak and its neighbors
                    actual_freq = freqs[relevant_indices[p]] + offset * (freqs[1] - freqs[0])
                # If it is 0 then all three are equal and we just stick with the middle bin
                else:
                    actual_freq = freqs[relevant_indices[p]]
            # If no valid bin on one side then just stick with our note
            else:
                actual_freq = freqs[relevant_indices[p]]

            # And finally asign to a midi note
            m_note = int(np.round(freq_to_midi(actual_freq)))

            # Checks if our note was already playing. If so, it only needs to be 0.12*max amplitude to be sustained
            # Otherwise, the threshold is a bit higher (0.19*max) if a note is new. lenient so we catch more melody
            is_active = m_note in active_notes
            threshold = max_val * 0.12 if is_active else max_val * 0.19
            
            # If we don't hit this threshold for the peaks we continue
            if mag < threshold:
                continue
                
            # Logic to check if a sustained note has been replayed while active
            # If it jumps by more than 50% in magnitude then we re hit the key
            # This only occurs if it has been present for more than 3 frames, otherwise we just 
            # will update its velocity
            birth_frame = note_start_frame.get(m_note, frame_count)
            age_in_frames = frame_count - birth_frame

            retrigger = False
            if is_active and m_note in prev_magnitudes:
                if mag > prev_magnitudes[m_note] * 3.0 and age_in_frames > 3:
                        retrigger = True
            
            # Stores this magnitude for future comparisons in following chunks
            prev_magnitudes[m_note] = mag

            # Filtering out the harmonics.
            # Goes through all notes we currently have, if it is harmonic we do not add it
            is_harmonic = False
            for existing_note in current_chunk_midi_notes:
                diff = abs(m_note - existing_note)
                if diff in harmonic_intervals:
                    existing_mag_val = current_magnitudes.get(existing_note, 0)
                    # The magnitude must be at least half of the current note (which is higher magnitude since
                    # that is how we sorted them) to also be added. Allows us to capture an octave played versus
                    # simply the harmonic distortion
                    if mag < (existing_mag_val * 0.5):
                        is_harmonic = True
                        break

            # Checks to make sure that the note is on a keyboard and also not just harmonic distortion
            if 21 <= m_note <= 108 and not is_harmonic:
                # If the retrigger was correct, we pop the 'old' note from active and add it to
                # our note archive so we can add the replayed note to our current midi ntoes
                if retrigger:
                    old_start, old_vel = active_notes.pop(m_note)
                    instr.notes.append(pretty_midi.Note(old_vel, m_note, old_start, t_start))
                    # also need to resent the note start frame again
                    note_start_frame[m_note] = frame_count
                
                # Add the valid note to our current notes and set its magnitude
                current_chunk_midi_notes.add(m_note)
                current_magnitudes[m_note] = mag

                # higher notes tend to be quieter, so scaling appropriately
                velocity_multiplier = 3500 + (max(0, m_note - 50) * 150) 
                
                # This helps with ensuring magnitudes in files that would have different chunk sizes
                # Clip allows us to set a min and max value on the current velocity that matches midi's expected values
                normalized_mag = mag/CHUNK_SIZE 
                current_vel = int(np.clip(normalized_mag * velocity_multiplier, 80, 110))

                # If the note was not active before (or just got cancelled due to retriggering) we add it to active notes
                # with its velocity and set its debounce frames and set its start frame
                if m_note not in active_notes:
                    active_notes[m_note] = [t_start, current_vel]
                    note_persistence[m_note] = 2
                    note_start_frame[m_note] = frame_count
                # These are notes that were not newly pressed or retriggered but still met the threshold to be sustained
                # so we keep their debounce frames high and increase the vel of the active note if higher. This is mostly 
                # for the case that a note just caught the prev frame quietly but was actually louder, yet we don't want to retrigger
                else:
                    note_persistence[m_note] = 2
                    if current_vel > active_notes[m_note][1]:
                        active_notes[m_note][1] = current_vel

        # going through all the active notes to update them properly based on new data
        all_tracked_notes = list(active_notes.keys())
        for n in all_tracked_notes:
            if n not in current_chunk_midi_notes:
                # If it wasn't currently played, we lower its persistence
                # Once persistence hits 0 (3 frames without playing) then we pop it from active notes,
                # clean up our prev_magnitudes, note_start_frame, and note_persistence dicts 
                # and append the note onto our midi file
                note_persistence[n] -= 1
                if note_persistence[n] <= 0:
                    start_t, vel = active_notes.pop(n)
                    if n in prev_magnitudes: del prev_magnitudes[n]
                    if n in note_start_frame: del note_start_frame[n]
                    del note_persistence[n]
                    instr.notes.append(pretty_midi.Note(vel, n, start_t, t_start))

        # cleaning up our data every time to minimize the memory requirements
        del current_chunk_midi_notes
        del current_magnitudes
        del fft_mag
        del hps

        # updating the current sample by the hop size
        current_sample += HOP_SIZE
        
        # updating our memory monitory
        mem = process.memory_info().rss / (1024 * 1024)
        if mem > peak_memory:
            peak_memory = mem

    audio_stream.close()

    # one the wav file ends, we update our midi instrument with all remaining notes
    for pitch, (start_t, vel) in active_notes.items():
        instr.notes.append(pretty_midi.Note(vel, pitch, start_t, total_samples / sr))

    # post processing - used to further clean up the midi file use reasoning behind how melodies are constructed
    # This mostly consists of sanity checks on note range, timing and polyphony. There is also a bit of 'rounding'
    # to assist with making the output audio sound less cluttered or jumpy

    # remove notes outside of the 30-90 midi range since those are almost certainly noise
    instr.notes = [n for n in instr.notes if 30 <= n.pitch <= 90]

    # detect the key of the song by weighting each pitch class by how long and loud
    # its notes are, then trying every major and minor key to see which fits best.
    # once we know the key we snap any off-key notes to the nearest in-key pitch
    if instr.notes:
        # weight each pitch class by duration * velocity so longer louder notes count more
        pc_weights = np.zeros(12)
        for n in instr.notes:
            pc_weights[n.pitch % 12] += (n.end - n.start) * n.velocity

        # try every root note with both major and minor scales, pick the highest score
        best_pcs, best_score = set(), -1.0
        for root in range(12):
            for _, intervals in [("major", {0, 2, 4, 5, 7, 9, 11}), ("minor", {0, 2, 3, 5, 7, 8, 10})]:
                pcs = frozenset((root + iv) % 12 for iv in intervals)
                score = sum(pc_weights[pc] for pc in pcs)
                if score > best_score:
                    best_pcs, best_score = pcs, score

        # for each pitch class not in the key, figure out the closest in-key pitch
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

        # apply the snap to every note
        for n in instr.notes:
            shift = snap_map.get(n.pitch % 12, 0)
            if shift != 0:
                n.pitch += shift

    # merge notes of the same pitch that are separated by less than 150ms
    # the spectral cleaning can cause melody notes to flicker on and off across
    # frames, so this fuses those fragments back into one sustained note
    if instr.notes:
        instr.notes.sort(key=lambda n: (n.pitch, n.start))
        merged = []
        prev = instr.notes[0]
        for note in instr.notes[1:]:
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
        instr.notes = merged

    # remove any notes shorter than 150ms since those are almost certainly noise artifacts
    instr.notes = [n for n in instr.notes if (n.end - n.start) >= 0.150]

    # cap polyphony at 4 simultaneous notes. when there are more than 4 playing at once
    # we keep the ones with the highest duration*velocity score since those are most
    # likely to be the actual melody rather than noise
    if instr.notes:
        notes = instr.notes
        starts = np.array([n.start for n in notes])
        ends = np.array([n.end for n in notes])
        durations = ends - starts
        vels = np.array([n.velocity for n in notes])
        priority = durations * vels
        end_t = float(np.max(ends))
        drop = set()
        t = 0.0
        # step through the song in 50ms increments and check for polyphony violations
        while t <= end_t:
            active_mask = (starts <= t) & (ends > t)
            active_i = np.where(active_mask)[0]
            if len(active_i) > 4:
                order = np.argsort(-priority[active_i])
                for idx in active_i[order[4:]]:
                    drop.add(int(idx))
            t += 0.050
        instr.notes = [n for i, n in enumerate(notes) if i not in drop]

    # write our midi to the output file
    pm.instruments.append(instr)
    pm.write(OUTPUT_MIDI)

    elapsed = time.time() - start_time
    print(f"{WAV_FILE}")
    print(f"  Time: {elapsed:.2f}s  |  Peak RAM: {peak_memory:.1f} MB  |  Chunks Processed: {frame_count}")
    print()
