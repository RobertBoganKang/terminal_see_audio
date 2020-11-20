import numpy as np
import soundfile as sf

from utils.common import Common


class AnalyzeCommon(Common):
    def __init__(self):
        super().__init__()
        # key names
        self.note_name_lib = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        self.note_name_lib_flat = ['A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab']

    def _analyze_log_min_max_transform(self, array, log=True):
        if log:
            array = np.log(np.array(array) + self.min_analyze_power)
        array = self._max_norm(array, min_transform=True)
        return array

    def _fft_position_to_frequency(self, position):
        return position * self.sample_rate / self.analyzer_n_window

    def _frequency_to_pitch(self, frequency):
        return np.log2(frequency / self.a4_frequency) + 5

    def _key_to_frequency(self, key, remainder=0):
        return 2 ** (key / 12 + remainder / 1200) * self.a4_frequency

    def _frequency_to_key(self, frequency):
        return np.log2(frequency / self.a4_frequency) * 12

    def _fft_data_transform_single(self, audio_data_single, phase=False):
        if not phase:
            fft_single = self._calc_sp(audio_data_single, self.analyzer_n_window, self.n_overlap)
            return fft_single
        else:
            fft_single, phase_single = self._calc_sp(audio_data_single, self.analyzer_n_window, self.n_overlap,
                                                     angle=True)
            return fft_single, phase_single

    def _check_analyze_duration(self, starting_time):
        """ check if raw audio too short for analyze plot """
        if self._get_audio_time() < self.analyze_min_duration + starting_time or starting_time < 0:
            return False
        else:
            return True

    def _get_ifft_data_single(self, fft_single):
        ff1 = np.array(list(fft_single) + list(-fft_single[::-1]))
        ifft_single = np.real(np.fft.ifft(ff1))
        ifft_single /= np.max(np.abs(ifft_single))
        return ifft_single[:self.analyzer_n_window]

    def _ifft_audio_export(self, fft_data):
        ifft_data = np.transpose(np.array([self._get_ifft_data_single(x) for x in fft_data]))
        sf.write(self.ifft_audio_path, ifft_data, samplerate=self.sample_rate)

    def _translate_music_note_name_to_frequency(self, string):
        """ translate music notes """
        # get note name
        if string[1] in ['#', 'b']:
            note_name = string[0].upper() + string[1]
            i = 2
        else:
            note_name = string[0].upper()
            i = 1
        key_position = None
        if note_name in self.note_name_lib:
            key_position = self.note_name_lib.index(note_name)
        elif note_name in self.note_name_lib_flat:
            key_position = self.note_name_lib_flat.index(note_name)
        if key_position is None:
            return None

        # get note octave
        j = i
        while j < len(string):
            if string[j] not in ['+', '-']:
                j += 1
            else:
                break
        note_octave = string[i:j]
        if not self._is_int(note_octave):
            return None
        key_octave = int(note_octave) - 4
        if key_position >= 3:
            key_octave -= 1
        # if too large `>20kHz`
        if key_octave > 6:
            return None

        # get cents
        if j != len(string):
            if string[-1].lower() == 'c':
                k = len(string) - 1
            else:
                k = len(string)
            note_cent = string[j:k]
            if not self._is_float(note_cent):
                return None
            key_cent = float(note_cent)
        else:
            key_cent = 0

        # combine to frequency
        frequency = self._key_to_frequency(key_position + 12 * key_octave, remainder=key_cent)
        return frequency

    def _translate_frequency_to_music_note(self, frequency):
        raw_key = np.log2(frequency / 440) * 12
        key = round(raw_key)
        remainder_in_cent = (raw_key - key) * 100
        key_name = self.note_name_lib[key % 12]
        key_octave = (key + 9) // 12 + 4
        return key_name, key_octave, remainder_in_cent

    def _analyze_get_audio_fft_data(self, starting_time, phase=False):
        # get starting sample index
        starting_sample = int(starting_time * self.sample_rate)
        # get data for spectral
        if len(self.data) != 2:
            audio_data = np.sum(self.data, axis=0)[starting_sample:starting_sample + self.analyzer_n_window]
            audio_data = [audio_data, audio_data]
        else:
            audio_data = [x[starting_sample:starting_sample + self.analyzer_n_window] for x in self.data]
        if not phase:
            fft_data = [self._fft_data_transform_single(x)[0] for x in audio_data]
            return np.array(fft_data)
        else:
            fft_data = []
            phase_data = []
            for ad in audio_data:
                fft_single, phase_single = self._fft_data_transform_single(ad, phase=True)
                fft_data.append(fft_single[0])
                phase_data.append(phase_single[0])
            return np.array(fft_data), np.array(phase_data)

    def _translate_string_to_frequency(self, string):
        """ translate command to frequency """
        string = string.strip()
        if 'hz' in string.lower():
            freq = string[:-2]
            if self._is_float(freq):
                return float(freq)
            else:
                return None
        else:
            return self._translate_music_note_name_to_frequency(string)

    def _analyze_two_channels_data_preparation(self, starting_time):
        phase = self._phase_mode_check()
        if not phase:
            # prepare fft data
            fft_data = self._analyze_get_audio_fft_data(starting_time)
            log_fft_data = self._analyze_log_min_max_transform(fft_data)
            v_fft_data = self._max_norm(log_fft_data[0] + log_fft_data[1], min_transform=False)
            return fft_data, log_fft_data, None, None, v_fft_data
        else:
            fft_data, phase_data = self._analyze_get_audio_fft_data(starting_time, phase=True)
            log_fft_data = self._analyze_log_min_max_transform(fft_data)
            v_fft_data = self._max_norm(log_fft_data[0] + log_fft_data[1], min_transform=False)
            h_phase_data = phase_data[1] - phase_data[0]
            h_phase_data = np.mod(h_phase_data / 2 / np.pi, 1)
            s_fft_magnitude_diff_data = np.abs(log_fft_data[0] - log_fft_data[1])
            return fft_data, log_fft_data, h_phase_data, s_fft_magnitude_diff_data, v_fft_data
