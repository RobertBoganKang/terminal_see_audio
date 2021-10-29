import numpy as np
import soundfile as sf

from utils.analyze_common import AnalyzeCommon


class PlayPitch(AnalyzeCommon):
    def __init__(self):
        super().__init__()
        self.pitch_head_cap_duration = 0.1
        self.pitch_end_cap_duration = 0.3

    def _pitch_build_caps(self):
        head_num_sample = self.pitch_head_cap_duration * self.sample_rate
        end_num_sample = self.pitch_end_cap_duration * self.sample_rate
        sample_heading = np.blackman(int(head_num_sample * 2))[:int(head_num_sample)]
        sample_ending = np.blackman(int(end_num_sample * 2))[-int(end_num_sample):]
        return sample_heading, sample_ending

    def _pitch_extract_space_info(self, frequency, angle, distance):
        e = self.ear_distance / 2
        v = self._analyze_sound_speed()
        f = frequency
        ll = distance
        d0 = np.sqrt((ll * np.sin(angle / 180 * np.pi) + e) ** 2 + (ll * np.cos(angle / 180 * np.pi) ** 2))
        d1 = np.sqrt((ll * np.sin(angle / 180 * np.pi) - e) ** 2 + (ll * np.cos(angle / 180 * np.pi) ** 2))
        d = d0 - d1
        phi = d * f / v
        rt = d1 / d0
        if rt > 1:
            amp = [1, 1 / rt]
        else:
            amp = [rt, 1]
        return amp, phi

    def _pitch_generate_wave(self, frequency, angle, distance, duration):
        """ get wave data for specific frequency """
        wave_data_0 = []
        wave_data_1 = []
        amp, phi = self._pitch_extract_space_info(frequency, angle, distance)
        for i in range(int(duration * self.sample_rate)):
            t = (i / self.sample_rate)
            wave_data_0.append(amp[0] * np.sin(frequency * 2 * np.pi * t - 2 * np.pi * phi))
            wave_data_1.append(amp[1] * np.sin(frequency * 2 * np.pi * t))
        wave_data_0 = np.array(wave_data_0)
        wave_data_1 = np.array(wave_data_1)
        # apply cap
        head_cap, end_cap = self._pitch_build_caps()
        wave_data_0[:len(head_cap)] *= head_cap
        wave_data_1[:len(head_cap)] *= head_cap
        wave_data_0[-len(end_cap):] *= end_cap
        wave_data_1[-len(end_cap):] *= end_cap
        return np.transpose([wave_data_0, wave_data_1])

    def _pitch_export_wave_frequency(self, string):
        inputs = string.split()
        frequency = None
        # default settings
        distance = 1
        angle = 0
        duration = 0.8
        # get inputs
        for input_ in inputs:
            frequency_candidate = self._translate_string_to_frequency(input_)
            if frequency_candidate is not None:
                frequency = frequency_candidate
            if input_.lower().endswith('m'):
                if self._is_float(input_[:-1]):
                    distance = float(input_[:-1])
            if input_.lower().endswith('deg'):
                if self._is_float(input_[:-3]):
                    angle = float(input_[:-3])
            if input_.lower().endswith('rad'):
                if self._is_float(input_[:-3]):
                    angle = float(input_[:-3]) / np.pi * 180
            if input_.lower().endswith('s'):
                if self._is_float(input_[:-1]):
                    duration = float(input_[:-1])
        if frequency is None:
            print(f'<!> pitch input `{string}` unknown or frequency too high')
            return False
        elif duration < self.pitch_head_cap_duration + self.pitch_end_cap_duration:
            print(f'<!> play duration `{duration}` too short '
                  f'(< {self.pitch_head_cap_duration + self.pitch_end_cap_duration})')
            return False
        else:
            show_frequency = round(frequency, 3)
            if frequency > self.max_hearing_frequency:
                print(f'<!> higher than hearing frequency `{show_frequency}Hz` (>`{self.max_hearing_frequency}Hz`)')
                return False
            print(f'<*> `{show_frequency}Hz` ~~> `{distance}m` ~~> `{round(angle, 3)}deg` ~~> `{round(duration)}s`')
            wave_data = self._pitch_generate_wave(frequency, angle, distance, duration)
            # export wave
            sf.write(self.pitch_audio_path, wave_data, samplerate=self.sample_rate)
            return True
