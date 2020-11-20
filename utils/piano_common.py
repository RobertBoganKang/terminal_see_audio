import numpy as np

from utils.analyze_common import AnalyzeCommon


class PianoCommon(AnalyzeCommon):
    def __init__(self):
        super().__init__()
        self.piano_tuning_shape_power = 1 / 2
        self.piano_key_range = [-48, 40]
        self.piano_spectral_height = 0.1
        self.piano_position_gap = 0.4
        self.piano_line_width = 0.8

        # `0` as white, `1` as black key
        self.piano_key_bw_switch = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]

        # colors & themes
        self.piano_base_color = '#222'
        self.piano_roll_base_color = '#444'
        self.piano_roll_black_key_color = '#333'
        self.piano_key_color = 'mediumspringgreen'
        self.piano_roll_color = 'mediumspringgreen'
        self.piano_spectral_color = 'dimgray'

    def _piano_tuning_method(self, key, value):
        # set `^` or `n` shape tuning
        return np.mean(value[:, 0] * np.power(1 - 2 * np.abs(key - value[:, 1]), self.piano_tuning_shape_power))

    def _piano_key_spectral_data(self, array):
        key_dict = {}
        raw_keys = []
        key_ffts = []
        for i, t in enumerate(array):
            if i > 0:
                raw_key = self._frequency_to_key(self._fft_position_to_frequency(i))
                key = round(raw_key)
                if self.piano_key_range[0] <= key < self.piano_key_range[1]:
                    raw_keys.append(raw_key)
                    key_ffts.append(t)
                    if key not in key_dict:
                        key_dict[key] = [[t, raw_key]]
                    else:
                        key_dict[key].append([t, raw_key])
        for k, v in key_dict.items():
            v = np.array(v)
            key_dict[k] = self._piano_tuning_method(k, v)
        max_value = max(list(key_dict.values()))
        for k, v in key_dict.items():
            key_dict[k] = v / max_value
        return key_dict, raw_keys, key_ffts
