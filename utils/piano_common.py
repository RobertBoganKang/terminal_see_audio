import numpy as np

from utils.analyze_common import AnalyzeCommon


class PianoCommon(AnalyzeCommon):
    def __init__(self):
        super().__init__()
        self.piano_tuning_n_shape_coef = 1 / 2
        self.piano_tuning_bell_shape_coef = 4
        self.piano_tuning_shape_function = 'bell'
        self.piano_key_range = [-48, 40]
        self.piano_key_chroma_range = [-9, 3]
        self.piano_chroma_ignore_energy_ratio = 0.01

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

        # piano key alpha color transform with power function
        self.piano_key_color_transform_power = 2

    def _piano_tuning_n_shape_function(self, key, value):
        """
        shape: (1 - abs(x)) ^ a
          _-^-_
        .|     |.
        """
        return np.power(1 - 2 * np.abs(key - value[:, 1]), self.piano_tuning_n_shape_coef)

    def _piano_tuning_bell_shape_function(self, key, value):
        """
        shape: (1 - x^2) ^ a
           _-_
        __/  \\__
        """
        return np.power(2 * (key - value[:, 1]), self.piano_tuning_bell_shape_coef)

    def _piano_tuning_method(self, key, value):
        # set `^` / `n` or bell shape tuning
        if self.piano_tuning_shape_function == 'bell':
            tuning = self._piano_tuning_bell_shape_function(key, value)
        elif self.piano_tuning_shape_function == 'n':
            tuning = self._piano_tuning_n_shape_function(key, value)
        else:
            raise ValueError('shape parameter error')
        return np.mean(value[:, 0] * tuning)

    def _piano_key_spectral_data(self, array, chroma=True):
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

        # chroma mode
        if chroma:
            max_value = max(list(key_dict.values()))
            chroma_key_dict = {}
            # collect key energy
            for k, v in key_dict.items():
                # shrink to one octave starting at `C`
                chroma_key = (k + 9) % 12 - 9
                # ignore not important peaks
                if v > self.piano_chroma_ignore_energy_ratio * max_value:
                    if chroma_key not in chroma_key_dict:
                        chroma_key_dict[chroma_key] = [v]
                    else:
                        chroma_key_dict[chroma_key].append(v)
            # get average overview
            for k, v in chroma_key_dict.items():
                chroma_key_dict[k] = np.mean(v)
            key_dict = chroma_key_dict

        # modify max value to get more stable video graphics change
        max_value = max(list(key_dict.values()))
        self.analyze_log_piano_key_max_value = (self.analyze_log_piano_key_max_value +
                                                (max_value - self.analyze_log_piano_key_max_value) / (
                                                        self.analyze_log_piano_key_max_value_reach_time *
                                                        self.analyze_video_frame_rate)
                                                )
        modified_max_value = max(max_value, self.analyze_log_piano_key_max_value)
        for k, v in key_dict.items():
            if modified_max_value != 0:
                key_dict[k] = v / modified_max_value
        return key_dict, raw_keys, key_ffts
