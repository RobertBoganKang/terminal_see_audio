import numpy as np

from utils.analyze_common import AnalyzeCommon


class PianoCommon(AnalyzeCommon):
    def __init__(self):
        super().__init__()
        # tuning coefficient
        self.piano_tuning_n_shape_coefficient = 1 / 2
        self.piano_tuning_bell_shape_coefficient_a = 4
        self.piano_tuning_bell_shape_coefficient_b = 2
        self.piano_tuning_shape_function = 'bell'

        # key range
        self.piano_key_range = [-48, 40]
        self.piano_key_chroma_range = [-9, -9 + 24]

        # piano dimensions
        self.piano_spectral_height = 0.1
        self.piano_position_gap = 0.4
        self.piano_line_width = 0.8

        # `0` as white, `1` as black key
        self.piano_key_bw_switch = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]

        # colors & themes
        self.piano_base_color = '#222'
        self.piano_roll_base_color = '#444'
        self.piano_roll_black_key_color = '#333'
        self.piano_spectral_color = 'dimgray'

        # piano key alpha color transform with power function
        self.piano_key_color_transform_power = 2

    def _piano_tuning_n_shape_function(self, key_diff):
        """
        key_diff: -1 ~ 1
        shape: (1 - abs(x)) ^ a
        default: a=1/2
          _-^-_
        .|     |.
        """
        return np.power(1 - np.abs(key_diff), self.piano_tuning_n_shape_coefficient)

    def _piano_tuning_bell_shape_function(self, key_diff):
        """
        key_diff: -1 ~ 1
        shape: (1 - x^b) ^ a
        default: b=2, a=4
           _-_
        __/  \\__
        """
        return np.power((1 - np.power(key_diff, self.piano_tuning_bell_shape_coefficient_b)),
                        self.piano_tuning_bell_shape_coefficient_a)

    def _piano_tuning_method(self, key, value):
        # get key difference for tuning
        key_diff = 2 * (key - value[:, 1])
        # set `^` / `n` or bell shape tuning
        if self.piano_tuning_shape_function == 'bell':
            tuning = self._piano_tuning_bell_shape_function(key_diff)
        elif self.piano_tuning_shape_function == 'n':
            tuning = self._piano_tuning_n_shape_function(key_diff)
        else:
            raise ValueError('tuning shape parameter error')
        return np.mean(value[:, 0] * tuning)

    def _piano_chroma_key_dict_transform(self, key_dict):
        chroma_key_dict = {}
        # collect key energy
        for k, v in key_dict.items():
            # shrink to one octave starting at `C`
            chroma_key = (k + 9) % 12 - 9
            if chroma_key not in chroma_key_dict:
                chroma_key_dict[chroma_key] = [v]
            else:
                chroma_key_dict[chroma_key].append(v)
        # get average overview
        key_dict = {}
        for k, v in chroma_key_dict.items():
            value = np.mean(v)
            key_dict[k] = value
            # add one octave (more octaves for display)
            kk = k
            while kk + 12 < self.piano_key_chroma_range[1]:
                key_dict[kk + 12] = value
                kk += 12
            kk = k
            while kk - 12 >= self.piano_key_chroma_range[0]:
                key_dict[kk - 12] = value
                kk -= 12
        return key_dict

    def _piano_key_spectral_data(self, array, chroma=False):
        key_dict = {}
        raw_keys = []
        key_ffts = []
        for i, t in enumerate(array):
            if i > 0:
                raw_key = self._frequency_to_key(self._fft_position_to_frequency(i))
                key = round(raw_key)
                # chroma mode will consider wider range around 8kHz
                if chroma:
                    status = self.piano_key_range[0] <= key < self.piano_key_range[1] + 12
                else:
                    status = self.piano_key_range[0] <= key < self.piano_key_range[1]
                if status:
                    raw_keys.append(raw_key)
                    key_ffts.append(t)
                    if key not in key_dict:
                        key_dict[key] = [[t, raw_key]]
                    else:
                        key_dict[key].append([t, raw_key])
        for k, v in key_dict.items():
            v = np.array(v)
            key_dict[k] = self._piano_tuning_method(k, v)

        # chroma key dict transform
        if chroma:
            key_dict = self._piano_chroma_key_dict_transform(key_dict)

        # modify max value to get more stable video graphics change
        max_value = max(list(key_dict.values()))
        self.analyze_log_piano_key_max_value = (self.analyze_log_piano_key_max_value +
                                                (max_value - self.analyze_log_piano_key_max_value) / (
                                                        self.analyze_log_piano_key_max_value_reach_time *
                                                        self.analyze_video_frame_rate))
        modified_max_value = max(max_value, self.analyze_log_piano_key_max_value)
        for k, v in key_dict.items():
            if modified_max_value != 0:
                key_dict[k] = v / modified_max_value
        return key_dict, raw_keys, key_ffts

    def _piano_get_range(self, chroma):
        if chroma:
            piano_key_range = self.piano_key_chroma_range
        else:
            piano_key_range = self.piano_key_range
        return piano_key_range
