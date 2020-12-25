import matplotlib.pyplot as plt
import numpy as np

from utils.piano_common import PianoCommon


class PianoAnalyzer(PianoCommon):
    def __init__(self):
        super().__init__()
        # piano analyzer
        self.piano_figure_size = (15, 5)
        self.piano_dpi = 300
        self.piano_cover_width = 0.1
        # piano key size
        # b/w width: 13.7mm/23.5mm
        # b/w length: 9.5mm/15cm
        # assume white key width is 1
        self.piano_white_key_length = 15 / 2.35
        self.piano_black_key_width = 13.7 / 23.5
        self.piano_black_key_length = 9.5 / 15 * self.piano_white_key_length

    def _piano_generate_frequency_graph_single(self, ax, raw_key, key_fft, channel):
        # get key position
        positions_0 = []
        positions_1 = []
        for i in range(len(raw_key)):
            one_piano_length = self.piano_white_key_length + self.piano_spectral_height + self.piano_position_gap
            key_x = raw_key[i] * 7 / 12
            positions_0.append([key_x, -channel * one_piano_length])
            positions_1.append([key_x, -channel * one_piano_length + self.piano_spectral_height])
        for i in range(len(raw_key) - 1):
            x_positions = [positions_0[i][0], positions_1[i + 1][0], positions_1[i + 1][0], positions_0[i][0]]
            y_positions = [positions_0[i][1], positions_0[i + 1][1], positions_1[i + 1][1], positions_1[i][1]]
            freq_alpha = max(key_fft[i], key_fft[i + 1])
            if freq_alpha > self.figure_minimum_alpha:
                ax.fill(x_positions, y_positions, edgecolor=self.piano_spectral_color,
                        facecolor=self.piano_spectral_color,
                        linewidth=self.piano_line_width, zorder=1, alpha=freq_alpha)

    def _piano_data_prepare(self, starting_time, dynamic_max_value=False, chroma=False):
        # get starting sample index
        starting_sample = int(starting_time * self.sample_rate)
        # get data for spectral
        if chroma:
            data = np.array([np.mean(self.data, axis=0)])
        else:
            data = self.data
        audio_data = [x[starting_sample:starting_sample + self.analyze_n_window] for x in data]
        fft_data = [self._fft_data_transform_single(x)[0] for x in audio_data]
        log_fft_data = self._analyze_log_min_max_transform(fft_data, dynamic_max_value=dynamic_max_value)
        spectral_data = [self._piano_key_spectral_data(x, chroma=chroma) for x in log_fft_data]
        key_dicts = []
        raw_keys = []
        key_ffts = []
        for key_dict, raw_key, key_fft, in spectral_data:
            key_dicts.append(key_dict)
            raw_keys.append(raw_key)
            key_ffts.append(key_fft)
        return fft_data, key_dicts, raw_keys, key_ffts

    def _piano_graph_single(self, ax, key_dict, channel, piano_key_range):
        # plot cover
        left_most, _ = self._piano_generate_key_position(piano_key_range[0], channel)
        right_most, _ = self._piano_generate_key_position(piano_key_range[1] - 1, channel)
        cover_x_positions = [left_most[0, 0], right_most[1, 0], right_most[1, 0], left_most[0, 0]]
        cover_y_positions = [left_most[0, 1], left_most[0, 1], left_most[0, 1] - self.piano_cover_width,
                             left_most[0, 1] - self.piano_cover_width]
        ax.fill(cover_x_positions, cover_y_positions, edgecolor=self.piano_base_color, facecolor=self.piano_base_color,
                linewidth=self.piano_line_width, zorder=5, alpha=0.9)
        # plot key
        for k in range(piano_key_range[0], piano_key_range[1], 1):
            positions, bw = self._piano_generate_key_position(k, channel)
            if k in key_dict:
                fft_value = key_dict[k]
            else:
                fft_value = 0
            # background
            ax.fill(positions[:, 0], positions[:, 1], facecolor='black', edgecolor=self.piano_base_color,
                    linewidth=self.piano_line_width, zorder=2 * bw + 1)
            # plot key
            alpha = fft_value ** self.piano_key_color_transform_power
            if alpha > self.figure_minimum_alpha:
                ax.fill(positions[:, 0], positions[:, 1], edgecolor=self.piano_key_color,
                        facecolor=self.piano_key_color,
                        linewidth=self.piano_line_width, zorder=2 * bw + 2, alpha=alpha)
            # `a4` position
            if k % 12 == 0:
                if k == 0:
                    opacity = 0.5
                else:
                    opacity = 0.15
                ax.fill(positions[:, 0], cover_y_positions,
                        edgecolor=self.a_pitch_color, facecolor=self.a_pitch_color, linewidth=self.piano_line_width,
                        zorder=6, alpha=opacity)

    def _piano_generate_key_position(self, key, channel):
        # `7` for octave switch
        key_position_switch = [0, 0.5, 1, 2, 2.5, 3, 3.5, 4, 5, 5.5, 6, 6.5]
        key_position = key % 12
        key_octave = int(np.ceil((key + 0.5) / 12)) - 1
        middle_x = key_octave * 7 + key_position_switch[key_position]
        # get position dimension
        if self.piano_key_bw_switch[key_position] == 0:
            width = 1
            length = self.piano_white_key_length
        else:
            width = self.piano_black_key_width
            length = self.piano_black_key_length
        # key position
        one_piano_length = self.piano_white_key_length + self.piano_spectral_height + self.piano_position_gap
        position_0 = [middle_x - width / 2, -channel * one_piano_length]
        position_1 = [middle_x + width / 2, -channel * one_piano_length]
        position_2 = [middle_x + width / 2, -channel * one_piano_length - length]
        position_3 = [middle_x - width / 2, -channel * one_piano_length - length]
        return np.array([position_0, position_1, position_2, position_3]), self.piano_key_bw_switch[key_position]

    def _prepare_graph_piano(self, starting_time, save_path, dynamic_max_value=False, chroma=False):
        valid = self._check_analyze_duration(starting_time)
        if not valid:
            return False
        else:
            fft_data, key_dicts, raw_keys, key_ffts = self._piano_data_prepare(starting_time,
                                                                               dynamic_max_value=dynamic_max_value,
                                                                               chroma=chroma)
            # set range
            piano_key_range = self._piano_get_range(chroma)
            # plot
            fig = plt.figure(figsize=self.piano_figure_size)
            ax = fig.add_subplot(111)
            ax.set_xlim(
                [self._piano_generate_key_position(piano_key_range[0], 0)[0][0, 0] - 0.5,
                 self._piano_generate_key_position(piano_key_range[1] - 1, 0)[0][
                     1, 0] + 0.5])
            ax.set_ylim([-0.2 - self.piano_white_key_length * len(fft_data) - (
                    self.piano_spectral_height + self.piano_position_gap) * (len(fft_data) - 1),
                         0.2 + self.piano_spectral_height])
            # plot piano
            for i in range(len(fft_data)):
                self._piano_graph_single(ax, key_dicts[i], i, piano_key_range)
                if not chroma:
                    self._piano_generate_frequency_graph_single(ax, raw_keys[i], key_ffts[i], i)

            # set plot ratio
            self._set_1to1_ratio_figure()

            # save figure
            fig.savefig(save_path, dpi=self.piano_dpi, bbox_inches='tight')
            self._matplotlib_clear_memory(fig)

            # prepare ifft play
            if not dynamic_max_value:
                self._ifft_audio_export(self._analyze_log_min_max_transform(fft_data, log=False))
            return True

    def _prepare_video_piano(self, starting_time, ending_time, save_path):
        """ save video for piano """
        starting_time, ending_time, status = self._prepare_video_analyzer(starting_time, ending_time,
                                                                          save_analyzer_path=save_path,
                                                                          analyzer_function=self._prepare_graph_piano)
        return starting_time, ending_time, status
