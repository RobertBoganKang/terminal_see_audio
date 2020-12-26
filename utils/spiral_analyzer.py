import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from utils.analyze_common import AnalyzeCommon


class SpiralAnalyzer(AnalyzeCommon):
    def __init__(self):
        super().__init__()

        # spiral analyzer
        self.spiral_dpi = 150
        self.spiral_figure_size = (15, 15)
        self.spiral_line_width = 1.5
        # default for 12 equal temperament
        self.spiral_n_temperament = 12

        # color & themes
        self.spiral_axis_color = '#444'

    @staticmethod
    def _spiral_pitch_to_plot_position(pitch, offset):
        x_position = np.cos(pitch * 2 * np.pi + np.pi) * (pitch + offset)
        y_position = -np.sin(pitch * 2 * np.pi + np.pi) * (pitch + offset)
        return x_position, y_position

    def _spiral_polar_transform(self, arrays, s, v):
        array_0 = arrays[0]
        array_1 = arrays[1]
        x_array_0 = []
        y_array_0 = []
        x_array_1 = []
        y_array_1 = []
        pitches = []
        keys = []
        vs = []
        ss = []
        for i in range(len(array_0)):
            t0 = array_0[i]
            t1 = array_1[i]
            # skip low frequency part
            if i > 0:
                frequency = self._fft_position_to_frequency(i)
                pitch = self._frequency_to_pitch(frequency)
                key = self._frequency_to_key(frequency)
                if pitch > 0:
                    vs.append(v[i])
                    ss.append(s[i])
                    pitches.append(pitch)
                    keys.append(key)
                    x_position, y_position = self._spiral_pitch_to_plot_position(pitch, -t1 / 2)
                    x_array_0.append(x_position)
                    y_array_0.append(y_position)
                    x_position, y_position = self._spiral_pitch_to_plot_position(pitch, t0 / 2)
                    x_array_1.append(x_position)
                    y_array_1.append(y_position)
        return (x_array_0, y_array_0), (x_array_1, y_array_1), pitches, keys, ss, vs

    def _prepare_graph_spiral(self, starting_time, save_path, dynamic_max_value=False):
        valid = self._check_analyze_duration(starting_time)
        if not valid:
            return False
        else:
            # prepare data
            (fft_data, log_fft_data, s_fft_magnitude_ratio_data,
             v_fft_data) = self._analyze_two_channels_data_preparation(starting_time,
                                                                       dynamic_max_value=dynamic_max_value)
            # prepare position info
            (position_0, position_1, pitches, keys, ss, vs) = self._spiral_polar_transform(log_fft_data,
                                                                                           s_fft_magnitude_ratio_data,
                                                                                           v_fft_data)
            min_pitch = pitches[0]
            # pitch ticks for `n` temperament
            pitch_ticks_end = [
                (x + int(self._frequency_to_pitch(self.sample_rate / 2) * self.spiral_n_temperament)
                 - (self.spiral_n_temperament - 1)) / self.spiral_n_temperament for x in
                range(self.spiral_n_temperament)]
            pitch_ticks_start = [x - int(x - min_pitch) for x in pitch_ticks_end]
            pitch_end_ticks_position = [self._spiral_pitch_to_plot_position(x, 0) for x in pitch_ticks_end]
            pitch_start_ticks_position = [self._spiral_pitch_to_plot_position(x, 0) for x in pitch_ticks_start]
            ax_position, ay_position = self._spiral_pitch_to_plot_position(5, 0)

            # making plots
            fig = plt.figure(figsize=self.spiral_figure_size)
            ax = fig.add_subplot(111)

            # plot ticks for `n` temperament
            for i in range(len(pitch_end_ticks_position)):
                ax.plot([pitch_start_ticks_position[i][0], pitch_end_ticks_position[i][0]],
                        [pitch_start_ticks_position[i][1], pitch_end_ticks_position[i][1]],
                        c=self.spiral_axis_color, zorder=1, alpha=0.4, linewidth=self.spiral_line_width)
                cir_start = Circle((pitch_start_ticks_position[i][0], pitch_start_ticks_position[i][1]), radius=0.05,
                                   zorder=1, facecolor='black', edgecolor=self.spiral_axis_color, alpha=0.4)
                cir_end = Circle((pitch_end_ticks_position[i][0], pitch_end_ticks_position[i][1]), radius=0.05,
                                 zorder=1, facecolor='black', edgecolor=self.spiral_axis_color, alpha=0.4)
                ax.add_patch(cir_start)
                ax.add_patch(cir_end)

            # plot base axis
            ax.plot(position_0[0], position_0[1], c=self.spiral_axis_color, linewidth=self.spiral_line_width, zorder=1,
                    alpha=0.4)

            # plot spiral
            for i in range(len(position_0[0]) - 1):
                pos0 = [position_0[0][i], position_0[1][i]]
                pos1 = [position_1[0][i], position_1[1][i]]
                pos2 = [position_1[0][i + 1], position_1[1][i + 1]]
                pos3 = [position_0[0][i + 1], position_0[1][i + 1]]
                poly_position = np.array([pos0, pos1, pos2, pos3])
                k = keys[i]
                v_opacity = max(vs[i], vs[i + 1])
                if v_opacity > self.figure_minimum_alpha:
                    if self.colorful_theme:
                        color = self._hsb_to_rgb((k - 3) % 12 / 12, ss[i], 1)
                    else:
                        color = self.mono_theme_color
                    ax.fill(poly_position[:, 0], poly_position[:, 1], facecolor=color,
                            edgecolor=color, linewidth=self.spiral_line_width,
                            alpha=v_opacity, zorder=2)
            # plot `A4` position
            cir_end = Circle((ax_position, ay_position), radius=0.2, zorder=3, facecolor=self.a_pitch_color,
                             linewidth=self.spiral_line_width, edgecolor=self.a_pitch_color, alpha=0.6)
            ax.add_patch(cir_end)

            # set figure ratio
            self._set_1to1_ratio_figure()

            # save figure
            fig.savefig(save_path, dpi=self.spiral_dpi, bbox_inches='tight')
            self._matplotlib_clear_memory(fig)

            # prepare ifft play
            if not dynamic_max_value:
                self._ifft_audio_export(self._analyze_log_min_max_transform(fft_data, log=False))
            return True

    def _prepare_video_spiral(self, starting_time, ending_time, save_path):
        """ save video for spiral """
        starting_time, ending_time, status = self._prepare_video_analyzer(starting_time, ending_time,
                                                                          save_analyzer_path=save_path,
                                                                          analyzer_function=self._prepare_graph_spiral)
        return starting_time, ending_time, status
