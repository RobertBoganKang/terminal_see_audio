import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Arc

from utils.analyze_common import AnalyzeCommon


class StringsAnalyzer(AnalyzeCommon):
    def __init__(self):
        super().__init__()
        # strings analyzer
        self.strings_dpi = 300
        self.strings_figure_size = (15, 5)
        self.strings_frequency_line_width = 0.8
        self.strings_line_width = 2
        self.strings_ticks_line_width = 1
        self.strings_line_width_power_base = 0.7
        self.strings_line_width_harmonics = 1.2
        self.strings_spectral_ratio_power_transform = 0.4
        self.strings_cut_diff_ratio_ratio_power_transform = 0.5
        self.strings_natural_harmonics_size_power_coefficient = 0.8
        # default for 12 equal temperament
        self.strings_n_temperament = 12

        # max fractal of natural harmonics
        self.strings_max_fractal_natural_harmonics = 6
        self.strings_roman_numbers = [self._integer_to_roman(x, upper_case=False) for x in
                                      range(self.strings_max_fractal_natural_harmonics + 1)]

        # number of ticks showing
        self.strings_ticks_octave = 5

        # string length
        self.strings_length = 40

        # color & themes
        self.strings_axis_color = '#444'
        self.strings_ticks_color = '#888'

        # strings name or frequency
        self.strings_pitch_name_or_frequency = np.array([4 / 9, 2 / 3, 1, 3 / 2]) * self.a4_frequency

    def _strings_frequency_to_plot_position(self, ratio, string_i, offset):
        x_position = -1 / ratio * self.strings_length
        y_position = string_i + offset
        return x_position, y_position

    def _strings_fractals_generation(self, max_fractal):
        # error handling
        if not isinstance(max_fractal, int) or max_fractal <= 0:
            return []

        # get fractal result
        result = []
        for denominator in range(1, max_fractal + 1):
            for numerator in range(1, denominator):
                n, d = self._simplify_fractal(numerator, denominator)
                if d == denominator:
                    result.append([n, d])
        return result

    def _strings_natural_harmonics_info(self, reference_frequency, string_i, log_fft_data, max_fractal=6):
        log_fft_data_mean = np.mean(log_fft_data, axis=0)
        fractals = self._strings_fractals_generation(max_fractal)
        result = []
        for num, den in fractals:
            harmonic_frequency = reference_frequency * den
            harmonic_fft_position = self._frequency_to_fft_position(harmonic_frequency)
            harmonic_log_fft_power = log_fft_data_mean[harmonic_fft_position]
            if harmonic_log_fft_power > self.min_analyze_power:
                pitch = self._frequency_to_pitch_color(harmonic_frequency)
                ratio = den / num
                transform_ratio = self._strings_spectral_transform_ratio(ratio)
                x_position, y_position = self._strings_frequency_to_plot_position(ratio, string_i, 0)
                result.append([harmonic_log_fft_power, pitch, harmonic_fft_position,
                               x_position, y_position, num, den, transform_ratio])
        return result

    def _strings_spectral_transform_ratio(self, ratio):
        """ the width of spectral will shrink if higher notes """
        return 1 / ratio ** self.strings_spectral_ratio_power_transform

    def _strings_position_transform(self, arrays, v, reference_frequency, string_i):
        array_0 = arrays[0]
        array_1 = arrays[1]
        x_array_0 = []
        y_array_0 = []
        x_array_1 = []
        y_array_1 = []
        pitches = []
        vs = []
        i_s = []
        low_frequency_to_show = reference_frequency * (2 ** (-1 / self.strings_n_temperament))
        for i in range(len(array_0)):
            t0 = array_0[i]
            t1 = array_1[i]
            # skip low frequency part
            if i > 0:
                frequency = self._fft_position_to_frequency(i)
                if frequency >= low_frequency_to_show:
                    pitch = self._frequency_to_pitch_color(frequency)
                    ratio = frequency / reference_frequency
                    # cut the left edge
                    if frequency < reference_frequency:
                        cut_diff_ratio = 1 - (reference_frequency / frequency) / (
                                1 / 2 ** (-1 / self.strings_n_temperament))
                        cut_diff_ratio = cut_diff_ratio ** self.strings_cut_diff_ratio_ratio_power_transform
                    else:
                        cut_diff_ratio = 1
                    transform_ratio = self._strings_spectral_transform_ratio(ratio)
                    vs.append(v[i])
                    i_s.append(i)
                    pitches.append(pitch)
                    x_position, y_position = self._strings_frequency_to_plot_position(ratio, string_i,
                                                                                      t0 / 2 * transform_ratio
                                                                                      * cut_diff_ratio)
                    x_array_0.append(x_position)
                    y_array_0.append(y_position)
                    x_position, y_position = self._strings_frequency_to_plot_position(ratio, string_i,
                                                                                      -t1 / 2 * transform_ratio
                                                                                      * cut_diff_ratio)
                    x_array_1.append(x_position)
                    y_array_1.append(y_position)
        return (x_array_0, y_array_0), (x_array_1, y_array_1), pitches, vs, i_s

    def _strings_plot_ticks(self, ax, strings_frequencies):
        for i in range(self.strings_ticks_octave * self.strings_n_temperament):
            color_cir = self._hsb_to_rgb(i / self.strings_n_temperament % 1, 1, 1)
            x_position = -self.strings_length / 2 ** (i / self.strings_n_temperament)
            if i % self.strings_n_temperament == 0:
                tick_ratio = 3
                cir_ratio = 2
                bottom_position = -1
                cir_bottom_position = -1
                tick_color = self.strings_ticks_color
            else:
                tick_ratio = 1
                cir_ratio = 1
                bottom_position = 0
                cir_bottom_position = -0.7
                tick_color = self.strings_axis_color
            ratio = 0.98 ** i
            ax.plot([x_position, x_position], [bottom_position, len(strings_frequencies) - 1],
                    linewidth=self.strings_ticks_line_width * tick_ratio * ratio,
                    c=tick_color, zorder=1, alpha=0.5 * ratio)
            cir = Circle((x_position, cir_bottom_position), radius=0.08 * ratio * cir_ratio,
                         linewidth=self.strings_ticks_line_width * tick_ratio * ratio,
                         zorder=2, facecolor='black', edgecolor=color_cir, alpha=0.6 * ratio)
            ax.add_patch(cir)
            # plot tick number
            if i % self.strings_n_temperament != 0:
                ax.text(x_position, cir_bottom_position / 2, s=self._integer_to_roman(i % 12),
                        horizontalalignment='center', verticalalignment='center', fontsize=6 * ratio,
                        zorder=2, c=self.strings_ticks_color, alpha=ratio)

    def _strings_plot_string_spectral(self, ax, strings_frequencies, log_fft_data, ss, v_fft_data):
        for string_i, reference_frequency in enumerate(strings_frequencies):
            # plot string
            ax.plot([-self.strings_length, 0], [string_i, string_i],
                    linewidth=self.strings_line_width_power_base ** string_i * self.strings_line_width,
                    c=self.strings_ticks_color, zorder=1, alpha=0.4)

            # plot string natural harmonics
            harmonics_info = self._strings_natural_harmonics_info(reference_frequency, string_i, log_fft_data,
                                                                  self.strings_max_fractal_natural_harmonics)
            for (log_fft_power, pitch, fft_pos,
                 x_position, y_position,
                 n, d, transform_ratio) in harmonics_info:
                rgb_color = self._hsb_to_rgb(pitch % 1, ss[fft_pos], 1)
                if n != 1:
                    size_ratio = self.strings_natural_harmonics_size_power_coefficient ** (d - 2)
                    diameter = log_fft_power * transform_ratio * size_ratio
                    line_style = 'solid'
                else:
                    diameter = log_fft_power * transform_ratio
                    line_style = 'dotted'
                cir = Arc((x_position, y_position), width=diameter, height=diameter,
                          linewidth=self.strings_line_width_harmonics, linestyle=line_style,
                          zorder=2, color=rgb_color, alpha=log_fft_power)
                ax.add_patch(cir)
                # show ratios
                ax.text(x_position, y_position + diameter, s=self.strings_roman_numbers[d - n],
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=14 * diameter, zorder=2, c=rgb_color, alpha=log_fft_power)
                ax.text(x_position, y_position - diameter, s=self.strings_roman_numbers[d],
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=14 * diameter, zorder=2, c=rgb_color, alpha=log_fft_power)

            # plot strings frequency
            (position_0, position_1,
             pitches, vs, i_s) = self._strings_position_transform(log_fft_data, v_fft_data,
                                                                  reference_frequency, string_i)
            for i in range(len(position_0[0])):
                if i != 0:
                    pos0 = [position_0[0][i], position_0[1][i]]
                    pos1 = [position_1[0][i], position_1[1][i]]
                    pos2 = [position_1[0][i - 1], position_1[1][i - 1]]
                    pos3 = [position_0[0][i - 1], position_0[1][i - 1]]
                    poly_position = np.array([pos0, pos1, pos2, pos3])
                    v_opacity = max(vs[i], vs[i - 1])
                    if v_opacity > self.figure_minimum_alpha:
                        rgb_color = self._hsb_to_rgb(pitches[i] % 1, ss[i_s[i]], 1)
                        ax.fill(poly_position[:, 0], poly_position[:, 1], facecolor=rgb_color,
                                edgecolor=rgb_color, linewidth=self.strings_frequency_line_width,
                                alpha=v_opacity, zorder=3)

    def _prepare_graph_strings(self, starting_time, save_path, dynamic_max_value=False):
        valid = self._check_analyze_duration(starting_time)
        if not valid:
            return False
        else:
            # prepare data
            (fft_data, log_fft_data, ss,
             v_fft_data) = self._analyze_two_channels_data_preparation(starting_time,
                                                                       dynamic_max_value=dynamic_max_value)

            # prepare position info
            strings_frequencies = []
            for s in self.strings_pitch_name_or_frequency:
                if self._is_float(s):
                    strings_frequencies.append(float(s))
                else:
                    strings_frequencies.append(self._translate_music_note_name_to_frequency(s))
            # making plots
            fig = plt.figure(figsize=self.strings_figure_size)
            ax = fig.add_subplot(111)

            # plot ticks
            self._strings_plot_ticks(ax, strings_frequencies)

            # loop strings
            self._strings_plot_string_spectral(ax, strings_frequencies, log_fft_data, ss, v_fft_data)

            ax.set_xlim(left=-self.strings_length / 2 ** (-1 / self.strings_n_temperament),
                        right=0.5)
            ax.set_ylim(bottom=-1.5, top=len(strings_frequencies) + 0.5)
            # set figure ratio
            self._post_processing_to_figure()

            # save figure
            fig.savefig(save_path, dpi=self.strings_dpi, bbox_inches='tight')
            self._matplotlib_clear_memory(fig)

            # prepare ifft play
            self._ifft_audio_export(fft_data, dynamic_max_value)
            return True

    def _prepare_video_string(self, starting_time, ending_time, save_path):
        """ save video for spiral """
        starting_time, ending_time, status = self._prepare_video_analyzer(starting_time, ending_time,
                                                                          save_analyzer_path=save_path,
                                                                          analyzer_function=self._prepare_graph_strings)
        return starting_time, ending_time, status
