import matplotlib.pyplot as plt
import numpy as np

from utils.analyze_common import AnalyzeCommon


class TuningAnalyzer(AnalyzeCommon):
    def __init__(self):
        super().__init__()
        self.tuning_figure_size = (12, 4)
        self.tuning_line_width = 0.7
        self.tuning_target_line_width = 1.2
        self.tuning_dpi = 300
        # define max layer number
        self.tuning_max_layer = 48

        # color & theme
        self.tuning_base_color = '#444'
        self.tuning_line_color = 'red'

    @staticmethod
    def _tuning_get_layer_position(frequency, tuning):
        prepare = frequency / tuning + 0.5
        layer = int(prepare)
        position = prepare - layer - 0.5
        return layer, position

    def _tuning_get_positions(self, arrays, tuning, s, v):
        array_0 = arrays[0]
        array_1 = arrays[1]
        position_info = []
        keys = []
        vs = []
        ss = []
        for i in range(len(array_0)):
            if i > 0:
                frequency = self._fft_position_to_frequency(i)
                key = self._frequency_to_key(frequency)
                layer, position = self._tuning_get_layer_position(frequency, tuning)
                if layer > self.tuning_max_layer:
                    break
                position_x_0 = layer - array_0[i] / 2
                position_x_1 = layer + array_1[i] / 2
                position_y = position
                position_info.append([position_x_0, position_x_1, position_y, layer])
                vs.append(v[i])
                keys.append(key)
                ss.append(s[i])
        return position_info, keys, ss, vs

    def _tuning_plot(self, ax, position_info, pitches, ss, vs):
        position_stack = []
        position_sub_stack = []
        for i in range(len(position_info) - 1):
            position_x_0, position_x_1, position_y, layer = position_info[i]
            position_x_0_a, position_x_1_a, position_y_a, layer_a = position_info[i + 1]
            if layer == layer_a:
                x_positions = [position_x_0, position_x_1, position_x_1_a, position_x_0_a]
                y_positions = [position_y, position_y, position_y_a, position_y_a]
                v_opacity = max(vs[i], vs[i + 1])
                k = pitches[i]
                position_sub_stack.append([position_x_0, position_y])
                # plot frequencies
                if v_opacity > self.figure_minimum_alpha:
                    if self.colorful_theme:
                        color = self._hsb_to_rgb((k - 3) % 12 / 12, ss[i], 1)
                    else:
                        color = self.mono_theme_color
                    ax.fill(x_positions, y_positions, edgecolor=color, facecolor=color,
                            linewidth=self.tuning_line_width, zorder=2, alpha=v_opacity)
            else:
                position_stack.append(position_sub_stack)
                position_sub_stack = []
        # plot base line
        max_position_x = 0
        for positions in position_stack:
            positions = np.array(positions)
            max_position_x = positions[-1, 0]
            ax.plot(positions[:, 0], positions[:, 1], c=self.tuning_base_color, alpha=0.4,
                    linewidth=self.tuning_line_width, zorder=1)
        # plot tuning line
        ax.plot([0, max_position_x], [0, 0], c=self.tuning_line_color, alpha=0.3,
                linewidth=self.tuning_target_line_width, zorder=3)
        return max_position_x

    def _prepare_graph_tuning(self, starting_time, tuning_string, save_path):
        valid = self._check_analyze_duration(starting_time)
        if not valid:
            return False
        else:
            # translate tuning
            tuning = self._translate_string_to_frequency(tuning_string)
            if tuning is None:
                return False
            if tuning < self.min_hearing_frequency:
                print(f'<!> tuning frequency too low (<{self.min_hearing_frequency})')
            # prepare data
            (fft_data, log_fft_data,
             s_fft_magnitude_ratio_data, v_fft_data) = self._analyze_two_channels_data_preparation(starting_time)
            # get position info
            position_info, keys, ss, vs = self._tuning_get_positions(log_fft_data, tuning,
                                                                     s_fft_magnitude_ratio_data, v_fft_data)
            # plot
            fig = plt.figure(figsize=self.tuning_figure_size)
            ax = fig.add_subplot(111)
            ax.set_ylim(bottom=-0.5, top=0.5)
            max_x_position = self._tuning_plot(ax, position_info, keys, ss, vs)
            ax.set_xlim(left=-0.5, right=max_x_position + 0.5)

            plt.axis('off')
            fig.savefig(save_path, dpi=self.tuning_dpi, bbox_inches='tight')

            self._matplotlib_clear_memory(fig)

            # prepare ifft play
            self._ifft_audio_export(self._analyze_log_min_max_transform(fft_data, log=False))
            return True
