import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from utils.piano_common import PianoCommon


class TonnetzAnalyzer(PianoCommon):
    """ ref: `tonnetz` """

    def __init__(self):
        super().__init__()
        # piano analyzer
        self.tonnetz_figure_size = (12, 12)
        self.tonnetz_dpi = 150

        # parameters
        self.tonnetz_scale = (8, 8)
        self.tonnetz_min_circle = 0.25
        self.tonnetz_text_minimum_alpha = 0.05
        self.tonnetz_key_color_transform_power = 3
        self.tonnetz_triangle_edge_max_alpha = 0.4
        self.tonnetz_triangle_face_max_alpha = 0.2

        # rain circle theme
        self.tonnetz_rain_circle_theme = False
        self.tonnetz_rain_circle_shrink = 0.8
        self.tonnetz_rain_circle_power = -2

        # calculate
        self.tonnetz_position_dict = None
        self.tonnetz_chroma_matrix = None
        self.tonnetz_chroma_position_matrix = None
        self.tonnetz_pitch_range = self.piano_key_length - 1

    def _tonnetz_data_prepare(self, starting_time, dynamic_max_value=False):
        # get starting sample index
        starting_sample = int(starting_time * self.sample_rate)
        # get data for spectral
        data = self.data
        audio_data = [x[starting_sample:starting_sample + self.analyze_n_window] for x in data]
        fft_data = [self._fft_data_transform_single(x)[0] for x in audio_data]
        log_fft_data = self._analyze_log_min_max_transform(fft_data, dynamic_max_value=dynamic_max_value)
        spectral_data = [self._piano_key_spectral_data(x, chroma=False) for x in log_fft_data]
        key_dicts = []
        for key_dict, raw_key, key_fft, in spectral_data:
            key_dicts.append(key_dict)
        return fft_data, key_dicts

    def _tonnetz_key_transition(self, key):
        """ transform `A0` as index `0` """
        return key - self.piano_key_range[0]

    def _tonnetz_circle_position(self):
        position_dict = {}
        chroma_matrix = []
        chroma_pos_matrix = []
        for column in range(self.tonnetz_scale[1]):
            chroma_row = []
            chroma_pos_row = []
            for row in range(self.tonnetz_scale[0]):
                if column % 2 == 0:
                    key = column // 2 + 7 * row
                    x = 2 * row
                else:
                    key = column // 2 + 7 * row + 4
                    x = 2 * row + 1
                y = 3 ** 0.5 * column
                chroma = key % 12
                chroma_row.append(chroma)
                chroma_pos_row.append([x, y])
                if chroma not in position_dict:
                    position_dict[chroma] = [[x, y]]
                else:
                    position_dict[chroma].append([x, y])
            chroma_matrix.append(chroma_row)
            chroma_pos_matrix.append(chroma_pos_row)
        return position_dict, chroma_matrix, chroma_pos_matrix

    def _tonnetz_circle_radius(self, key):
        k = self._tonnetz_key_transition(key)
        return self.tonnetz_min_circle + (1 - self.tonnetz_min_circle) * (1 - k / self.tonnetz_pitch_range)

    def _tonnetz_line_width(self, value, scale):
        return scale / max(self.tonnetz_scale) * value

    def _tonnetz_merge_key_dicts(self, key_dicts):
        if len(key_dicts) == 1 or len(key_dicts) > 2:
            merged_key_list = []
            for key, v in key_dicts[0].items():
                merged_key_list.append([key, v])
            amplitude_ratio = {}
        else:
            rebuild_dict = {}
            merged_key_list = []
            for key_dict in key_dicts:
                for key, v in key_dict.items():
                    if key not in rebuild_dict:
                        rebuild_dict[key] = [v]
                    else:
                        rebuild_dict[key].append(v)
            amplitude_ratio = {}
            for key, v in rebuild_dict.items():
                merged_key_list.append([key, np.mean(v)])
                amplitude_ratio[key] = self._amplitude_ratio(*v)
        max_chroma_value_dict = {}
        for key, v in merged_key_list:
            chroma = key % 12
            if chroma not in max_chroma_value_dict:
                max_chroma_value_dict[chroma] = v
            else:
                max_chroma_value_dict[chroma] = max(max_chroma_value_dict[chroma], v)
        # sort by value for plot (light notes on the top, dark notes plot first)
        merged_key_list.sort(key=lambda x: x[1])
        return merged_key_list, amplitude_ratio, max_chroma_value_dict

    def _tonnetz_build_triangle_edge(self, ax, i, j, max_chroma_value_dict, base_fft_value, base_node_pos,
                                     triangle_color):
        node_chroma = self.tonnetz_chroma_matrix[i][j]
        node_pos = self.tonnetz_chroma_position_matrix[i][j]
        fft_value = max_chroma_value_dict[node_chroma]
        fft_value = self._merge_array_values([base_fft_value, fft_value])
        alpha = fft_value ** self.tonnetz_key_color_transform_power * self.tonnetz_triangle_edge_max_alpha
        if alpha > self.figure_minimum_alpha:
            ax.plot([base_node_pos[0], node_pos[0]], [base_node_pos[1], node_pos[1]], c=triangle_color,
                    linewidth=self._tonnetz_line_width(fft_value, 24), alpha=alpha, zorder=1)

    def _tonnetz_build_triangle_face(self, ax, i, j, base_chroma, max_chroma_value_dict, base_fft_value, base_node_pos,
                                     fft_1, pos_1, triangle_color, font_size, capital=True):
        pos_2 = self.tonnetz_chroma_position_matrix[i][j]
        fft_2 = max_chroma_value_dict[self.tonnetz_chroma_matrix[i][j]]
        merged_fft = self._merge_array_values([base_fft_value, fft_1, fft_2])
        alpha = merged_fft ** self.tonnetz_key_color_transform_power
        if alpha > self.figure_minimum_alpha:
            # plot triangle
            ax.fill([base_node_pos[0], pos_1[0], pos_2[0]], [base_node_pos[1], pos_1[1], pos_2[1]],
                    facecolor=triangle_color, alpha=alpha * self.tonnetz_triangle_face_max_alpha, zorder=1)
            # plot text
            center_pos = np.mean([base_node_pos, pos_1, pos_2], axis=0)
            chord_name = self.note_name_lib[base_chroma]
            if not capital:
                chord_name = chord_name.lower()
            ax.text(center_pos[0], center_pos[1], chord_name, c=triangle_color, horizontalalignment='center',
                    verticalalignment='center', fontdict={'style': 'italic'}, fontsize=font_size, alpha=alpha, zorder=2)

    def _tonnetz_plot_node_text(self, ax, max_chroma_value_dict):
        font_size = 128 / max(self.tonnetz_scale)
        for chroma, coordinates in self.tonnetz_position_dict.items():
            fft_value = max_chroma_value_dict[chroma]
            background_color = self._hsb_to_rgb(0, 0, self.tonnetz_text_minimum_alpha)
            if fft_value > self.tonnetz_text_minimum_alpha:
                color_saturation = (fft_value - self.tonnetz_text_minimum_alpha) / (
                        1 - self.tonnetz_text_minimum_alpha)
                if self.colorful_theme:
                    color = self._hsb_to_rgb(((chroma - 3) / 12) % 1, color_saturation, 1)
                else:
                    color = self.mono_theme_color
                alpha = color_saturation ** self.tonnetz_key_color_transform_power
            else:
                color = 'k'
                alpha = 0

            for x, y in coordinates:
                ax.text(x, y, self.note_name_lib[chroma], c=background_color, horizontalalignment='center',
                        verticalalignment='center', fontsize=font_size, zorder=1)
                if alpha != 0:
                    ax.text(x, y, self.note_name_lib[chroma], c=color, horizontalalignment='center',
                            verticalalignment='center', fontsize=font_size, alpha=alpha, zorder=4)
                    # add text background
                    cir_end = Circle((x, y), radius=self.tonnetz_min_circle, zorder=2, alpha=0.5 * alpha ** 0.5,
                                     facecolor='k')
                    ax.add_patch(cir_end)

    def _tonnetz_plot_triangle(self, ax, max_chroma_value_dict):
        font_size = 64 / max(self.tonnetz_scale)
        if self.colorful_theme:
            triangle_color = 'gray'
        else:
            triangle_color = self.mono_theme_color
        for i in range(len(self.tonnetz_chroma_matrix)):
            for j in range(len(self.tonnetz_chroma_matrix[0])):
                base_node_chroma = self.tonnetz_chroma_matrix[i][j]
                base_node_pos = self.tonnetz_chroma_position_matrix[i][j]
                base_fft_value = max_chroma_value_dict[base_node_chroma]
                # plot node connection
                if i != len(self.tonnetz_chroma_matrix) - 1:
                    self._tonnetz_build_triangle_edge(ax, i + 1, j, max_chroma_value_dict, base_fft_value,
                                                      base_node_pos, triangle_color)
                if j != len(self.tonnetz_chroma_matrix[0]) - 1:
                    self._tonnetz_build_triangle_edge(ax, i, j + 1, max_chroma_value_dict, base_fft_value,
                                                      base_node_pos, triangle_color)
                if i % 2 != 0 and i != len(self.tonnetz_chroma_matrix) - 1 and j != len(
                        self.tonnetz_chroma_matrix[0]) - 1:
                    self._tonnetz_build_triangle_edge(ax, i + 1, j + 1, max_chroma_value_dict, base_fft_value,
                                                      base_node_pos, triangle_color)
                if i % 2 == 0 and i != len(self.tonnetz_chroma_matrix) - 1 and j != 0:
                    self._tonnetz_build_triangle_edge(ax, i + 1, j - 1, max_chroma_value_dict, base_fft_value,
                                                      base_node_pos, triangle_color)
                # plot triangle (& chord text)
                if j != len(self.tonnetz_chroma_matrix[0]) - 1:
                    pos_1 = self.tonnetz_chroma_position_matrix[i][j + 1]
                    fft_1 = max_chroma_value_dict[self.tonnetz_chroma_matrix[i][j + 1]]
                    if i % 2 == 0:
                        if i != len(self.tonnetz_chroma_matrix) - 1:
                            self._tonnetz_build_triangle_face(ax, i + 1, j, base_node_chroma, max_chroma_value_dict,
                                                              base_fft_value, base_node_pos, fft_1, pos_1,
                                                              triangle_color, font_size, capital=True)
                        if i != 0:
                            self._tonnetz_build_triangle_face(ax, i - 1, j, base_node_chroma, max_chroma_value_dict,
                                                              base_fft_value, base_node_pos, fft_1, pos_1,
                                                              triangle_color, font_size, capital=False)
                    if i % 2 == 1:
                        if i != len(self.tonnetz_chroma_matrix) - 1:
                            self._tonnetz_build_triangle_face(ax, i + 1, j + 1, base_node_chroma, max_chroma_value_dict,
                                                              base_fft_value, base_node_pos, fft_1, pos_1,
                                                              triangle_color, font_size, capital=True)
                        if i != 0:
                            self._tonnetz_build_triangle_face(ax, i - 1, j + 1, base_node_chroma, max_chroma_value_dict,
                                                              base_fft_value, base_node_pos, fft_1, pos_1,
                                                              triangle_color, font_size, capital=False)

    def _tonnetz_plot_circle(self, ax, merged_key_list, amplitude_ratio_dict):
        for key, fft_value in merged_key_list:
            chroma = key % 12
            alpha = fft_value ** self.tonnetz_key_color_transform_power
            if alpha < self.figure_minimum_alpha:
                continue
            if key in amplitude_ratio_dict:
                amplitude_ratio = amplitude_ratio_dict[key]
            else:
                amplitude_ratio = 1
            if self.tonnetz_rain_circle_theme:
                radius = self.tonnetz_rain_circle_shrink * self._tonnetz_circle_radius(chroma) + np.log(
                    fft_value ** self.tonnetz_rain_circle_power)
            else:
                radius = self._tonnetz_circle_radius(key)
            line_width = self._tonnetz_line_width(fft_value, 40)
            for x, y in self.tonnetz_position_dict[chroma]:
                if self.colorful_theme:
                    color = self._hsb_to_rgb(((chroma - 3) / 12) % 1,
                                             amplitude_ratio,
                                             1)
                else:
                    color = self.mono_theme_color
                cir_end = Circle((x, y), radius=radius, zorder=2, fill=False, alpha=alpha,
                                 linewidth=line_width, edgecolor=color)
                ax.add_patch(cir_end)

    def _prepare_graph_tonnetz(self, starting_time, save_path, dynamic_max_value=False):
        valid = self._check_analyze_duration(starting_time)
        if not valid:
            return False
        else:
            # prepare data
            fft_data, key_dicts = self._tonnetz_data_prepare(starting_time, dynamic_max_value=dynamic_max_value)
            merged_key_list, amplitude_ratio_dict, max_chroma_value_dict = self._tonnetz_merge_key_dicts(key_dicts)

            # prepare tonnetz position
            if self.tonnetz_position_dict is None:
                (self.tonnetz_position_dict, self.tonnetz_chroma_matrix,
                 self.tonnetz_chroma_position_matrix) = self._tonnetz_circle_position()

            # make plot
            fig = plt.figure(figsize=self.tonnetz_figure_size)
            ax = fig.add_subplot(111)
            # plot text
            self._tonnetz_plot_node_text(ax, max_chroma_value_dict)
            # plot triangle
            self._tonnetz_plot_triangle(ax, max_chroma_value_dict)
            # plot circle
            self._tonnetz_plot_circle(ax, merged_key_list, amplitude_ratio_dict)

            ax.set_xlim(left=0, right=2 * (self.tonnetz_scale[0] - 1) + 1)
            ax.set_ylim(bottom=0, top=3 ** 0.5 * (self.tonnetz_scale[1] - 1))

            # set plot ratio
            self._post_processing_to_figure()

            # save figure
            fig.savefig(save_path, dpi=self.tonnetz_dpi, bbox_inches='tight')
            self._matplotlib_clear_memory(fig)

            # prepare ifft play
            self._ifft_audio_export(fft_data, dynamic_max_value)
            return True

    def _prepare_video_tonnetz(self, starting_time, ending_time, save_path):
        """ save video for tonnetz """
        starting_time, ending_time, status = self._prepare_video_analyzer(starting_time, ending_time,
                                                                          save_analyzer_path=save_path,
                                                                          analyzer_function=self._prepare_graph_tonnetz)
        return starting_time, ending_time, status
