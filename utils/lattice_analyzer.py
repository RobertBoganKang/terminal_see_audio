import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from utils.piano_common import PianoCommon


class LatticeAnalyzer(PianoCommon):
    def __init__(self):
        super().__init__()
        # piano analyzer
        self.lattice_figure_size = (12, 12)
        self.lattice_dpi = 200

        # parameters
        self.lattice_scale = (8, 8)
        self.lattice_min_circle = 0.25
        self.lattice_text_minimum_alpha = 0.15
        self.lattice_text_color_power = 0.3
        self.lattice_position_dict = None

        # calculate
        self.lattice_pitch_range = self.piano_key_length - 1

    def _lattice_data_prepare(self, starting_time, dynamic_max_value=False):
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

    def _lattice_key_transition(self, key):
        """ transform `A0` as index `0` """
        return key - self.piano_key_range[0]

    def _lattice_circle_position(self):
        lattice_position_dict = {}
        for column in range(self.lattice_scale[1]):
            for row in range(self.lattice_scale[0]):
                if column % 2 == 0:
                    key = column // 2 + 7 * row
                    x = 2 * row
                else:
                    key = column // 2 + 7 * row + 4
                    x = 2 * row + 1
                y = 3 ** 0.5 * column
                chroma = key % 12
                if chroma not in lattice_position_dict:
                    lattice_position_dict[chroma] = [[x, y]]
                else:
                    lattice_position_dict[chroma].append([x, y])
        return lattice_position_dict

    def _lattice_circle_radius(self, key):
        k = self._lattice_key_transition(key)
        return self.lattice_min_circle + (1 - self.lattice_min_circle) * (1 - k / self.lattice_pitch_range)

    def _lattice_merge_key_dicts(self, key_dicts):
        if len(key_dicts) == 1 or len(key_dicts) > 2:
            merged_key_list = key_dicts[0]
            amplitude_ratio = {}
        else:
            rebuild_dict = {}
            merged_key_list = []
            for key_dict in key_dicts:
                for k, v in key_dict.items():
                    if k not in rebuild_dict:
                        rebuild_dict[k] = [v]
                    else:
                        rebuild_dict[k].append(v)
            amplitude_ratio = {}
            for k, v in rebuild_dict.items():
                merged_key_list.append([k, np.mean(v)])
                amplitude_ratio[k] = self._amplitude_ratio(*v)
        max_chroma_value_dict = {}
        for k, v in merged_key_list:
            key = k % 12
            if key not in max_chroma_value_dict:
                max_chroma_value_dict[key] = v
            else:
                max_chroma_value_dict[key] = max(max_chroma_value_dict[key], v)
        # sort by value for plot (light notes on the top, dark notes plot first)
        merged_key_list.sort(key=lambda x: x[1])
        return merged_key_list, amplitude_ratio, max_chroma_value_dict

    def _prepare_graph_lattice(self, starting_time, save_path, dynamic_max_value=False):
        valid = self._check_analyze_duration(starting_time)
        if not valid:
            return False
        else:
            # prepare data
            fft_data, key_dicts = self._lattice_data_prepare(starting_time, dynamic_max_value=dynamic_max_value)
            merged_key_list, amplitude_ratio_dict, max_chroma_value_dict = self._lattice_merge_key_dicts(key_dicts)
            # prepare lattice position
            if self.lattice_position_dict is None:
                self.lattice_position_dict = self._lattice_circle_position()
            # make plot
            fig = plt.figure(figsize=self.lattice_figure_size)
            ax = fig.add_subplot(111)
            # plot
            # plot text
            for k, coordinates in self.lattice_position_dict.items():
                fft_value = max_chroma_value_dict[k]
                if fft_value > self.lattice_text_minimum_alpha:
                    color_saturation = (fft_value ** self.lattice_text_color_power -
                                        self.lattice_text_minimum_alpha) / (1 - self.lattice_text_minimum_alpha)
                    color = self._hsb_to_rgb(((k - 3) / 12) % 1, color_saturation, fft_value)
                else:
                    color = self._hsb_to_rgb(0, 0, self.lattice_text_minimum_alpha)
                for x, y in coordinates:
                    ax.text(x, y, self.note_name_lib[k], c=color, horizontalalignment='center',
                            verticalalignment='center', fontsize=1 / self.lattice_scale[0] * 128,
                            zorder=2)
            # plot circle
            for key, fft_value in merged_key_list:
                k = key % 12
                alpha = fft_value ** self.piano_key_color_transform_power
                if alpha < self.figure_minimum_alpha:
                    continue
                if key in amplitude_ratio_dict:
                    amplitude_ratio = amplitude_ratio_dict[key]
                else:
                    amplitude_ratio = 1
                if fft_value > 0.1:
                    rad = self._lattice_circle_radius(key)
                    for x, y in self.lattice_position_dict[k]:
                        color = self._hsb_to_rgb(((k - 3) / 12) % 1,
                                                 amplitude_ratio,
                                                 alpha)
                        cir_end = Circle((x, y), radius=rad, zorder=1, fill=False,
                                         linewidth=1 / self.lattice_scale[0] * 40, edgecolor=color)
                        ax.add_patch(cir_end)

            ax.set_xlim(left=0, right=2 * (self.lattice_scale[0] - 1) + 1)
            ax.set_ylim(bottom=0, top=3 ** 0.5 * (self.lattice_scale[1] - 1))
            # set plot ratio
            self._set_1to1_ratio_figure()

            # save figure
            fig.savefig(save_path, dpi=self.lattice_dpi, bbox_inches='tight')
            self._matplotlib_clear_memory(fig)

            # prepare ifft play
            if not dynamic_max_value:
                self._ifft_audio_export(fft_data)
            return True

    def _prepare_video_lattice(self, starting_time, ending_time, save_path):
        """ save video for lattice """
        starting_time, ending_time, status = self._prepare_video_analyzer(starting_time, ending_time,
                                                                          save_analyzer_path=save_path,
                                                                          analyzer_function=self._prepare_graph_lattice)
        return starting_time, ending_time, status
