import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from utils.flower_common import FlowerCommon


class PhaseAnalyzer(FlowerCommon):
    def __init__(self):
        super().__init__()

    def _phase_position_transform(self, arrays, phases):
        array_0 = arrays[0]
        array_1 = arrays[1]
        phase_0 = phases[0]
        phase_1 = phases[1]
        x_array = []
        y_array = []
        x_peak = []
        y_peak = []
        e_array = []
        pitches = []
        ratio_array = []
        for i in range(len(array_0)):
            t0 = array_0[i]
            t1 = array_1[i]
            p0 = phase_0[i]
            p1 = phase_1[i]
            # rotate counter-clockwise for `pi/2`
            angle = p1 - p0 + np.pi / 2
            # skip low frequency part
            if i > 0:
                frequency = self._fft_position_to_frequency(i)
                pitch = self._frequency_to_pitch(frequency)
                energy = (t0 + t1) / 2
                if pitch > 0 and energy > self.flower_min_analyze_power:
                    # pitch to `C` as ticks
                    pitch = (pitch * 12 - 3) / 12
                    x_position = pitch * np.cos(angle)
                    y_position = pitch * np.sin(angle)
                    x_peak_position = (pitch + np.power(energy,
                                                        self.flower_stem_power_coefficient)) * np.cos(angle)
                    y_peak_position = (pitch + np.power(energy,
                                                        self.flower_stem_power_coefficient)) * np.sin(angle)
                    x_array.append(x_position)
                    y_array.append(y_position)
                    x_peak.append(x_peak_position)
                    y_peak.append(y_peak_position)
                    e_array.append(energy)
                    pitches.append(pitch)
                    ratio_array.append(self._amplitude_ratio(t0, t1))
        return x_array, y_array, x_peak, y_peak, e_array, pitches, ratio_array

    def _prepare_graph_phase(self, starting_time, save_path, dynamic_max_value=False):
        valid = self._check_analyze_duration(starting_time)
        if not valid:
            return False
        else:
            fft_data, phase_data = self._analyze_get_audio_fft_data(starting_time, phase=True)
            log_fft_data = self._analyze_log_min_max_transform(fft_data, dynamic_max_value=dynamic_max_value)
            (x_positions, y_positions,
             x_peaks, y_peaks,
             energies, pitches, ratio_array) = self._phase_position_transform(log_fft_data, phase_data)
            # making plots
            fig = plt.figure(figsize=self.flower_figure_size)
            ax = fig.add_subplot(111)
            # plot baseline
            max_baseline_circle = int(self._frequency_to_pitch(self.sample_rate / 2)) - 1
            for i in range(1, max_baseline_circle + 1)[::-1]:
                alpha = self.flower_baseline_transform_alpha ** (-i)
                cir_end = Circle((0, 0), radius=i, zorder=1, facecolor='black',
                                 linewidth=self.flower_baseline_width, edgecolor=self.flower_baseline_color,
                                 alpha=alpha)
                ax.add_patch(cir_end)
            # plot grass
            for i in range(len(x_positions)):
                pitch = pitches[i]
                energy = energies[i]
                color = self._hsb_to_rgb(pitch % 1,
                                         ratio_array[i],
                                         1)
                if energy > self.figure_minimum_alpha:
                    ax.plot([x_positions[i], x_peaks[i]],
                            [y_positions[i], y_peaks[i]], c=color, linewidth=self.flower_line_width, alpha=energy,
                            zorder=2)
                    cir_end = Circle((x_peaks[i], y_peaks[i]), radius=energy / 5, zorder=3, facecolor=color,
                                     linewidth=self.flower_line_width, edgecolor=color, alpha=energy)
                    ax.add_patch(cir_end)
                if i != 0 and self._flower_get_angle(x_positions[i],
                                                     x_positions[i - 1],
                                                     y_positions[i],
                                                     y_positions[i - 1]) \
                        < self.flower_min_angle_connection / 180 * np.pi:
                    mean_energy = (energies[i] + energies[i - 1]) / 2
                    if mean_energy > self.figure_minimum_alpha:
                        ax.plot([x_positions[i], x_positions[i - 1]],
                                [y_positions[i], y_positions[i - 1]], c=color,
                                linewidth=self.flower_ground_line_width,
                                alpha=mean_energy,
                                zorder=2)

            # set figure ratio
            ax.set_ylim(bottom=-max_baseline_circle - 1, top=max_baseline_circle + 1)
            ax.set_xlim(left=-max_baseline_circle - 1, right=max_baseline_circle + 1)
            self._post_processing_to_figure()

            # save figure
            fig.savefig(save_path, dpi=self.flower_dpi, bbox_inches='tight')
            self._matplotlib_clear_memory(fig)

            # prepare ifft play
            self._ifft_audio_export(fft_data, dynamic_max_value)
            return True

    def _prepare_video_phase(self, starting_time, ending_time, save_path):
        """ save video for spiral """
        (starting_time,
         ending_time,
         status) = self._prepare_video_analyzer(starting_time, ending_time,
                                                save_analyzer_path=save_path,
                                                analyzer_function=self._prepare_graph_phase)
        return starting_time, ending_time, status
