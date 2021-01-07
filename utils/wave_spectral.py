import colorsys

import matplotlib.pyplot as plt
import numpy as np

from utils.analyze_common import AnalyzeCommon


class WaveSpectral(AnalyzeCommon):
    def __init__(self):
        super().__init__()
        # spectral mode
        self.ws_spectral_transform_y = 'fbank'
        self.ws_spectral_transform_v = 'log'
        self.ws_spectral_mode = 'spectral'

        # colors & themes
        self.ws_plot_axis_color = 'white'
        self.ws_plot_spectral_color = 'inferno'
        self.ws_plot_wave_color = 'mediumspringgreen'
        self.ws_plot_entropy_color = 'red'

        # wave/spectral
        # line width parameters with `thin`, `thick`, `mode_switch_time`
        self.ws_wave_line_width_params = [.2, 1.2, 3]
        self.ws_entropy_line_width = 0.8
        self.ws_graphics_ratio = 5
        self.ws_figure_size = (12, 4)
        self.ws_figure_dpi = 300

    def _ws_get_line_width(self, duration):
        if duration > self.ws_wave_line_width_params[2]:
            line_width = self.ws_wave_line_width_params[0]
        else:
            line_width = (self.ws_wave_line_width_params[1] - (
                    self.ws_wave_line_width_params[1] - self.ws_wave_line_width_params[0]) /
                          self.ws_wave_line_width_params[-1] * duration)
        return line_width

    def _ws_mel_filter(self, spectral_raw):
        """
        convert spectral to mel-spectral
            mel = 2595 * log10(1 + f/700)
            f = 700 * (10^(m/2595) - 1
        --> from [https://zhuanlan.zhihu.com/p/130926693]
        :param spectral_raw: spectral
        :return: mel spectral
        """
        fs = self.sample_rate
        n_filter = self.n_window
        n_fft = self.n_window
        # lowest hearing frequency
        mel_min = self.min_hearing_frequency
        mel_max = 2595 * np.log10(1 + fs / 2.0 / 700)
        mel_points = np.linspace(mel_min, mel_max, n_filter + 2)
        hz_points = 700 * (10 ** (mel_points / 2595.0) - 1)
        filter_edge = np.floor(hz_points * (n_fft + 1) / fs)

        f_bank = np.zeros((n_filter, int(n_fft / 2 + 1)))
        for m in range(1, 1 + n_filter):
            f_left = int(round(filter_edge[m - 1]))
            f_center = int(round(filter_edge[m]))
            f_right = int(round(filter_edge[m + 1]))
            # fix broken image
            if f_left == f_center:
                f_left -= 1
            if f_right == f_center:
                f_right += 1

            for k in range(f_left, f_center):
                f_bank[m - 1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                f_bank[m - 1, k] = (f_right - k) / (f_right - f_center)

        filter_banks = np.dot(spectral_raw, f_bank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        return filter_banks

    def _ws_spectral_transform(self, spectral, clip_power=None, min_max_norm=True):
        if clip_power is None:
            clip_power = self.min_hearing_power
        if self.ws_spectral_transform_v == 'power':
            spectral = np.power(spectral, self.spectral_power_transform_coefficient)
        elif self.ws_spectral_transform_v == 'log':
            spectral = np.clip(spectral, clip_power, None)
            spectral = np.log(spectral)
        else:
            raise ValueError(f'spectral transform `Values` [{self.ws_spectral_transform_v}] unrecognized')
        if min_max_norm:
            spectral -= np.min(spectral)
            spectral /= np.max(spectral)
        return spectral

    def _ws_data_prepare(self, starting_time, ending_time):
        """ prepare partition of audios """
        starting_time, ending_time = self._fix_input_starting_ending_time(starting_time, ending_time)
        if not self._check_audio_duration_valid(starting_time, ending_time, self.n_window):
            return None, None, starting_time, ending_time, False
        data_, time_ = self._export_audio(starting_time, ending_time, audio_part_path=self.audio_part_path)
        return data_, time_, starting_time, ending_time, True

    def _ws_plot_wave(self, data_one, time_, grid, plot_position):
        """ plot wave """
        # plot audio wave
        fig_wave = plt.subplot(grid[plot_position, 0])
        # create a function to define line width
        duration = time_[-1] - time_[0]
        line_width = self._ws_get_line_width(duration)
        # plot norm
        if np.max(np.abs(data_one)):
            data_one_norm = data_one / np.max(np.abs(data_one))
            fig_wave.plot(time_, data_one_norm, linewidth=line_width, color=self.ws_plot_wave_color, alpha=0.3,
                          zorder=1)
        # plot wave
        fig_wave.plot(time_, data_one, linewidth=line_width, color=self.ws_plot_wave_color, zorder=2)
        fig_wave.set_xlim(left=time_[0], right=time_[-1])
        fig_wave.axes.get_yaxis().set_ticks([])
        fig_wave.axes.set_ylim([-1, 1])
        fig_wave.spines['left'].set_visible(False)
        fig_wave.spines['right'].set_visible(False)
        fig_wave.spines['top'].set_visible(False)
        if plot_position == self.channel_num - 1:
            fig_wave.spines['bottom'].set_color(self.ws_plot_axis_color)
            fig_wave.tick_params(axis='x', colors=self.ws_plot_axis_color)
        else:
            fig_wave.axes.get_xaxis().set_ticks([])
            fig_wave.spines['bottom'].set_visible(False)

    def _ws_plot_spectral(self, spectral, grid, plot_position):
        """ plot spectral """
        # plot spectral
        if self.ws_spectral_transform_y == 'fbank':
            spectral = self._ws_mel_filter(spectral)
        elif self.ws_spectral_transform_y == 'fft':
            pass
        else:
            raise ValueError(f'spectral transform `Y` [{self.ws_spectral_transform_y}] unrecognized')
        # transform to show
        spectral = self._ws_spectral_transform(spectral)
        spectral = np.flip(spectral, axis=1)
        spectral = np.transpose(spectral)

        # plot
        fig_spectral = plt.subplot(
            grid[self.channel_num + (self.ws_graphics_ratio - 1) * plot_position:
                 self.channel_num + (self.ws_graphics_ratio - 1) * (plot_position + 1), 0])
        fig_spectral.imshow(spectral, aspect='auto', cmap=self.ws_plot_spectral_color)
        fig_spectral.axis('off')

    def _ws_plot_entropy(self, spectral, time_, grid, plot_position):
        """ plot entropy """
        entropy = [self._get_shannon_entropy(x) for x in spectral]
        # create a function to define line width
        duration = time_[-1] - time_[0]
        time_rebuild = [time_[0] + i / len(spectral) * duration for i in range(len(spectral))]
        # plot
        fig_entropy = plt.subplot(
            grid[self.channel_num + (self.ws_graphics_ratio - 1) * plot_position:
                 self.channel_num + (self.ws_graphics_ratio - 1) * (plot_position + 1), 0])
        fig_entropy.set_xlim(left=time_[0], right=time_[-1])
        fig_entropy.fill_between(time_rebuild, y1=entropy, y2=0, facecolor=self.ws_plot_entropy_color, alpha=0.6,
                                 edgecolor=self.ws_plot_entropy_color, linewidth=self.ws_entropy_line_width)
        fig_entropy.spines['left'].set_visible(True)
        fig_entropy.spines['right'].set_visible(True)
        fig_entropy.spines['top'].set_visible(False)
        if plot_position == self.channel_num - 1:
            fig_entropy.spines['bottom'].set_visible(True)
        else:
            fig_entropy.spines['bottom'].set_visible(False)
        fig_entropy.axes.get_xaxis().set_ticks([])

    def _ws_plot_pitch(self, spectral, grid, plot_position):
        """ plot pitch """
        # plot pitch
        pitch_data = np.array(
            [[(self._frequency_to_pitch(self._fft_position_to_frequency(i)) * 12 - 3) % 12 / 12 for i in
              range(len(spectral[0]))] for _ in range(len(spectral))])
        if self.ws_spectral_transform_y == 'fbank':
            spectral = self._ws_mel_filter(spectral)
            pitch_data = self._ws_mel_filter(pitch_data)
        elif self.ws_spectral_transform_y == 'fft':
            pass
        else:
            raise ValueError(f'spectral transform `Y` [{self.ws_spectral_transform_y}] unrecognized')

        # transform to show
        spectral = self._ws_spectral_transform(spectral, min_max_norm=True)

        image_reconstruction = []
        for i in range(len(spectral)):
            one_row = []
            for j in range(len(spectral[0])):
                h = pitch_data[i][j]
                s = spectral[i][j]
                v = spectral[i][j] ** 2
                rgb = colorsys.hsv_to_rgb(h, s, v)
                one_row.append(rgb)
            image_reconstruction.append(one_row)
        # transform to show
        image_reconstruction = np.flip(image_reconstruction, axis=1)
        image_reconstruction = np.transpose(image_reconstruction, axes=(1, 0, 2))

        # plot
        fig_pitch = plt.subplot(
            grid[self.channel_num + (self.ws_graphics_ratio - 1) * plot_position:
                 self.channel_num + (self.ws_graphics_ratio - 1) * (plot_position + 1), 0])
        fig_pitch.imshow(image_reconstruction, aspect='auto')
        fig_pitch.axis('off')

    def _prepare_graph_wave_spectral(self, starting_time, ending_time):
        """ prepare graphics and audio files """
        data_, time_, starting_time, ending_time, valid = self._ws_data_prepare(starting_time, ending_time)
        self._initialize_spectral(starting_time, ending_time)
        if not valid:
            return starting_time, ending_time, False

        # plot image
        grid = plt.GridSpec(self.ws_graphics_ratio * self.channel_num, 1, wspace=0, hspace=0)
        fig = plt.figure(figsize=self.ws_figure_size)

        # plot spectral
        for i in range(len(data_)):
            spectral = self._calc_sp(data_[i], self.n_window, self.n_overlap)
            if self.ws_spectral_mode == 'spectral':
                if self.colorful_theme:
                    self._ws_plot_pitch(spectral, grid, i)
                else:
                    self._ws_plot_spectral(spectral, grid, i)
            elif self.ws_spectral_mode == 'entropy':
                self._ws_plot_entropy(spectral, time_, grid, i)
            # plot wave
            self._ws_plot_wave(data_[i], time_, grid, i)

        # save figure
        fig.savefig(self.wave_spectral_graphics_path, dpi=self.ws_figure_dpi, bbox_inches='tight')
        self._matplotlib_clear_memory(fig)
        return starting_time, ending_time, True

    def _ws_initial_or_restore_running(self, starting_time=None, ending_time=None, plot=True):
        """ first run & restore run """
        self._initialization()
        if starting_time is None:
            starting_time = 0
        if ending_time is None:
            ending_time = self._get_audio_time()
        self._initialize_spectral(starting_time, ending_time)
        self._prepare_graph_wave_spectral(starting_time, ending_time)
        if plot:
            self._terminal_plot(self.wave_spectral_graphics_path)
