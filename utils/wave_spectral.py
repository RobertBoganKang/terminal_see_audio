import colorsys

import matplotlib.pyplot as plt
import numpy as np

from utils.common import Common


class WaveSpectral(Common):
    def __init__(self):
        super().__init__()
        # spectral mode
        self.spectral_transform_y = 'fbank'
        self.spectral_transform_v = 'log'
        self.spectral_phase_mode = 'spectral'

        # colors & themes
        self.plot_axis_color = 'white'
        self.plot_spectral_color = 'inferno'
        self.plot_wave_color = 'mediumspringgreen'

        # wave/spectral
        # line width parameters with `thin`, `thick`, `mode_switch_time`
        self.line_width_params = [.2, 1.2, 3]
        self.graphics_ratio = 5
        self.figure_size = (12, 4)
        self.figure_dpi = 300

    def _mel_filter(self, spectral_raw):
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

    def _spectral_transform(self, spectral, clip_power=None, max_norm=True):
        if clip_power is None:
            clip_power = self.min_hearing_power
        if self.spectral_transform_v == 'power':
            spectral = np.power(spectral, self.spectral_power_transform_coefficient)
        elif self.spectral_transform_v == 'log':
            spectral = np.clip(spectral, clip_power, None)
            spectral = np.log(spectral)
        else:
            raise ValueError(f'spectral transform `Values` [{self.spectral_transform_v}] unrecognized')
        if max_norm:
            spectral = self._max_norm(spectral)
        return spectral

    def _data_prepare(self, starting_time, ending_time):
        """ prepare partition of audios """
        starting_time, ending_time = self._fix_input_starting_ending_time(starting_time, ending_time)
        if not self._check_audio_duration_valid(starting_time, ending_time, self.min_duration):
            return None, None, starting_time, ending_time, False
        data_, time_ = self._export_audio(starting_time, ending_time, audio_part_path=self.audio_part_path)
        return data_, time_, starting_time, ending_time, True

    def _plot_wave(self, data_one, time_, grid, plot_position):
        """ plot wave """
        # plot audio wave
        fig_wave = plt.subplot(grid[plot_position, 0])
        # create a function to define line width
        duration = time_[-1] - time_[0]
        if duration > self.line_width_params[2]:
            line_width = self.line_width_params[0]
        else:
            line_width = (self.line_width_params[1] - (self.line_width_params[1] - self.line_width_params[0]) /
                          self.line_width_params[-1] * duration)
        # plot norm
        if np.max(np.abs(data_one)):
            data_one_norm = data_one / np.max(np.abs(data_one))
            fig_wave.plot(time_, data_one_norm, linewidth=line_width, color=self.plot_wave_color, alpha=0.3, zorder=1)
        # plot wave
        fig_wave.plot(time_, data_one, linewidth=line_width, color=self.plot_wave_color, zorder=2)
        fig_wave.set_xlim(left=time_[0], right=time_[-1])
        fig_wave.axes.get_yaxis().set_ticks([])
        fig_wave.axes.set_ylim([-1, 1])
        fig_wave.spines['left'].set_visible(False)
        fig_wave.spines['right'].set_visible(False)
        fig_wave.spines['top'].set_visible(False)
        if plot_position == self.channel_num - 1:
            fig_wave.spines['bottom'].set_color(self.plot_axis_color)
            fig_wave.tick_params(axis='x', colors=self.plot_axis_color)
        else:
            fig_wave.axes.get_xaxis().set_ticks([])
            fig_wave.spines['bottom'].set_visible(False)

    def _plot_spectral(self, data_one, grid, plot_position):
        """ plot spectral """
        # plot spectral
        spectral = self._calc_sp(data_one, self.n_window, self.n_overlap)
        if self.spectral_transform_y == 'fbank':
            spectral = self._mel_filter(spectral)
        elif self.spectral_transform_y == 'fft':
            pass
        else:
            raise ValueError(f'spectral transform `Y` [{self.spectral_transform_y}] unrecognized')
        # transform to show
        spectral = self._spectral_transform(spectral)
        spectral = np.flip(spectral, axis=1)
        spectral = np.transpose(spectral)

        # plot
        fig_spectral = plt.subplot(
            grid[self.channel_num + (self.graphics_ratio - 1) * plot_position:
                 self.channel_num + (self.graphics_ratio - 1) * (plot_position + 1), 0])
        fig_spectral.imshow(spectral, aspect='auto', cmap=self.plot_spectral_color)
        fig_spectral.axis('off')

    def _plot_phase(self, data, grid):
        """ plot phase """
        # plot phase
        data_0 = data[0]
        data_1 = data[1]
        fft_0, phase_0 = self._calc_sp(data_0, self.n_window, self.n_overlap, angle=True)
        fft_1, phase_1 = self._calc_sp(data_1, self.n_window, self.n_overlap, angle=True)

        fft_data = fft_0 + fft_1
        fft_magnitude_tendency = self._max_norm([
            self._spectral_transform(fft_0, max_norm=False),
            self._spectral_transform(fft_1, max_norm=False)])
        fft_magnitude_tendency = np.abs(fft_magnitude_tendency[1] - fft_magnitude_tendency[0])
        phase_data = phase_1 - phase_0
        phase_data = np.mod(phase_data / 2 / np.pi, 1)
        if self.spectral_transform_y == 'fbank':
            phase_data = self._mel_filter(phase_data)
            fft_data = self._mel_filter(fft_data)
            fft_magnitude_tendency = self._mel_filter(fft_magnitude_tendency)
        elif self.spectral_transform_y == 'fft':
            pass
        else:
            raise ValueError(f'phase transform `Y` [{self.spectral_transform_y}] unrecognized')
        fft_data = self._max_norm(self._spectral_transform(fft_data, clip_power=self.min_hearing_power),
                                  min_transform=True)
        fft_magnitude_tendency = self._max_norm(fft_magnitude_tendency)
        image_reconstruction = []
        for i in range(len(fft_data)):
            one_row = []
            for j in range(len(fft_data[0])):
                h = phase_data[i][j]
                s = 1 - fft_magnitude_tendency[i][j]
                v = fft_data[i][j]
                rgb = colorsys.hsv_to_rgb(h, s, v)
                one_row.append(rgb)
            image_reconstruction.append(one_row)
        # transform to show
        image_reconstruction = np.flip(image_reconstruction, axis=1)
        image_reconstruction = np.transpose(image_reconstruction, axes=(1, 0, 2))
        # plot
        fig_phase = plt.subplot(
            grid[self.channel_num:self.graphics_ratio * self.channel_num, 0])
        fig_phase.imshow(image_reconstruction, aspect='auto')
        fig_phase.axis('off')

    def _prepare_graph_audio(self, starting_time, ending_time):
        """ prepare graphics and audio files """
        phase = self._phase_mode_check()
        # default settings
        grid = plt.GridSpec(self.graphics_ratio * self.channel_num, 1, wspace=0, hspace=0)
        fig = plt.figure(figsize=self.figure_size)
        data_, time_, starting_time, ending_time, valid = self._data_prepare(starting_time, ending_time)
        self._initialize_spectral(starting_time, ending_time)
        if not valid:
            return starting_time, ending_time, False
        # plot image
        if phase:
            self._plot_phase(data_, grid)
        for i in range(len(data_)):
            if not phase:
                self._plot_spectral(data_[i], grid, i)
            self._plot_wave(data_[i], time_, grid, i)

        # save figure
        fig.savefig(self.wave_spectral_graphics_path, dpi=self.figure_dpi, bbox_inches='tight')
        self._matplotlib_clear_memory(fig)
        return starting_time, ending_time, True

    def _initial_or_restore_running(self, starting_time=None, ending_time=None, plot=True):
        """ first run & restore run """
        self._initialization()
        if starting_time is None:
            starting_time = 0
        if ending_time is None:
            ending_time = self._get_audio_time()
        self._initialize_spectral(starting_time, ending_time)
        self._prepare_graph_audio(starting_time, ending_time)
        if plot:
            self._terminal_plot(self.wave_spectral_graphics_path)
