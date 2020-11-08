import glob
import os
import readline
import shutil
import subprocess
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile as sf
from matplotlib.patches import Circle


class TerminalSeeAudio(object):
    """
    this class will plot audio similar to `Adobe Audition` with:
        * plot wave & spectral
        * play music
    """

    def __init__(self):
        # demo mode
        self.demo_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'demo/june.ogg'))

        # io parameters
        self.input = None
        self.temp_folder = os.path.join(os.path.dirname(__file__), 'tmp')
        self.readme_path = os.path.join(os.path.dirname(__file__), 'README.md')

        # system parameters
        # audio parameters
        self.sample_rate = 16000
        self.min_sample_rate = 1000
        self.a4_frequency = 440
        # `mono` or `stereo` (>1 channel)
        self.channel_type = 'stereo'

        # spectral mode
        self.spectral_transform_y = 'fbank'
        self.spectral_transform_v = 'log'

        # plot & play command (path will be replaced by `{}`)
        self.plot_command = 'timg {}'
        self.play_command = 'play {}'

        # figure parameters
        # wave/spectral
        # line width parameters with `thin`, `thick`, `mode_switch_time`
        self.line_width_params = [.2, 1.2, 3]
        self.graphics_ratio = 5
        self.figure_size = (12, 4)
        self.figure_dpi = 200

        # spiral analyzer
        self.spiral_dpi = 150
        self.spiral_figure_size = (15, 15)
        self.spiral_line_width = 1.5
        # default for 12 equal temperament
        self.spiral_n_temperament = 12

        # piano analyzer
        self.piano_line_width = 0.8
        self.piano_figure_size = (15, 5)
        self.piano_dpi = 300
        self.piano_position_gap = 0.3
        self.piano_cover_width = 0.1
        self.piano_key_range = [-48, 40]
        self.piano_tuning_shape_power = 1 / 2

        # audio parameters
        # resolution of frequency (y) dimension
        self.n_window = 1024
        self.analyzer_n_window = 4096
        # resolution of time (x) dimension
        self.n_step = 128
        # max duration for audio to play (30s)
        self.play_max_duration = 30
        # spectral transform power coefficient for `power` mode
        self.spectral_power_transform_coefficient = 1 / 5
        # minimum hearing power for `log` mode
        self.min_hearing_power = 0.0005
        self.min_analyze_power = 0.1
        # minimum hearing frequency
        self.min_mel_freq = 16

        # shell parameters
        # timeout for shell script
        self.sh_timeout = 10

        # colors & themes
        self.plot_axis_color = 'white'
        self.plot_spectral_color = 'magma'
        self.plot_wave_color = 'mediumspringgreen'
        self.spiral_color = 'mediumspringgreen'
        self.a_pitch_color = 'red'
        self.spiral_axis_color = '#444'
        self.piano_base_color = '#222'
        self.piano_key_color = 'mediumspringgreen'

        # graphics modes
        self.graphics_modes = ['fft', 'fbank', 'power', 'log', 'mono', 'stereo']

    def _get_and_fix_input_path(self, in_path):
        """ get input path """
        # if `None`: demo mode
        if in_path is None:
            if os.path.exists(self.demo_file):
                self.input = self.demo_file
                print(f'<+> demo file `{self.demo_file}` will be tested')
            else:
                raise ValueError('demo file missing, example cannot be proceeded.')
        # regular mode
        else:
            self.input = os.path.abspath(in_path)

    def _initialization(self):
        # demo mode message
        os.makedirs(self.temp_folder, exist_ok=True)
        self._initialize_audio()
        self.n_overlap = self.n_window - self.n_step
        self.min_duration = self.n_window / self.sample_rate
        self.analyze_min_duration = self.analyzer_n_window / self.sample_rate
        self._check_audio_duration()
        # initialize path autocomplete
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self._path_autocomplete)

    def _initialize_temp(self):
        """ temp file path initialization """
        self.graphics_path = os.path.join(self.temp_folder, 'wave_spectral.png')
        self.spiral_graphics_path = os.path.join(self.temp_folder, 'spiral.png')
        self.piano_graphics_path = os.path.join(self.temp_folder, 'piano.png')
        self.ifft_audio_path = os.path.join(self.temp_folder, 'analyze_ifft.wav')
        self.audio_part_path = os.path.join(self.temp_folder, 'audio.wav')

    def _initialize_audio(self):
        """ read audio and prepare data """
        if self.channel_type == 'mono':
            self.data, _ = librosa.load(self.input, sr=self.sample_rate, mono=True)
        elif self.channel_type == 'stereo':
            self.data, _ = librosa.load(self.input, sr=self.sample_rate, mono=False)
        else:
            raise ValueError(f'audio channel type `{self.channel_type}` unrecognized')
        # fix mono mode
        self.data = np.array(self.data)
        if len(self.data.shape) == 1:
            self.data = [self.data]
        self.channel_num = len(self.data)
        self.time = range(len(self.data[0]))
        self.time = [x / self.sample_rate for x in self.time]

    def _check_audio_duration(self):
        """ check if raw audio too short """
        if len(self.data[0]) / self.sample_rate < self.min_duration:
            raise ValueError('audio too short; exit')

    def _check_spiral_duration(self, starting_time):
        """ check if raw audio too short for spiral plot """
        if len(self.data[0]) / self.sample_rate < self.analyze_min_duration + starting_time or starting_time < 0:
            return False
        else:
            return True

    def _check_audio_duration_valid(self, starting, ending):
        """ check if greater than minimum duration """
        if ending - starting < self.min_duration:
            print(f'<!> {ending} - {starting} = {ending - starting} (< {self.min_duration}; minimum duration)\n'
                  f'<!> time duration too short')
            return False
        else:
            return True

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
        mel_min = self.min_mel_freq
        mel_max = 2595 * np.log10(1 + fs / 2.0 / 700)
        mel_points = np.linspace(mel_min, mel_max, n_filter + 2)
        hz_points = 700 * (10 ** (mel_points / 2595.0) - 1)
        filter_edge = np.floor(hz_points * (n_fft + 1) / fs)

        f_bank = np.zeros((n_filter, int(n_fft / 2 + 1)))
        for m in range(1, 1 + n_filter):
            f_left = int(round(filter_edge[m - 1]))
            f_center = int(round(filter_edge[m]))
            f_right = int(round(filter_edge[m + 1]))
            # `+1` to avoid broken image (modified)
            f_right += 1

            for k in range(f_left, f_center):
                f_bank[m - 1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                f_bank[m - 1, k] = (f_right - k) / (f_right - f_center)

        filter_banks = np.dot(spectral_raw, f_bank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        return filter_banks

    @staticmethod
    def _calc_sp(audio, n_window, n_overlap):
        """
        Calculate spectrogram.
        :param audio: list(float): audio data
        :return: list(list(float)): the spectral data
        """
        ham_win = np.hamming(n_window)
        [_, _, x] = signal.spectral.spectrogram(
            audio,
            window=ham_win,
            nperseg=n_window,
            noverlap=n_overlap,
            detrend=False,
            return_onesided=True,
            mode='magnitude')
        x = x.T
        x = x.astype(np.float64)
        return x

    def _data_prepare(self, starting_time, ending_time):
        """ prepare partition of audios """
        test_ending = len(self.data[0]) / self.sample_rate
        if starting_time < 0:
            print(f'<!> reset starting time {starting_time}-->{0}')
            starting_time = 0
        if ending_time < 0 or ending_time > test_ending:
            print(f'<!> reset ending time {ending_time}-->{test_ending}')
            ending_time = test_ending
        if starting_time >= ending_time:
            print('<!> starting time >= ending time ~~> reset all')
            starting_time = 0
            ending_time = len(self.data[0]) / self.sample_rate
        if not self._check_audio_duration_valid(starting_time, ending_time):
            return None, None, starting_time, ending_time, False
        # extract starting & ending sample
        starting_sample = max(int(self.sample_rate * starting_time), 0)
        ending_sample = min(int(self.sample_rate * ending_time), len(self.data[0]))
        # make clip
        data_ = np.array([x[starting_sample:ending_sample] for x in self.data])
        data_transpose = np.transpose(data_)
        time_ = self.time[starting_sample:ending_sample]
        sf.write(self.audio_part_path, data_transpose, self.sample_rate)
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
            fig_wave.plot(time_, data_one_norm, linewidth=line_width, color=self.plot_wave_color, alpha=0.3)
        # plot wave
        fig_wave.plot(time_, data_one, linewidth=line_width, color=self.plot_wave_color)
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
        spectral -= np.min(spectral)
        if self.spectral_transform_v == 'power':
            spectral = np.power(spectral, self.spectral_power_transform_coefficient)
        elif self.spectral_transform_v == 'log':
            spectral = np.clip(spectral, self.min_hearing_power, None)
            spectral = np.log(spectral)
        else:
            raise ValueError(f'spectral transform `Values` [{self.spectral_transform_v}] unrecognized')
        spectral = np.flip(spectral, axis=1)
        spectral = np.transpose(spectral)

        # plot
        fig_spectral = plt.subplot(
            grid[self.channel_num + (self.graphics_ratio - 1) * plot_position:
                 self.channel_num + (self.graphics_ratio - 1) * (plot_position + 1), 0])
        fig_spectral.imshow(spectral, aspect='auto', cmap=self.plot_spectral_color)
        fig_spectral.axis('off')

    def _prepare_graph_audio(self, starting_time, ending_time):
        """ prepare graphics and audio files """
        # default settings
        grid = plt.GridSpec(self.graphics_ratio * self.channel_num, 1, wspace=0, hspace=0)
        plt.figure(figsize=self.figure_size)
        plt.style.use('dark_background')

        data_, time_, starting_time, ending_time, valid = self._data_prepare(starting_time, ending_time)
        if not valid:
            return starting_time, ending_time, False
        for i in range(len(data_)):
            self._plot_spectral(data_[i], grid, i)
            self._plot_wave(data_[i], time_, grid, i)

        # save figure
        plt.savefig(self.graphics_path, dpi=self.figure_dpi, bbox_inches='tight')
        return starting_time, ending_time, True

    def _fft_position_to_frequency(self, position):
        return position * self.sample_rate / self.analyzer_n_window

    def _frequency_to_pitch(self, frequency):
        return np.log2(frequency / self.a4_frequency) + 5

    @staticmethod
    def _spiral_pitch_to_plot_position(pitch, offset):
        x_position = np.cos(pitch * 2 * np.pi + np.pi) * (pitch + offset)
        y_position = -np.sin(pitch * 2 * np.pi + np.pi) * (pitch + offset)
        return x_position, y_position

    def _log_min_max_transform(self, array):
        array = np.array([np.log(x + self.min_analyze_power) for x in array])
        array -= np.min(array)
        if np.max(array) != 0:
            array /= np.max(array)
        return array

    def _spiral_polar_transform(self, arrays):
        array_0, array_1 = arrays
        array_0 = self._log_min_max_transform(array_0)
        array_1 = self._log_min_max_transform(array_1)
        x_array_0 = []
        y_array_0 = []
        x_array_1 = []
        y_array_1 = []
        pitches = []
        transformed_array = []
        for i in range(len(array_0)):
            t0 = array_0[i]
            t1 = array_1[i]
            # skip low frequency part
            if i > 0:
                pitch = self._frequency_to_pitch(self._fft_position_to_frequency(i))
                if pitch > 0:
                    pitches.append(pitch)
                    transformed_array.append((t0 + t1) / 2)
                    x_position, y_position = self._spiral_pitch_to_plot_position(pitch, -t1 / 2)
                    x_array_0.append(x_position)
                    y_array_0.append(y_position)
                    x_position, y_position = self._spiral_pitch_to_plot_position(pitch, t0 / 2)
                    x_array_1.append(x_position)
                    y_array_1.append(y_position)
        return (x_array_0, y_array_0), (x_array_1, y_array_1), transformed_array, pitches

    def _get_ifft_data_single(self, fft_single):
        ff1 = np.array(list(fft_single) + list(-fft_single[::-1]))
        ifft_single = np.real(np.fft.ifft(ff1))
        ifft_single /= np.max(np.abs(ifft_single))
        return ifft_single[:self.analyzer_n_window]

    def _fft_data_transform_single(self, fft_single):
        fft_single = self._calc_sp(fft_single, self.analyzer_n_window, self.n_overlap)[0]
        if np.max(fft_single) != 0:
            fft_single /= np.max(fft_single)
        return fft_single

    def _ifft_audio_export(self, fft_data):
        ifft_data = np.transpose(np.array([self._get_ifft_data_single(x) for x in fft_data]))
        sf.write(self.ifft_audio_path, ifft_data, samplerate=self.sample_rate)

    def _prepare_graph_spiral(self, starting_time):
        valid = self._check_spiral_duration(starting_time)
        if not valid:
            print(
                f'<!> starting time set false\n'
                f'<!> number should be `0`~ `{len(self.data[0]) / self.sample_rate - self.analyze_min_duration}`s'
            )
            return False
        else:
            # get starting sample index
            starting_sample = int(starting_time * self.sample_rate)
            # get data for spectral
            if len(self.data) != 2:
                audio_data = np.sum(self.data, axis=0)
                audio_data = [audio_data, audio_data]
            else:
                audio_data = [x[starting_sample:starting_sample + self.analyzer_n_window] for x in self.data]
            fft_data = [self._fft_data_transform_single(x) for x in audio_data]

            # prepare data
            position_0, position_1, fft_data_transformed, pitches = self._spiral_polar_transform(fft_data)
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
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=self.spiral_figure_size)

            # plot ticks for `n` temperament
            for i in range(len(pitch_end_ticks_position)):
                plt.plot([pitch_start_ticks_position[i][0], pitch_end_ticks_position[i][0]],
                         [pitch_start_ticks_position[i][1], pitch_end_ticks_position[i][1]],
                         c=self.spiral_axis_color, zorder=1, alpha=0.4, linewidth=self.spiral_line_width)
                cir_start = Circle((pitch_start_ticks_position[i][0], pitch_start_ticks_position[i][1]), radius=0.05,
                                   zorder=2, facecolor='black', edgecolor=self.spiral_axis_color, alpha=0.4)
                cir_end = Circle((pitch_end_ticks_position[i][0], pitch_end_ticks_position[i][1]), radius=0.05,
                                 zorder=2, facecolor='black', edgecolor=self.spiral_axis_color, alpha=0.4)
                ax.add_patch(cir_start)
                ax.add_patch(cir_end)

            # plot base axis
            plt.plot(position_0[0], position_0[1], c=self.spiral_axis_color, linewidth=self.spiral_line_width, zorder=2,
                     alpha=0.4)

            # plot spiral
            for i in range(len(position_0[0]) - 1):
                pos0 = [position_0[0][i], position_0[1][i]]
                pos1 = [position_1[0][i], position_1[1][i]]
                pos2 = [position_1[0][i + 1], position_1[1][i + 1]]
                pos3 = [position_0[0][i + 1], position_0[1][i + 1]]
                poly_position = np.array([pos0, pos1, pos2, pos3])
                opacity = max(fft_data_transformed[i], fft_data_transformed[i + 1])
                plt.fill(poly_position[:, 0], poly_position[:, 1], facecolor=self.spiral_color,
                         edgecolor=self.spiral_color, linewidth=self.spiral_line_width,
                         alpha=opacity, zorder=2)
            # plot `A4` position
            cir_end = Circle((ax_position, ay_position), radius=0.2, zorder=3, facecolor=self.a_pitch_color,
                             linewidth=self.spiral_line_width, edgecolor=self.a_pitch_color, alpha=0.6)
            ax.add_patch(cir_end)

            # set figure ratio
            plt.gca().set_aspect(1)
            plt.axis('off')

            # save figure
            plt.savefig(self.spiral_graphics_path, dpi=self.spiral_dpi, bbox_inches='tight')

            # prepare ifft play
            self._ifft_audio_export(fft_data)
            return True

    def _frequency_to_key(self, frequency):
        return np.log2(frequency / self.a4_frequency) * 12

    def _generate_key_position(self, key, channel):
        # `0` as white, `1` as black
        key_bw_switch = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
        # `7` for octave switch
        key_position_switch = [0, 0.5, 1, 2, 2.5, 3, 3.5, 4, 5, 5.5, 6, 6.5]
        key_position = key % 12
        key_octave = int(np.ceil((key + 0.5) / 12)) - 1
        middle_x = key_octave * 7 + key_position_switch[key_position]
        # get position dimension
        # b/w width: 13.7mm/23.5mm
        # b/w length: 9.5mm/15cm
        piano_key_length = 6.382978723404255
        if key_bw_switch[key_position] == 0:
            width = 1 / 2
            length = piano_key_length
        else:
            width = 0.5829787234042553 / 2
            length = piano_key_length * 0.633
        # piano gap
        gap = channel * self.piano_position_gap
        # key position
        position_0 = [middle_x - width, -channel * piano_key_length - gap]
        position_1 = [middle_x + width, -channel * piano_key_length - gap]
        position_2 = [middle_x + width, -channel * piano_key_length - length - gap]
        position_3 = [middle_x - width, -channel * piano_key_length - length - gap]
        return np.array([position_0, position_1, position_2, position_3]), key_bw_switch[key_position]

    def _piano_key_spectral_data(self, array):
        array = self._log_min_max_transform(array)
        key_dict = {}
        for i, t in enumerate(array):
            if i > 0:
                raw_key = self._frequency_to_key(self._fft_position_to_frequency(i))
                key = round(raw_key)
                if self.piano_key_range[0] <= key < self.piano_key_range[1]:
                    if key not in key_dict:
                        key_dict[key] = [[t, raw_key]]
                    else:
                        key_dict[key].append([t, raw_key])
        for k, v in key_dict.items():
            # set `^` or `n` shape tuning
            v = np.array(v)
            key_dict[k] = np.mean(v[:, 0] * np.power(1 - 2 * np.abs(k - v[:, 1]), self.piano_tuning_shape_power))
        max_value = max(list(key_dict.values()))
        for k, v in key_dict.items():
            key_dict[k] = v / max_value
        return key_dict

    def _piano_graph_single(self, key_dict, channel):
        # plot cover
        left_most, _ = self._generate_key_position(self.piano_key_range[0], channel)
        right_most, _ = self._generate_key_position(self.piano_key_range[1] - 1, channel)
        cover_x_positions = [left_most[0, 0], right_most[1, 0], right_most[1, 0], left_most[0, 0]]
        cover_y_positions = [left_most[0, 1], left_most[0, 1], left_most[0, 1] - self.piano_cover_width,
                             left_most[0, 1] - self.piano_cover_width]
        plt.fill(cover_x_positions, cover_y_positions, edgecolor=self.piano_base_color, facecolor=self.piano_base_color,
                 linewidth=self.piano_line_width, zorder=5, alpha=0.9)
        # plot key
        for k in range(self.piano_key_range[0], self.piano_key_range[1], 1):
            positions, bw = self._generate_key_position(k, channel)
            if k in key_dict:
                fft_values = key_dict[k]
            else:
                fft_values = 0
            # background
            plt.fill(positions[:, 0], positions[:, 1], facecolor='black', edgecolor=self.piano_base_color,
                     linewidth=self.piano_line_width, zorder=2 * bw + 1)
            # plot key
            plt.fill(positions[:, 0], positions[:, 1], edgecolor=self.piano_key_color, facecolor=self.piano_key_color,
                     linewidth=self.piano_line_width, zorder=2 * bw + 2, alpha=fft_values)
            # `a4` position
            if k % 12 == 0:
                if k == 0:
                    opacity = 0.5
                else:
                    opacity = 0.15
                plt.fill(positions[:, 0], cover_y_positions,
                         edgecolor=self.a_pitch_color, facecolor=self.a_pitch_color, linewidth=self.piano_line_width,
                         zorder=6, alpha=opacity)

    def _prepare_graph_piano(self, starting_time):
        valid = self._check_spiral_duration(starting_time)
        if not valid:
            print(
                f'<!> starting time set false\n'
                f'<!> number should be `0`~ `{len(self.data[0]) / self.sample_rate - self.analyze_min_duration}`s'
            )
            return False
        else:
            # get starting sample index
            starting_sample = int(starting_time * self.sample_rate)
            # get data for spectral
            audio_data = [x[starting_sample:starting_sample + self.analyzer_n_window] for x in self.data]
            fft_data = [self._fft_data_transform_single(x) for x in audio_data]
            key_dicts = [self._piano_key_spectral_data(x) for x in fft_data]

            # plot
            plt.style.use('dark_background')

            plt.figure(figsize=self.piano_figure_size)
            fig = plt.subplot(111)
            fig.set_xlim([self._generate_key_position(self.piano_key_range[0], 0)[0][0, 0] - 0.5,
                          self._generate_key_position(self.piano_key_range[1] - 1, 0)[0][1, 0] + 0.5])
            # plot piano
            for i in range(len(fft_data)):
                self._piano_graph_single(key_dicts[i], i)

            # set plot ratio
            plt.gca().set_aspect(1)
            plt.axis('off')

            plt.savefig(self.piano_graphics_path, dpi=self.piano_dpi, bbox_inches='tight')

            # prepare ifft play
            self._ifft_audio_export(fft_data)
            return True

    def _terminal_plot(self, path):
        """ plot in terminal function """
        if not os.path.exists(path):
            print('<!> temp image cannot find')
            return
        command = self.plot_command.format(path)
        # noinspection PyBroadException
        try:
            subprocess.call(command, shell=True)
        except Exception:
            print(f'<!> please fix problem:\n<?> {command}')

    def _terminal_play(self, start, end, path):
        """ play in terminal function """
        if not os.path.exists(path):
            print('<!> temp audio cannot find')
            return
        if end - start > self.play_max_duration:
            print(f'<!> audio too long for {end - start}s')
            while True:
                answer = input('</> do you wish to play [y/n]: ')
                if answer == 'y':
                    break
                elif answer == 'n':
                    return
                else:
                    print('<!> please type `y` or `n`')
                    continue
        command = self.play_command.format(path)
        # noinspection PyBroadException
        try:
            subprocess.call(command, shell=True)
        except Exception:
            print(f'<!> please fix problem:\n<?> {command}')

    @staticmethod
    def _is_float(string):
        # noinspection PyBroadException
        try:
            float(string)
            return True
        except Exception:
            return False

    @staticmethod
    def _is_int(string):
        # noinspection PyBroadException
        try:
            int(string)
            return True
        except Exception:
            return False

    def _initial_or_restore_running(self):
        """ first run & restore run """
        self._prepare_graph_audio(0, len(self.data[0]) / self.sample_rate)
        self._terminal_plot(self.graphics_path)

    @staticmethod
    def _path_input_check(input_split):
        return input_split[1][0] == input_split[1][-1] and (
                input_split[1][0] == '\'' or input_split[1][0] == '\"' or input_split[1][0] == '`')

    def _get_try_path(self, input_split):
        if self._path_input_check(input_split):
            try_input = input_split[1][1:-1]
        else:
            try_input = input_split[1]
        return try_input

    def print_help(self):
        """ print help file """
        if os.path.exists(self.readme_path):
            print('<*> ' + 'help ...')
            with open(self.readme_path, 'r') as f:
                text = f.readlines()
                for t in text:
                    print(' | ' + t.rstrip())
            print('<*> ' + '... help')
        else:
            print('<!> readme file missing')

    @staticmethod
    def _path_autocomplete(text, state):
        """
        [https://gist.github.com/iamatypeofwalrus/5637895]
        This is the tab completer for systems paths.
        Only tested on *nix systems
        --> WATCH: space ` ` is replaced by `\\s`
        """
        # replace ~ with the user's home dir
        if '~' in text:
            text = text.replace('~', os.path.expanduser('~'))

        # fix path
        if text != '':
            text = os.path.abspath(text).replace('//', '/')

        # autocomplete directories with having a trailing slash
        if os.path.isdir(text) and text != '/':
            text += '/'

        return [x.replace(' ', '\\s') for x in glob.glob(text.replace('\\s', ' ') + '*')][state]

    def _get_time(self):
        return 0, len(self.data) / self.sample_rate

    def main(self, in_path):
        """ main function """
        self._get_and_fix_input_path(in_path)
        # initialization
        self._initialization()
        self._initialize_temp()
        # prepare
        last_starting = 0
        last_ending = len(self.data[0]) / self.sample_rate
        # 0. first run
        self._initial_or_restore_running()
        # loop to get inputs
        while True:
            print('-' * 50)
            input_ = input('</> ').strip()

            # 1. multiple input function (calculation)
            # 1.1 `=` for advanced script
            # watch: space ` ` will be replaced by `\s`
            if len(input_) >= 1:
                command = input_[1:].strip()
            else:
                command = ''
            if input_.startswith('='):
                command_success = False
                command_result = []
                # 1.1.1 to evaluate
                if not command_success:
                    try:
                        return_string = eval(command)
                        print(f'<*> {return_string}')
                        command_success = True
                        continue
                    except Exception as e:
                        command_result.append(e)
                # 1.1.2 to execute
                if not command_success:
                    try:
                        exec(command)
                        self._initialization()
                        self._prepare_graph_audio(last_starting, last_ending)
                        print(f'<*> executed `{command}`')
                        command_success = True
                        continue
                    except Exception as e:
                        command_result.append(e)
                # 1.1.* advanced script error
                if not command_success:
                    print(f'<!> evaluate error message: `{command_result[0]}`')
                    print(f'<!> execute error message: `{command_result[1]}`')
                    continue
            # 1.2 get spiral (`@`) analyzer
            elif input_.startswith('@'):
                # 1.2.1 number as starting time
                if self._is_float(command):
                    status = self._prepare_graph_spiral(float(command))
                    if status:
                        self._terminal_plot(self.spiral_graphics_path)
                # 1.2.2 plot last image
                elif command == '':
                    self._terminal_plot(self.spiral_graphics_path)
                else:
                    print('<!> `spiral` inputs unknown')
                continue
            # 1.3 get piano (`#`) analyzer
            elif input_.startswith('#'):
                # 1.3.1 number as staring time
                if self._is_float(command):
                    status = self._prepare_graph_piano(float(command))
                    if status:
                        self._terminal_plot(self.piano_graphics_path)
                # 1.3.2 plot last image
                elif command == '':
                    self._terminal_plot(self.piano_graphics_path)
                else:
                    print('<!> `piano` inputs unknown')
                continue

            # 2. contain space case
            if ' ' in input_:
                # 2.0 prepare
                space_idx = input_.find(' ')
                input_split = [input_[:space_idx], input_[space_idx + 1:]]
                # 2.1 shell command
                if input_split[0] == 'sh':
                    # noinspection PyBroadException
                    try:
                        sh_output = subprocess.check_output(input_split[1].replace('\\s', '\\ '), shell=True,
                                                            stderr=subprocess.STDOUT, timeout=self.sh_timeout)
                        print(f'<*> {str(sh_output.decode("utf-8").strip())}')
                    except Exception as e:
                        print(f'<!> error message: `{e}`')
                    continue

                # 2.2 two input functions
                if ' ' not in input_split[1] or self._path_input_check(input_split):
                    try_input = self._get_try_path(input_split).replace('\\s', ' ')
                    # 2.2.0 set time parameters for wave spectral plot
                    if self._is_float(input_split[0]) and self._is_float(input_split[1]):
                        last_starting, last_ending, valid = self._prepare_graph_audio(float(input_split[0]),
                                                                                      float(input_split[1]))
                        if valid:
                            self._terminal_plot(self.graphics_path)
                    # 2.2.1 set modes
                    elif input_split[0] == 'm':
                        if input_split[1] in self.graphics_modes:
                            if input_split[1] in ['fft', 'fbank']:
                                self.spectral_transform_y = input_split[1]
                            elif input_split[1] in ['power', 'log']:
                                self.spectral_transform_v = input_split[1]
                            elif input_split[1] in ['mono', 'stereo']:
                                self.channel_type = input_split[1]
                                self._initialize_audio()
                            # recalculating
                            self._prepare_graph_audio(last_starting, last_ending)
                            print(f'<+> mode `{input_split[1]}` set')
                        else:
                            print(f'<?> mode `{input_split[1]}` unknown\n<!> modes are within {self.graphics_modes}')
                    # 2.2.2 set sample rate
                    elif input_split[0] == 'sr':
                        if self._is_int(input_split[1]):
                            if int(input_split[1]) >= self.min_sample_rate:
                                self.sample_rate = int(input_split[1])
                                self._initialize_audio()
                                # recalculating
                                self._prepare_graph_audio(last_starting, last_ending)
                                print(f'<+> sample rate `{input_split[1]}` set')
                            else:
                                print(f'<!> sample rate `{input_split[1]}` (< {self.min_sample_rate}) too low')
                        else:
                            print(f'<!> sample rate `{input_split[1]}` unknown')
                    # 2.2.3 switch file to open
                    elif input_split[0] == 'o':
                        if os.path.exists(try_input):
                            if self.input == try_input:
                                print('<!> same file path')
                            else:
                                self.input = os.path.abspath(try_input)
                                self._initialize_audio()
                                print('<+> file path changed')
                                self._initial_or_restore_running()
                                # reset time
                                last_starting, last_ending = self._get_time()
                        else:
                            print(f'<!> file path `{try_input}` does not exist')
                    # 2.2.4 change the temp folder path
                    elif input_split[0] == 'tmp':
                        if not os.path.exists(try_input):
                            # remove old temp folder
                            shutil.move(self.temp_folder, try_input)
                            print(f'<+> temp folder path changed: `{self.temp_folder}` --> `{try_input}`')
                            # set new temp folder
                            self.temp_folder = try_input
                            self._initialize_temp()
                        else:
                            print(f'<!> file path `{try_input}` already exist')
                    # 2.2.* two inputs case error
                    else:
                        print('<!> two inputs case unknown')
                    continue
                # 2.* too many inputs error
                else:
                    print('<!> please check number of input')
                    continue

            # 3. single input
            # 3.1 `p` to play music
            elif input_ == 'p':
                self._terminal_play(last_starting, last_ending, self.audio_part_path)
            # 3.2 `p*` to play short period audio analyzed by spectral analyzer
            elif input_ == 'p*':
                self._terminal_play(0, 0.1, self.ifft_audio_path)
            # 3.3 `` to show last image
            elif input_ == '':
                self._terminal_plot(self.graphics_path)
            # 3.4 `q` to quit program
            elif input_ == 'q':
                break
            # 3.5 `r` to reset all
            elif input_ == 'r':
                print('<!> reset all')
                self._initial_or_restore_running()
                # reset time
                last_starting, last_ending = self._get_time()
            # 3.6 `h` to print help file
            elif input_ == 'h':
                self.print_help()
            # 3.* single input case error
            else:
                print('<!> unknown command')
                continue
        # remove temp folder at quit
        shutil.rmtree(self.temp_folder)


if __name__ == '__main__':
    tsa = TerminalSeeAudio()

    # demo mode
    if len(sys.argv) == 1:
        tsa.main(None)
    # argument error
    elif len(sys.argv) > 2:
        print('argument error, please check number of arguments')
    # default mode
    else:
        input_path = sys.argv[1]
        if input_path in ['-h', '--help']:
            TerminalSeeAudio().print_help()
        elif not os.path.exists(input_path):
            print(f'path [{input_path}] does not exist!')
        else:
            tsa.main(input_path)
