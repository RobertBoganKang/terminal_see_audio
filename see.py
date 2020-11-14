import glob
import os
import readline
import shutil
import subprocess
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import peakutils
import scipy.signal as signal
import soundfile as sf
from matplotlib.patches import Circle


class Common(object):
    def __init__(self):
        # demo mode
        self.demo_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'demo/june.ogg'))

        # io parameters
        self.input = None
        self.temp_folder = os.path.join(os.path.dirname(__file__), 'tmp')
        self.readme_path = os.path.join(os.path.dirname(__file__), 'README.md')

        # constants
        self.golden_ratio = (np.sqrt(5) - 1) / 2
        self.a4_frequency = 440

        # figure parameters
        self.figure_minimum_alpha = 0.05

        # audio parameters
        self.sample_rate = 16000
        self.min_sample_rate = 1000
        # `mono` or `stereo` (>1 channel)
        self.channel_type = 'stereo'

        # resolution of frequency (y) dimension
        self.n_window = 1024
        # analyzer time
        self.analyzer_time = 0.256
        # resolution of time (x) dimension
        self.n_step = 128
        self.piano_roll_n_step = 1024
        # max duration for audio to play (30s)
        self.play_max_duration = 30
        # spectral transform power coefficient for `power` mode
        self.spectral_power_transform_coefficient = 1 / 5
        # minimum hearing power for `log` mode
        self.min_hearing_power = 0.0005
        self.min_analyze_power = 0.1
        # minimum hearing frequency
        self.min_mel_freq = 16

        # plot & play command (path will be replaced by `{}`)
        self.plot_command = 'timg {}'
        self.play_command = 'play {}'

        # color & themes
        self.a_pitch_color = 'red'

        # graphics modes
        self.graphics_modes = ['fft', 'fbank', 'power', 'log', 'mono', 'stereo']

    def _check_audio_duration_valid(self, starting, ending):
        """ check if greater than minimum duration """
        if ending - starting < self.min_duration:
            print(f'<!> {ending} - {starting} = {ending - starting} (< {self.min_duration}; minimum duration)\n'
                  f'<!> time duration too short')
            return False
        else:
            return True

    def _initialization(self):
        # demo mode message
        os.makedirs(self.temp_folder, exist_ok=True)
        self._initialize_audio()
        self.n_overlap = self.n_window - self.n_step
        self.analyzer_n_window = int(self.analyzer_time * self.sample_rate)
        self.piano_roll_n_overlap = self.analyzer_n_window - self.piano_roll_n_step
        self.min_duration = self.n_window / self.sample_rate
        self.analyze_min_duration = self.analyzer_n_window / self.sample_rate
        self._check_audio_duration()

    def _initialize_temp(self):
        """ temp file path initialization """
        self.graphics_path = os.path.join(self.temp_folder, 'wave_spectral.png')
        self.spiral_graphics_path = os.path.join(self.temp_folder, 'spiral.png')
        self.piano_graphics_path = os.path.join(self.temp_folder, 'piano.png')
        self.piano_roll_graphics_path = os.path.join(self.temp_folder, 'piano_roll.png')
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
            self.data = np.array([self.data])
        self.channel_num = len(self.data)
        self.time = range(len(self.data[0]))
        self.time = [x / self.sample_rate for x in self.time]

    def _get_audio_time(self):
        return 0, len(self.data[0]) / self.sample_rate

    def _check_audio_duration(self):
        """ check if raw audio too short """
        if self._get_audio_time()[-1] < self.min_duration:
            raise ValueError('audio too short; exit')

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

    def _fix_input_starting_ending_time(self, starting_time, ending_time):
        test_ending = self._get_audio_time()[-1]
        if starting_time < 0:
            print(f'<!> reset starting time {starting_time}-->{0}')
            starting_time = 0
        if ending_time < 0 or ending_time > test_ending:
            print(f'<!> reset ending time {ending_time}-->{test_ending}')
            ending_time = test_ending
        if starting_time >= ending_time:
            print('<!> starting time >= ending time ~~> reset all')
            starting_time = 0
            ending_time = self._get_audio_time()[-1]
        return starting_time, ending_time

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


class Shell(Common):
    def __init__(self):
        # shell parameters
        # timeout for shell script
        super().__init__()
        self.sh_timeout = 10

        # initialize shell
        self._initialize_shell()

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

    def _initialize_shell(self):
        # initialize path autocomplete
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self._path_autocomplete)


class AnalyzeCommon(Common):
    def __init__(self):
        super().__init__()

    def _log_min_max_transform(self, array, log=True):
        if log:
            array = np.log(np.array(array) + self.min_analyze_power)
        array -= np.min(array)
        if np.max(array) != 0:
            array /= np.max(array)
        return array

    def _fft_position_to_frequency(self, position):
        return position * self.sample_rate / self.analyzer_n_window

    def _frequency_to_pitch(self, frequency):
        return np.log2(frequency / self.a4_frequency) + 5

    def _fft_data_transform_single(self, fft_single):
        fft_single = self._calc_sp(fft_single, self.analyzer_n_window, self.n_overlap)
        return fft_single

    def _check_analyze_duration(self, starting_time):
        """ check if raw audio too short for analyze plot """
        if self._get_audio_time()[-1] < self.analyze_min_duration + starting_time or starting_time < 0:
            return False
        else:
            return True

    def _get_ifft_data_single(self, fft_single):
        ff1 = np.array(list(fft_single) + list(-fft_single[::-1]))
        ifft_single = np.real(np.fft.ifft(ff1))
        ifft_single /= np.max(np.abs(ifft_single))
        return ifft_single[:self.analyzer_n_window]

    def _ifft_audio_export(self, fft_data):
        ifft_data = np.transpose(np.array([self._get_ifft_data_single(x) for x in fft_data]))
        sf.write(self.ifft_audio_path, ifft_data, samplerate=self.sample_rate)


class WaveSpectral(AnalyzeCommon):
    def __init__(self):
        super().__init__()
        # spectral mode
        self.spectral_transform_y = 'fbank'
        self.spectral_transform_v = 'log'

        # colors & themes
        self.plot_axis_color = 'white'
        self.plot_spectral_color = 'magma'
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

    def _data_prepare(self, starting_time, ending_time):
        """ prepare partition of audios """
        starting_time, ending_time = self._fix_input_starting_ending_time(starting_time, ending_time)
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

    def _initial_or_restore_running(self):
        """ first run & restore run """
        self._prepare_graph_audio(0, self._get_audio_time()[-1])
        self._terminal_plot(self.graphics_path)


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
        self.spiral_color = 'mediumspringgreen'
        self.spiral_axis_color = '#444'

    @staticmethod
    def _spiral_pitch_to_plot_position(pitch, offset):
        x_position = np.cos(pitch * 2 * np.pi + np.pi) * (pitch + offset)
        y_position = -np.sin(pitch * 2 * np.pi + np.pi) * (pitch + offset)
        return x_position, y_position

    def _spiral_polar_transform(self, arrays):
        array_0 = arrays[0]
        array_1 = arrays[1]
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

    def _prepare_graph_spiral(self, starting_time):
        valid = self._check_analyze_duration(starting_time)
        if not valid:
            print(
                f'<!> starting time set false\n'
                f'<!> number should be `0`~ `{self._get_audio_time()[-1] - self.analyze_min_duration}`s'
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
            fft_data = self._log_min_max_transform([self._fft_data_transform_single(x)[0] for x in audio_data])

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
                                   zorder=1, facecolor='black', edgecolor=self.spiral_axis_color, alpha=0.4)
                cir_end = Circle((pitch_end_ticks_position[i][0], pitch_end_ticks_position[i][1]), radius=0.05,
                                 zorder=1, facecolor='black', edgecolor=self.spiral_axis_color, alpha=0.4)
                ax.add_patch(cir_start)
                ax.add_patch(cir_end)

            # plot base axis
            plt.plot(position_0[0], position_0[1], c=self.spiral_axis_color, linewidth=self.spiral_line_width, zorder=1,
                     alpha=0.4)

            # plot spiral
            for i in range(len(position_0[0]) - 1):
                pos0 = [position_0[0][i], position_0[1][i]]
                pos1 = [position_1[0][i], position_1[1][i]]
                pos2 = [position_1[0][i + 1], position_1[1][i + 1]]
                pos3 = [position_0[0][i + 1], position_0[1][i + 1]]
                poly_position = np.array([pos0, pos1, pos2, pos3])
                opacity = max(fft_data_transformed[i], fft_data_transformed[i + 1])
                if opacity > self.figure_minimum_alpha:
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
            self._ifft_audio_export(self._log_min_max_transform(fft_data, log=False))
            return True


class PianoCommon(AnalyzeCommon):
    def __init__(self):
        super().__init__()
        self.piano_tuning_shape_power = 1 / 2
        self.piano_key_range = [-48, 40]
        self.piano_spectral_height = 0.1
        self.piano_position_gap = 0.4
        self.piano_line_width = 0.8

        # `0` as white, `1` as black key
        self.piano_key_bw_switch = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]

        # colors & themes
        self.piano_base_color = '#222'
        self.piano_roll_base_color = '#444'
        self.piano_roll_black_key_color = '#333'
        self.piano_key_color = 'mediumspringgreen'
        self.piano_roll_color = 'mediumspringgreen'
        self.piano_spectral_color = 'dimgray'

    def _piano_tuning_method(self, key, value):
        # set `^` or `n` shape tuning
        return np.mean(value[:, 0] * np.power(1 - 2 * np.abs(key - value[:, 1]), self.piano_tuning_shape_power))

    def _frequency_to_key(self, frequency):
        return np.log2(frequency / self.a4_frequency) * 12

    def _piano_key_spectral_data(self, array):
        key_dict = {}
        raw_keys = []
        key_ffts = []
        for i, t in enumerate(array):
            if i > 0:
                raw_key = self._frequency_to_key(self._fft_position_to_frequency(i))
                key = round(raw_key)
                if self.piano_key_range[0] <= key < self.piano_key_range[1]:
                    raw_keys.append(raw_key)
                    key_ffts.append(t)
                    if key not in key_dict:
                        key_dict[key] = [[t, raw_key]]
                    else:
                        key_dict[key].append([t, raw_key])
        for k, v in key_dict.items():
            v = np.array(v)
            key_dict[k] = self._piano_tuning_method(k, v)
        max_value = max(list(key_dict.values()))
        for k, v in key_dict.items():
            key_dict[k] = v / max_value
        return key_dict, raw_keys, key_ffts


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
        self.piano_key_length = 6.382978723404255

    def _piano_generate_frequency_graph_single(self, raw_key, key_fft, channel):
        # get key position
        positions_0 = []
        positions_1 = []
        for i in range(len(raw_key)):
            one_piano_length = self.piano_key_length + self.piano_spectral_height + self.piano_position_gap
            key_x = raw_key[i] * 7 / 12
            positions_0.append([key_x, -channel * one_piano_length])
            positions_1.append([key_x, -channel * one_piano_length + self.piano_spectral_height])
        for i in range(len(raw_key) - 1):
            x_positions = [positions_0[i][0], positions_1[i + 1][0], positions_1[i + 1][0], positions_0[i][0]]
            y_positions = [positions_0[i][1], positions_0[i + 1][1], positions_1[i + 1][1], positions_1[i][1]]
            freq_alpha = max(key_fft[i], key_fft[i + 1])
            if freq_alpha > self.figure_minimum_alpha:
                plt.fill(x_positions, y_positions, edgecolor=self.piano_spectral_color,
                         facecolor=self.piano_spectral_color,
                         linewidth=self.piano_line_width, zorder=1, alpha=freq_alpha)

    def _piano_graph_single(self, key_dict, channel):
        # plot cover
        left_most, _ = self._piano_generate_key_position(self.piano_key_range[0], channel)
        right_most, _ = self._piano_generate_key_position(self.piano_key_range[1] - 1, channel)
        cover_x_positions = [left_most[0, 0], right_most[1, 0], right_most[1, 0], left_most[0, 0]]
        cover_y_positions = [left_most[0, 1], left_most[0, 1], left_most[0, 1] - self.piano_cover_width,
                             left_most[0, 1] - self.piano_cover_width]
        plt.fill(cover_x_positions, cover_y_positions, edgecolor=self.piano_base_color, facecolor=self.piano_base_color,
                 linewidth=self.piano_line_width, zorder=5, alpha=0.9)
        # plot key
        for k in range(self.piano_key_range[0], self.piano_key_range[1], 1):
            positions, bw = self._piano_generate_key_position(k, channel)
            if k in key_dict:
                fft_value = key_dict[k]
            else:
                fft_value = 0
            # background
            plt.fill(positions[:, 0], positions[:, 1], facecolor='black', edgecolor=self.piano_base_color,
                     linewidth=self.piano_line_width, zorder=2 * bw + 1)
            # plot key
            if fft_value > self.figure_minimum_alpha:
                plt.fill(positions[:, 0], positions[:, 1], edgecolor=self.piano_key_color,
                         facecolor=self.piano_key_color,
                         linewidth=self.piano_line_width, zorder=2 * bw + 2, alpha=fft_value)
            # `a4` position
            if k % 12 == 0:
                if k == 0:
                    opacity = 0.5
                else:
                    opacity = 0.15
                plt.fill(positions[:, 0], cover_y_positions,
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
            length = self.piano_key_length
        else:
            width = 0.5829787234042553
            length = self.piano_key_length * 0.633
        # key position
        one_piano_length = self.piano_key_length + self.piano_spectral_height + self.piano_position_gap
        position_0 = [middle_x - width / 2, -channel * one_piano_length]
        position_1 = [middle_x + width / 2, -channel * one_piano_length]
        position_2 = [middle_x + width / 2, -channel * one_piano_length - length]
        position_3 = [middle_x - width / 2, -channel * one_piano_length - length]
        return np.array([position_0, position_1, position_2, position_3]), self.piano_key_bw_switch[key_position]

    def _prepare_graph_piano(self, starting_time):
        valid = self._check_analyze_duration(starting_time)
        if not valid:
            print(
                f'<!> starting time set false\n'
                f'<!> number should be `0`~ `{self._get_audio_time()[-1] - self.analyze_min_duration}`s'
            )
            return False
        else:
            # get starting sample index
            starting_sample = int(starting_time * self.sample_rate)
            # get data for spectral
            audio_data = [x[starting_sample:starting_sample + self.analyzer_n_window] for x in self.data]
            fft_data = [self._fft_data_transform_single(x)[0] for x in audio_data]
            spectral_data = [self._piano_key_spectral_data(x) for x in fft_data]
            key_dicts = []
            raw_keys = []
            key_ffts = []
            for key_dict, raw_key, key_fft, in spectral_data:
                key_dicts.append(key_dict)
                raw_keys.append(raw_key)
                key_ffts.append(key_fft)

            # plot
            plt.style.use('dark_background')

            plt.figure(figsize=self.piano_figure_size)
            fig = plt.subplot(111)
            fig.set_xlim(
                [self._piano_generate_key_position(self.piano_key_range[0], 0)[0][0, 0] - 0.5,
                 self._piano_generate_key_position(self.piano_key_range[1] - 1, 0)[0][
                     1, 0] + 0.5])
            # plot piano
            for i in range(len(fft_data)):
                self._piano_graph_single(key_dicts[i], i)
                self._piano_generate_frequency_graph_single(raw_keys[i], key_ffts[i], i)

            # set plot ratio
            plt.gca().set_aspect(1)
            plt.axis('off')

            plt.savefig(self.piano_graphics_path, dpi=self.piano_dpi, bbox_inches='tight')

            # prepare ifft play
            self._ifft_audio_export(self._log_min_max_transform(fft_data, log=False))
            return True


class PianoRoll(PianoCommon):
    def __init__(self):
        super().__init__()
        self.piano_roll_figure_size = (20, 15)
        self.piano_roll_dpi = 200
        self.piano_roll_cover_width = 0.2
        self.piano_roll_key_length = 8
        self.piano_roll_length_ratio = 3
        self.piano_roll_length = (
                (self.piano_key_range[1] - self.piano_key_range[0]) * self.piano_roll_length_ratio
        )

    @staticmethod
    def _piano_roll_key_to_location_range(key):
        return key - 0.5, key + 0.5

    def _piano_roll_generate_key_position(self, key):
        # `7` for octave switch
        key_position_switch = range(12)
        key_position = key % 12
        key_octave = int(np.ceil((key + 0.5) / 12)) - 1
        middle_x = key_octave * 12 + key_position_switch[key_position]
        # white key width makeup
        lower_makeup = 0
        higher_makeup = 0
        if self.piano_key_bw_switch[(key - 1) % 12] == 1:
            lower_makeup = 0.5
        if self.piano_key_bw_switch[(key + 1) % 12] == 1:
            higher_makeup = 0.5
        # get position dimension
        width = 1
        if self.piano_key_bw_switch[key_position] == 0:
            length = self.piano_roll_key_length
        else:
            length = self.piano_roll_key_length * 0.633
        # key position
        position_0 = [middle_x - width / 2 - lower_makeup, -self.piano_roll_key_length]
        position_1 = [middle_x + width / 2 + higher_makeup, -self.piano_roll_key_length]
        position_2 = [middle_x + width / 2 + higher_makeup, -self.piano_roll_key_length + length]
        position_3 = [middle_x - width / 2 - lower_makeup, -self.piano_roll_key_length + length]
        return np.array([position_0, position_1, position_2, position_3]), self.piano_key_bw_switch[key_position]

    def _piano_roll_indicator(self):
        """ piano roll base """
        # plot cover & frame
        top_most, _ = self._piano_roll_generate_key_position(self.piano_key_range[0])
        bottom_most, _ = self._piano_roll_generate_key_position(self.piano_key_range[1] - 1)
        cover_x_positions = [0, 0, - self.piano_roll_cover_width, - self.piano_roll_cover_width]
        frame_x_positions = [0, 0, self.piano_roll_length, self.piano_roll_length]
        cover_y_positions = [top_most[0, 0], bottom_most[1, 0], bottom_most[1, 0], top_most[0, 0]]
        plt.fill(cover_x_positions, cover_y_positions, edgecolor=self.piano_roll_base_color,
                 facecolor=self.piano_base_color,
                 linewidth=self.piano_line_width, zorder=5, alpha=0.9)
        plt.fill(frame_x_positions, cover_y_positions, edgecolor=self.piano_roll_base_color,
                 facecolor='black', linewidth=self.piano_line_width, zorder=1)
        base_x_positions = [0, self.piano_roll_length, self.piano_roll_length, 0]
        # plot key & piano roll base
        for k in range(self.piano_key_range[0], self.piano_key_range[1], 1):
            positions, bw = self._piano_roll_generate_key_position(k)
            bottom_position, top_position = self._piano_roll_key_to_location_range(k)
            base_y_positions = [top_position, top_position, bottom_position, bottom_position]
            # background
            if bw:
                key_color = self.piano_roll_black_key_color
                roll_color = self.piano_roll_black_key_color
                alpha = 0.6
            else:
                key_color = 'black'
                if k % 12 == 0 or k % 12 == 1:
                    roll_color = self.a_pitch_color
                    if k in {0, 1}:
                        alpha = 0.3
                    else:
                        alpha = 0.1
                else:
                    roll_color = 'black'
                    alpha = 0.6
            # plot key
            plt.fill(positions[:, 1], positions[:, 0], facecolor=key_color, edgecolor=self.piano_roll_base_color,
                     linewidth=self.piano_line_width, zorder=bw + 1)
            # plot piano roll base
            plt.fill(base_x_positions, base_y_positions, facecolor=roll_color, zorder=1, alpha=alpha)

            # plot grid
            plt.plot([base_x_positions[0], base_x_positions[1]], [base_y_positions[-1], base_y_positions[-1]],
                     c=self.piano_roll_base_color, linewidth=self.piano_line_width, alpha=0.5, zorder=4)
            # makeup edge line
            if k == self.piano_key_range[0]:
                plt.plot([base_x_positions[0], base_x_positions[1]], [base_y_positions[0], base_y_positions[0]],
                         c=self.piano_roll_base_color, linewidth=self.piano_line_width, alpha=0.5, zorder=4)

    def _piano_roll_generate_frequency_graph_single(self, key_dict, step, all_step_number):
        # get key position
        step_size = self.piano_roll_length / all_step_number
        for k, v in key_dict.items():
            key_0, key_1 = self._piano_roll_key_to_location_range(k)
            x_positions = [step * step_size, (step + 1) * step_size, (step + 1) * step_size, step * step_size]
            y_positions = [key_0, key_0, key_1, key_1]
            freq_alpha = v
            if freq_alpha > self.figure_minimum_alpha:
                plt.fill(x_positions, y_positions, facecolor=self.piano_roll_color, zorder=3, alpha=freq_alpha)

    def _prepare_graph_piano_roll(self, starting_time, ending_time):
        # fix time first
        starting_time, ending_time = self._fix_input_starting_ending_time(starting_time, ending_time)
        if ending_time - starting_time < self.analyze_min_duration:
            print('<!> audio too short to show piano roll')
            return False
        else:
            # prepare spectrum
            # extract starting & ending sample
            starting_sample = int(self.sample_rate * starting_time)
            ending_sample = int(self.sample_rate * ending_time)
            # to `mono` for piano roll
            data_ = np.mean(self.data, axis=0)
            fft_data = self._log_min_max_transform(
                self._calc_sp(data_[starting_sample:ending_sample], self.analyzer_n_window,
                              self.piano_roll_n_overlap))
            # plot
            plt.style.use('dark_background')
            plt.figure(figsize=self.piano_roll_figure_size)
            fig = plt.subplot(111)
            fig.set_ylim([self._piano_roll_generate_key_position(self.piano_key_range[0])[0][0, 0] - 0.5,
                          self._piano_roll_generate_key_position(self.piano_key_range[1] - 1)[0][1, 0] + 0.5])
            fig.set_xlim([-self.piano_roll_key_length - 0.2, self.piano_roll_length + 0.2])
            # plot piano base
            self._piano_roll_indicator()
            # plot piano roll
            for i, data in enumerate(fft_data):
                key_dict, raw_key, key_fft = self._piano_key_spectral_data(data)
                self._piano_roll_generate_frequency_graph_single(key_dict, i, len(fft_data))
            # set plot ratio
            plt.gca().set_aspect(1)
            plt.axis('off')
            self._initialize_temp()
            plt.savefig(self.piano_roll_graphics_path, dpi=self.piano_roll_dpi, bbox_inches='tight')
            return True


class PeakAnalyzer(AnalyzeCommon):
    def __init__(self):
        super().__init__()
        self.peak_analyze_coefficient = 80
        self.peak_tuning_coefficient = 40
        self.peak_analyze_min_threshold = 0.1

        # name of key name
        self.key_name_lib = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    def _frequency_to_music_pitch(self, frequency):
        raw_key = np.log2(frequency / 440) * 12
        key = round(raw_key)
        remainder_in_cent = (raw_key - key) * 100
        key_name = self.key_name_lib[key % 12]
        key_octave = (key + 9) // 12 + 4
        return key_name, key_octave, remainder_in_cent

    def _peak_fft_to_pk_info(self, fft_data):
        """ extract peak from fft data """
        minimum_distance = int(self.analyzer_n_window / self.sample_rate * self.peak_analyze_coefficient)
        indexes = peakutils.indexes(fft_data, thres=self.peak_analyze_min_threshold, min_dist=minimum_distance)
        raw_peak_info = []
        for i in indexes:
            raw_peak_info.append([i, fft_data[i]])
        raw_peak_info.sort(key=lambda x: x[-1], reverse=True)
        return raw_peak_info

    def _peak_weighted_sum_fix_tuning_pk(self, log_fft_data, raw_peak_info, fft_data):
        """ use weighted sum to fix peak center frequency """
        tuning_range = self.analyzer_n_window / self.sample_rate * self.peak_tuning_coefficient
        half_tuning_range = int(tuning_range / 2)
        modified_peak_info = []
        for fft_position, _ in raw_peak_info:
            if fft_position > half_tuning_range:
                position_range = [fft_position - half_tuning_range, fft_position + half_tuning_range]
                around_peak_data = log_fft_data[position_range[0]:position_range[1] + 1]
                around_peak_data = np.array(around_peak_data)
                around_peak_data /= np.sum(around_peak_data)
                around_peak_position = np.array(range(position_range[0], position_range[1] + 1))
                position_weighted_sum = np.sum(around_peak_data * around_peak_position)
                raw_peak_power = fft_data[fft_position]
                modified_peak_info.append([position_weighted_sum, raw_peak_power])
        return modified_peak_info

    def _peak_show_peak_information(self, peak_info):
        for fft_position, peak_power in peak_info:
            frequency = self._fft_position_to_frequency(fft_position)
            key_name, key_octave, remainder_in_cent = self._frequency_to_music_pitch(frequency)
            if remainder_in_cent > 0:
                add_sign = '+'
            else:
                add_sign = ''
            print(
                f' | {round(frequency, 3)}Hz --> '
                f'{key_name}{key_octave}{add_sign}{round(remainder_in_cent, 2)}c ~~> '
                f'{round(peak_power * 100, 2)}%')

    def _prepare_audio_peak_info(self, starting_time):
        valid = self._check_analyze_duration(starting_time)
        if not valid:
            print(
                f'<!> starting time set false\n'
                f'<!> number should be `0`~ `{self._get_audio_time()[-1] - self.analyze_min_duration}`s')
        else:
            # get starting sample index
            starting_sample = int(starting_time * self.sample_rate)
            # get data for spectral
            audio_data_combine_channel = np.mean(self.data, axis=0)[
                                         starting_sample:starting_sample + self.analyzer_n_window]
            audio_data_separate_channel = [x[starting_sample:starting_sample + self.analyzer_n_window] for x in
                                           self.data]
            fft_data = [self._fft_data_transform_single(x)[0] for x in audio_data_separate_channel]
            fft_data_combine_channel = self._log_min_max_transform(
                self._fft_data_transform_single(audio_data_combine_channel)[0], log=False)
            log_fft_data_combine_channel = self._log_min_max_transform(
                self._fft_data_transform_single(audio_data_combine_channel)[0])
            if len(self.data) > 1:
                fft_data_multiple = [fft_data_combine_channel] + list(self._log_min_max_transform(fft_data, log=False))
                log_fft_data_multiple = [log_fft_data_combine_channel] + list(self._log_min_max_transform(fft_data))
            else:
                fft_data_multiple = [fft_data_combine_channel]
                log_fft_data_multiple = [log_fft_data_combine_channel]
            raw_peak_info_multiple = [self._peak_fft_to_pk_info(x) for x in log_fft_data_multiple]
            print(f'<*> peaks @ `{starting_time}s`...')
            for i in range(len(raw_peak_info_multiple)):
                if i == 0:
                    channel_name = '*'
                else:
                    channel_name = str(i)
                print(f'<+> channel ({channel_name}):')
                modified_peak_info = self._peak_weighted_sum_fix_tuning_pk(log_fft_data_multiple[i],
                                                                           raw_peak_info_multiple[i],
                                                                           fft_data_multiple[i])
                # show information
                self._peak_show_peak_information(modified_peak_info)
            print('<*> ...')

            # prepare ifft play
            self._ifft_audio_export(self._log_min_max_transform(fft_data, log=False))


class TerminalSeeAudio(WaveSpectral, SpiralAnalyzer, PianoAnalyzer, PianoRoll, PeakAnalyzer, Shell):
    """
    this class will plot audio similar to `Adobe Audition` with:
        * plot wave & spectral
        * play music
        * analyze spectral
    """

    def __init__(self):
        WaveSpectral.__init__(self)
        SpiralAnalyzer.__init__(self)
        PianoAnalyzer.__init__(self)
        PianoRoll.__init__(self)
        PeakAnalyzer.__init__(self)
        Shell.__init__(self)

    def main(self, in_path):
        """ main function """
        self._get_and_fix_input_path(in_path)
        # initialization
        self._initialization()
        self._initialize_temp()
        # prepare
        last_starting, last_ending = self._get_audio_time()
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
                # 1.3.2 plot last piano image
                elif command == '':
                    self._terminal_plot(self.piano_graphics_path)
                # 1.3.3 plot last piano image
                elif command == '#':
                    self._terminal_plot(self.piano_roll_graphics_path)
                # 1.3.4 two numbers: piano roll
                elif ' ' in command:
                    piano_inputs = command.split()
                    if len(piano_inputs) == 2 and self._is_float(piano_inputs[0]) and self._is_float(piano_inputs[1]):
                        piano_inputs = [float(x) for x in piano_inputs]
                        status = self._prepare_graph_piano_roll(piano_inputs[0], piano_inputs[1])
                        if status:
                            self._terminal_plot(self.piano_roll_graphics_path)
                    else:
                        print('<!> piano analyzer inputs unknown')
                else:
                    print('<!> `piano` inputs unknown')
                continue
            # 1.4 get tuning frequencies (`^`)
            elif input_.startswith('^'):
                # 1.4.1 number as staring time
                if self._is_float(command):
                    self._prepare_audio_peak_info(float(command))
                else:
                    print('<!> tuning frequency inputs unknown')
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
                                last_starting, last_ending = self._get_audio_time()
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
                last_starting, last_ending = self._get_audio_time()
            # 3.6 `h` to print help file
            elif input_ == 'h':
                self.print_help()
            # 3.7 `testing` to test experimental functions
            # TODO: test functions here
            elif input_ == 'test':
                print('<!> no experimental function now')
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
