import colorsys
import gc
import os
import shutil
import subprocess

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal


class Common(object):
    def __init__(self):
        # demo mode
        self.working_directory = os.path.dirname(os.path.dirname(__file__))
        self.demo_file = os.path.abspath(os.path.join(self.working_directory, 'demo/june.ogg'))

        # io parameters
        self.input = None
        self.temp_folder = os.path.join(self.working_directory, 'tmp')
        self.readme_path = os.path.join(self.working_directory, 'README.md')

        # constants
        self.golden_ratio = (np.sqrt(5) - 1) / 2
        self.a4_frequency = 440

        # figure parameters
        self.figure_minimum_alpha = 0.01

        # audio parameters
        self.sample_rate = 16000
        self.min_sample_rate = 1000
        # min/max hearing frequency
        self.min_hearing_frequency = 16
        self.max_hearing_frequency = 20000
        # `mono` or `stereo` (>1 channel)
        self.channel_type = 'stereo'
        # ear nonlinear model (cochlear will transform signal in non-linear form)
        self.ear_nonlinear = 0

        # resolution of frequency (y) dimension
        self.n_window = 1024
        # analyzer time
        self.analyze_time = 0.512
        # resolution of time (x) dimension
        self.n_min_step = 32
        self.n_spectral_max_width = 2048
        # max duration for audio to play (30s)
        self.play_max_duration = 30
        # spectral transform power coefficient for `power` mode
        self.spectral_power_transform_coefficient = 1 / 5
        # minimum hearing power for `log` mode
        self.min_hearing_power = 0.0005
        self.min_analyze_power = 0.1
        # spectral generation window method
        self.spectral_window_algorithm = 'hamming'

        # video generation for analyzers
        # get dynamic max log fft values
        self.analyze_log_fft_max_value = 0
        self.analyze_log_piano_key_max_value = 0
        # in steady state, how long can we reach that value
        self.analyze_log_piano_key_max_value_reach_time = 1

        # plot & play command (path will be replaced by `{}`)
        self.plot_command = 'tiv {}'
        self.play_command = 'play {}'
        self.video_command = 'play {} > /dev/null 2>&1 & timg {}'

        # color & themes
        self.a_pitch_color = 'red'

        # graphics modes
        self.graphics_modes = ['fft', 'fbank', 'power', 'log', 'mono', 'stereo', 'phase', 'spectral', 'entropy']

        # plot mode
        plt.style.use('dark_background')

    def _audio_nonlinear_transform(self, arrays, nonlinear):
        nonlinear_array = (nonlinear + 1) ** ((np.array(arrays) - 1) / 2)
        nonlinear_array = self._max_norm(nonlinear_array)
        return nonlinear_array

    def _check_audio_duration_valid(self, starting, ending, duration):
        """ check if greater than minimum duration """
        if ending - starting < self.min_duration:
            print(f'<!> {ending} - {starting} = {ending - starting} (< {duration}; minimum duration)\n'
                  f'<!> time duration too short')
            return False
        else:
            return True

    def _initialization(self):
        # demo mode message
        os.makedirs(self.temp_folder, exist_ok=True)
        self._initialize_audio()
        self.analyze_n_window = int(self.analyze_time * self.sample_rate)
        self.min_duration = self.n_window / self.sample_rate
        self.analyze_min_duration = self.analyze_n_window / self.sample_rate
        self._check_audio_duration()

    def _initialize_spectral(self, starting_time, ending_time):
        n_step = int((ending_time - starting_time) * self.sample_rate / self.n_spectral_max_width)
        n_step = max(self.n_min_step, n_step)
        self.n_overlap = self.n_window - n_step
        self.n_analyze_overlap = self.analyze_n_window - n_step

    def _initialize_temp(self):
        """ temp file path initialization """
        self.wave_spectral_graphics_path = os.path.join(self.temp_folder, 'wave_spectral.png')
        self.audio_part_path = os.path.join(self.temp_folder, 'audio.wav')
        self.ifft_audio_path = os.path.join(self.temp_folder, 'analyze_ifft.wav')
        self.pitch_audio_path = os.path.join(self.temp_folder, 'pitch.wav')

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
        if self.ear_nonlinear > 1e-3:
            self.data = self._audio_nonlinear_transform(self.data, self.ear_nonlinear)
        if len(self.data.shape) == 1:
            self.data = np.array([self.data])
        self.channel_num = len(self.data)
        self.time = range(len(self.data[0]))
        self.time = [x / self.sample_rate for x in self.time]

    def _get_audio_time(self):
        return len(self.data[0]) / self.sample_rate

    def _check_audio_duration(self):
        """ check if raw audio too short """
        if self._get_audio_time() < self.min_duration:
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

    def _export_audio(self, starting_time, ending_time, audio_part_path):
        """ export audio part """
        # extract starting & ending sample
        starting_sample = max(int(self.sample_rate * starting_time), 0)
        ending_sample = min(int(self.sample_rate * ending_time), len(self.data[0]))
        # make clip
        data_ = np.array([x[starting_sample:ending_sample] for x in self.data])
        data_transpose = np.transpose(data_)
        time_ = self.time[starting_sample:ending_sample]
        sf.write(audio_part_path, data_transpose, self.sample_rate)
        return data_, time_

    def _calc_sp(self, audio, n_window, n_overlap, angle=False):
        """
        Calculate spectrogram or angle.
        :param audio: list(float): audio data
        :return: list(list(float)): the spectral data or angles
        """
        try:
            win_algorithm = eval('signal.windows.' + self.spectral_window_algorithm)(n_window)
        except Exception:
            raise ValueError('`spectral_window_algorithm` error')
        if angle:
            [_, _, sp] = signal.spectral.spectrogram(
                audio,
                window=win_algorithm,
                nperseg=n_window,
                noverlap=n_overlap,
                detrend=False,
                return_onesided=True,
                mode='complex')
            sp = sp.T
            sp = sp.astype(np.complex64)
            x = np.abs(sp)
            y = np.angle(sp)
            return x, y
        else:
            [_, _, x] = signal.spectral.spectrogram(
                audio,
                window=win_algorithm,
                nperseg=n_window,
                noverlap=n_overlap,
                detrend=False,
                return_onesided=True,
                mode='magnitude')
            x = x.T
            x = x.astype(np.float64)
            return x

    def _check_audio_terminal_play(self, start, end, path):
        if not os.path.exists(path):
            print('<!> temp audio cannot find')
            return start, end, False
        need_check = True
        if start is None or end is None:
            print('<!> audio duration unknown')
            start = end = 0
        elif end - start > self.play_max_duration:
            print(f'<!> audio too long for {end - start}s')
        else:
            need_check = False
        if need_check:
            while True:
                answer = input('</> do you wish to play [y/n]: ')
                if answer == 'y':
                    break
                elif answer == 'n':
                    return start, end, False
                else:
                    print('<!> please type `y` or `n`')
                    continue
        return start, end, True

    def _terminal_play(self, start, end, path):
        """ play in terminal function """
        start, end, status = self._check_audio_terminal_play(start, end, path)
        if status:
            command = self.play_command.format(path)
            # noinspection PyBroadException
            try:
                subprocess.call(command, shell=True)
            except Exception:
                print(f'<!> please fix problem:\n<?> {command}')

    def _terminal_video(self, start, end, audio_path, video_path):
        if not os.path.exists(video_path):
            print('<!> temp video cannot find')
            return
        start, end, status = self._check_audio_terminal_play(start, end, audio_path)
        if status:
            command = self.video_command.format(audio_path, video_path)
            # noinspection PyBroadException
            try:
                subprocess.call(command, shell=True)
            except Exception:
                print(f'<!> please fix problem:\n<?> {command}')

    def _terminal_plot(self, path):
        """ plot in terminal function """
        if not os.path.exists(path):
            print(f'<!> temp image `{path}` cannot find')
            return
        command = self.plot_command.format(path)
        # noinspection PyBroadException
        try:
            subprocess.call(command, shell=True)
        except Exception:
            print(f'<!> please fix problem:\n<?> {command}')

    def _fix_input_starting_ending_time(self, starting_time, ending_time):
        test_ending = self._get_audio_time()
        if starting_time < 0:
            print(f'<!> reset starting time {starting_time}-->{0}')
            starting_time = 0
        if ending_time < 0 or ending_time > test_ending:
            print(f'<!> reset ending time {ending_time}-->{test_ending}')
            ending_time = test_ending
        if starting_time >= ending_time:
            print('<!> starting time >= ending time ~~> reset all')
            starting_time = 0
            ending_time = self._get_audio_time()
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

    @staticmethod
    def _set_1to1_ratio_figure(axis=False):
        plt.gca().set_aspect(1)
        if not axis:
            plt.axis('off')

    @staticmethod
    def _matplotlib_clear_memory(fig):
        plt.cla()
        plt.clf()
        plt.close(fig)
        gc.collect()

    def _max_norm(self, array, min_transform=False, dynamic_max_value=False):
        if min_transform:
            array -= np.min(array)
        if np.max(np.abs(array)) != 0:
            if not dynamic_max_value:
                array /= np.max(np.abs(array))
            else:
                self.analyze_log_fft_max_value = max(self.analyze_log_fft_max_value, np.max(array))
                if self.analyze_log_fft_max_value != 0:
                    array /= self.analyze_log_fft_max_value
        return array

    def _phase_mode_check(self):
        if len(self.data) != 2:
            if self.ws_spectral_mode == 'phase':
                print('<!> `phase` mode cannot be set since `channel_number!=2`\n'
                      '<!> revert to `spectral` mode')
            self.ws_spectral_mode = 'spectral'
        if self.ws_spectral_mode == 'phase':
            phase = True
        elif self.ws_spectral_mode in ['spectral', 'entropy']:
            phase = False
        else:
            raise ValueError(f'`spectral/phase` mode [{self.ws_spectral_mode}] unrecognized')
        return phase

    @staticmethod
    def _hsb_to_rgb(h, s, b):
        rgb = colorsys.hsv_to_rgb(h, s, b)
        rgb_color = '#' + ''.join(['{:02x}'.format(int(x * 255)) for x in rgb])
        return rgb_color

    @staticmethod
    def _convert_folder_path(file_path):
        """ convert to folder path and make folder """
        # remove folder
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
        os.makedirs(file_path)

    @staticmethod
    def _get_digits_number(number):
        """ return number of digits """
        return len(str(number))

    @staticmethod
    def _amplitude_ratio(t0, t1):
        t = [t0, t1]
        t.sort()
        if t[0] != 0:
            return t[0] / t[1]
        else:
            return 1

    def _amplitude_ratio_array(self, a0, a1):
        ratios = []
        for i in range(len(a0)):
            radio = self._amplitude_ratio(a0[i], a1[i])
            ratios.append(radio)
        return ratios

    @staticmethod
    def _get_shannon_entropy(array):
        # delete 0s
        delete_0_array = []
        for a in array:
            if a != 0:
                delete_0_array.append(a)
        # calculate shannon entropy
        if len(delete_0_array) != 0:
            delete_0_array = np.array(delete_0_array)
            return -np.sum(delete_0_array * np.log2(delete_0_array))
        else:
            return 0
