import colorsys
import gc
import os
import subprocess

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


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

        # resolution of frequency (y) dimension
        self.n_window = 1024
        # analyzer time
        self.analyzer_time = 0.512
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

        # video generation for analyzers
        # get dynamic max log fft values
        self.analyze_log_fft_max_value = 0

        # plot & play command (path will be replaced by `{}`)
        self.plot_command = 'tiv {}'
        self.play_command = 'play {}'
        self.video_command = 'play {} > /dev/null 2>&1 & timg {}'

        # color & themes
        self.a_pitch_color = 'red'

        # graphics modes
        self.graphics_modes = ['fft', 'fbank', 'power', 'log', 'mono', 'stereo', 'phase', 'spectral']

        # plot mode
        plt.style.use('dark_background')

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
        self.analyze_n_window = int(self.analyzer_time * self.sample_rate)
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
        self.graphics_path = os.path.join(self.temp_folder, 'wave_spectral.png')
        self.spiral_graphics_path = os.path.join(self.temp_folder, 'spiral.png')
        self.piano_graphics_path = os.path.join(self.temp_folder, 'piano.png')
        self.piano_roll_graphics_path = os.path.join(self.temp_folder, 'piano_roll.png')
        self.tuning_graphics_path = os.path.join(self.temp_folder, 'tuning.png')
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

    @staticmethod
    def _calc_sp(audio, n_window, n_overlap, angle=False):
        """
        Calculate spectrogram or angle.
        :param audio: list(float): audio data
        :return: list(list(float)): the spectral data or angles
        """
        ham_win = np.hamming(n_window)
        if angle:
            [_, _, sp] = signal.spectral.spectrogram(
                audio,
                window=ham_win,
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
                window=ham_win,
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
            return False
        if end - start > self.play_max_duration:
            print(f'<!> audio too long for {end - start}s')
            while True:
                answer = input('</> do you wish to play [y/n]: ')
                if answer == 'y':
                    break
                elif answer == 'n':
                    return False
                else:
                    print('<!> please type `y` or `n`')
                    continue
        return True

    def _terminal_play(self, start, end, path):
        """ play in terminal function """
        status = self._check_audio_terminal_play(start, end, path)
        if status:
            command = self.play_command.format(path)
            # noinspection PyBroadException
            try:
                subprocess.call(command, shell=True)
            except Exception:
                print(f'<!> please fix problem:\n<?> {command}')

    def _terminal_video(self, start, end, audio_path, video_path):
        status = self._check_audio_terminal_play(start, end, audio_path)
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
            print('<!> temp image cannot find')
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
    def _set_1to1_ratio_figure():
        plt.gca().set_aspect(1)
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
        if np.max(array) != 0:
            if not dynamic_max_value:
                array /= np.max(np.abs(array))
            else:
                self.analyze_fft_max_value = max(self.analyze_fft_max_value, np.max(array))
                array /= self.analyze_fft_max_value
        return array

    def _phase_mode_check(self):
        if len(self.data) != 2:
            if self.spectral_phase_mode == 'phase':
                print('<!> `phase` mode cannot be set since `channel_number!=2`\n'
                      '<!> revert to `spectral` mode')
            self.spectral_phase_mode = 'spectral'
        if self.spectral_phase_mode == 'phase':
            phase = True
        elif self.spectral_phase_mode == 'spectral':
            phase = False
        else:
            raise ValueError(f'`spectral/phase` mode [{self.spectral_phase_mode}] unrecognized')
        return phase

    @staticmethod
    def _hsb_to_rgb(h, s, b):
        rgb = colorsys.hsv_to_rgb(h, s, b)
        rgb_color = '#' + ''.join(['{:02x}'.format(int(x * 255)) for x in rgb])
        return rgb_color

    @staticmethod
    def _convert_folder_path(file_path):
        """ convert to folder path and make folder """
        folder_path = os.path.splitext(file_path)[0]
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
