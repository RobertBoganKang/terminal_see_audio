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


class TerminalSeeAudio(object):
    """
    this class will plot audio similar to `Adobe Audition` with:
        * plot wave & spectral
        * play music
    """

    def __init__(self, input_path):
        # demo mode
        self.demo_file = os.path.join(os.path.dirname(__file__), 'demo/june.ogg')

        # io parameters
        # if `None`: demo mode
        if input_path is None:
            if os.path.exists(self.demo_file):
                self.input = self.demo_file
            else:
                raise ValueError('demo file missing, example cannot be proceeded.')
        # regular mode
        else:
            self.input = os.path.abspath(input_path)
        self.temp_folder = os.path.join(os.path.dirname(__file__), 'tmp')
        self.readme_path = os.path.join(os.path.dirname(__file__), 'README.md')

        # system parameters
        self.sample_rate = 8000
        self.min_sample_rate = 1000
        self.figure_size = (12, 4)
        # spectral mode
        self.spectral_transform_y = 'fbank'
        self.spectral_transform_v = 'log'
        # line width parameters with `thin`, `thick`, `mode_switch_time`
        self.line_width_params = [.2, 1.2, 3]
        self.dpi = 200
        self.graphics_ratio = 5
        # resolution of frequency (y) dimension
        self.n_window = 1024
        # resolution of time (x) dimension
        self.n_step = 128
        # max duration for audio to play (30s)
        self.max_duration = 30
        # spectral transform power coefficient for `power` mode
        self.spectral_power_transform_coefficient = 1 / 5
        # minimum hearing power for `log` mode
        self.min_hearing_power = 0.0005
        # timeout for shell script
        self.sh_timeout = 10

        # colors & themes
        self.axis_color = 'white'
        self.spectral_color = 'magma'
        self.wave_color = 'mediumspringgreen'

        # initialization
        self.n_overlap = None
        self.min_duration = None
        self.data = []
        self.time = []

        # spectral modes
        self.spectral_modes = ['fft', 'fbank', 'power', 'log']

    def _initialization(self):
        # demo mode message
        if self.input == self.demo_file:
            print(f'<+> demo file `{self.demo_file}` will be tested')
        os.makedirs(self.temp_folder, exist_ok=True)
        self._initialize_audio()
        self.n_overlap = self.n_window - self.n_step
        self.min_duration = self.n_window / self.sample_rate
        self._check_audio_duration()
        # initialize path autocomplete
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self._path_autocomplete)

    def _initialize_temp(self):
        """ temp file path initialization """
        self.graphics_path = os.path.join(self.temp_folder, 'wave_spectral.png')
        self.audio_part_path = os.path.join(self.temp_folder, 'audio.wav')

    def _initialize_audio(self):
        """ read audio and prepare data """
        self.data, _ = librosa.load(self.input, sr=self.sample_rate, mono=True)
        self.time = range(len(self.data))
        self.time = [x / self.sample_rate for x in self.time]

    def _check_audio_duration(self):
        """ check if raw audio too short """
        if len(self.data) / self.sample_rate < self.min_duration:
            raise ValueError('audio too short; exit')

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
        mel_min = 16
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

    def _calc_sp(self, audio):
        """
        Calculate spectrogram.
        :param audio: list(float): audio data
        :return: list(list(float)): the spectral data
        """
        ham_win = np.hamming(self.n_window)
        [_, _, x] = signal.spectral.spectrogram(
            audio,
            window=ham_win,
            nperseg=self.n_window,
            noverlap=self.n_overlap,
            detrend=False,
            return_onesided=True,
            mode='magnitude')
        x = x.T
        x = x.astype(np.float64)
        return x

    def _data_prepare(self, starting_time, ending_time):
        """ prepare partition of audios """
        test_ending = len(self.data) / self.sample_rate
        if starting_time < 0:
            print(f'<!> reset starting time {starting_time}-->{0}')
            starting_time = 0
        if ending_time < 0 or ending_time > test_ending:
            print(f'<!> reset ending time {ending_time}-->{test_ending}')
            ending_time = test_ending
        if starting_time >= ending_time:
            print('<!> starting time >= ending time ~~> reset all')
            starting_time = 0
            ending_time = len(self.data) / self.sample_rate
        if not self._check_audio_duration_valid(starting_time, ending_time):
            return None, None, starting_time, ending_time, False
        # extract starting & ending sample
        starting_sample = max(int(self.sample_rate * starting_time), 0)
        ending_sample = min(int(self.sample_rate * ending_time), len(self.data))
        # make clip
        data_ = self.data[starting_sample:ending_sample]
        time_ = self.time[starting_sample:ending_sample]
        sf.write(self.audio_part_path, data_, self.sample_rate)
        return data_, time_, starting_time, ending_time, True

    def _plot_wave(self, data_, time_, grid):
        """ plot wave """
        # plot audio wave
        fig1 = plt.subplot(grid[0, 0])
        # create a function to define line width
        duration = time_[-1] - time_[0]
        if duration > self.line_width_params[2]:
            line_width = self.line_width_params[0]
        else:
            line_width = (self.line_width_params[1] - (self.line_width_params[1] - self.line_width_params[0]) /
                          self.line_width_params[-1] * duration)
        fig1.plot(time_, data_, linewidth=line_width, color=self.wave_color)
        fig1.set_xlim(left=time_[0], right=time_[-1])
        fig1.axes.get_yaxis().set_ticks([])
        fig1.spines['left'].set_visible(False)
        fig1.spines['right'].set_visible(False)
        fig1.spines['top'].set_visible(False)
        fig1.spines['bottom'].set_color(self.axis_color)
        fig1.tick_params(axis='x', colors=self.axis_color)

    def _plot_spectral(self, data, grid):
        """ plot spectral """
        # plot spectral
        spectral = self._calc_sp(data)
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
        fig2 = plt.subplot(grid[1:self.graphics_ratio, 0])
        fig2.imshow(spectral, aspect='auto', cmap=self.spectral_color)
        fig2.axis('off')

    def _prepare_graph_audio(self, starting_time, ending_time):
        """ prepare graphics and audio files """
        # default settings
        grid = plt.GridSpec(self.graphics_ratio, 1, wspace=0, hspace=0)
        plt.figure(figsize=self.figure_size)
        plt.style.use('dark_background')

        data_, time_, starting_time, ending_time, valid = self._data_prepare(starting_time, ending_time)
        if not valid:
            return starting_time, ending_time, False
        self._plot_spectral(data_, grid)
        self._plot_wave(data_, time_, grid)

        # save figure
        plt.savefig(self.graphics_path, dpi=self.dpi, bbox_inches='tight')
        return starting_time, ending_time, True

    def _terminal_plot(self):
        """ plot in terminal function """
        if not os.path.exists(self.graphics_path):
            print('<!> temp image cannot find')
            return
        command = ['timg', self.graphics_path]
        # noinspection PyBroadException
        try:
            subprocess.call(command)
        except Exception:
            print(f'<!> please fix problem:\n<?> {" ".join(command)}')

    def _terminal_play(self, start, end):
        """ play in terminal function """
        if not os.path.exists(self.audio_part_path):
            print('<!> temp audio cannot find')
            return
        if end - start > self.max_duration:
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
        command = ['play', self.audio_part_path]
        # noinspection PyBroadException
        try:
            subprocess.call(command)
        except Exception:
            print(f'<!> please fix problem:\n<?> {" ".join(command)}')

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
        self._prepare_graph_audio(0, len(self.data) / self.sample_rate)
        self._terminal_plot()

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
        return (glob.glob(text + '*') + [None])[state]

    def main(self):
        """ main function """
        # initialization
        self._initialization()
        self._initialize_temp()
        # prepare
        last_starting = 0
        last_ending = len(self.data) / self.sample_rate
        # 0. first run
        self._initial_or_restore_running()
        # loop to get inputs
        while True:
            print('-' * 50)
            input_ = input('</> ').strip()

            # 1. multiple input function (calculation)
            if input_.startswith('='):
                command_success = False
                command_result = []
                # 1.1 to evaluate
                if not command_success:
                    try:
                        return_string = eval(input_[1:])
                        print(f'<*> {return_string}')
                        command_success = True
                        continue
                    except Exception as e:
                        command_result.append(e)
                # 1.2 to execute
                if not command_success:
                    try:
                        exec(input_[1:])
                        self._initialization()
                        self._prepare_graph_audio(last_starting, last_ending)
                        print(f'<*> executed `{input_[1:]}`')
                        command_success = True
                        continue
                    except Exception as e:
                        command_result.append(e)
                # 1.* advanced script error
                if not command_success:
                    print(f'<!> evaluate error message: `{command_result[0]}`')
                    print(f'<!> execute error message: `{command_result[1]}`')
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
                        sh_output = subprocess.check_output(input_split[1], shell=True, stderr=subprocess.STDOUT,
                                                            timeout=self.sh_timeout)
                        print(f'<*> {sh_output}')
                    except Exception as e:
                        print(f'<!> error message: `{e}`')
                    continue

                # 2.2 two input functions
                if ' ' not in input_split[1] or self._path_input_check(input_split):
                    try_input = self._get_try_path(input_split)
                    # 2.2.0 set time parameters
                    if self._is_float(input_split[0]) and self._is_float(input_split[1]):
                        last_starting, last_ending, valid = self._prepare_graph_audio(float(input_split[0]),
                                                                                      float(input_split[1]))
                        if valid:
                            self._terminal_plot()
                    # 2.2.1 set modes
                    elif input_split[0] == 'm':
                        if input_split[1] in self.spectral_modes:
                            if input_split[1] in ['fft', 'fbank']:
                                self.spectral_transform_y = input_split[1]
                            elif input_split[1] in ['power', 'log']:
                                self.spectral_transform_v = input_split[1]
                            # recalculating
                            self._prepare_graph_audio(last_starting, last_ending)
                            print(f'<+> mode `{input_split[1]}` set')
                        else:
                            print(f'<?> mode `{input_split[1]}` unknown\n<!> modes are within {self.spectral_modes}')
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
                            self.input = try_input
                            self._initialize_audio()
                            print('<+> file path changed')
                            self._initial_or_restore_running()
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
                self._terminal_play(last_starting, last_ending)
            # 3.2 `` to show last image
            elif input_ == '':
                self._terminal_plot()
            # 3.3 `q` to quit program
            elif input_ == 'q':
                break
            # 3.4 `r` to reset starting and ending time
            elif input_ == 'r':
                print('<!> reset starting & ending time')
                self._initial_or_restore_running()
            # 3.5 `h` to print help file
            elif input_ == 'h':
                self.print_help()
            # 3.* single input case error
            else:
                print('<!> unknown command')
                continue
        # remove temp folder at quit
        shutil.rmtree(self.temp_folder)


if __name__ == '__main__':
    # demo mode
    if len(sys.argv) == 1:
        tsa = TerminalSeeAudio(None)
        tsa.main()
    # argument error
    elif len(sys.argv) > 2:
        print('argument error, please check number of arguments')
    # default mode
    else:
        in_path = sys.argv[1]
        if in_path in ['-h', '--help']:
            TerminalSeeAudio(None).print_help()
        elif not os.path.exists(in_path):
            print(f'path [{in_path}] does not exist!')
        else:
            tsa = TerminalSeeAudio(in_path)
            tsa.main()
