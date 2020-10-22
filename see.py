import argparse
import os
import shutil
import subprocess

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile as sf


class TerminalSeeAudio(object):
    """
    this class will plot audio similar to `Adobe Audition` with wave & spectral plot
    """

    def __init__(self, ops):
        # io parameters
        self.input = os.path.abspath(ops.input)
        self.temp_folder = os.path.abspath(ops.temp_folder)
        # spectral mode
        self.spectral_mode = ops.mode

        # define file paths
        self.graphics_path = os.path.join(self.temp_folder, 'wave_spectral.png')
        self.audio_part_path = os.path.join(self.temp_folder, 'audio.wav')

        # system parameters
        self.sample_rate = ops.sample_rate
        self.figure_size = (12, 4)
        self.line_width = .2
        self.dpi = 200
        self.spectral_power_transform_coefficient = 1 / 5
        self.graphics_ratio = 5
        # resolution of frequency dimension
        self.n_window = 1024
        # resolution of time dimension
        self.n_step = 128
        # max duration for audio to play (30s)
        self.max_duration = 30

        # colors & themes
        # self.axis_color = 'dimgray'
        self.axis_color = 'snow'
        self.spectral_color = 'magma'
        self.wave_color = 'mediumspringgreen'

        # import audio
        self.data = None
        self.time = None

        # initialization
        os.makedirs(self.temp_folder, exist_ok=True)
        self.initialize_audio()
        self.n_overlap = self.n_window - self.n_step

    def initialize_audio(self):
        """ read audio and parepare data """
        self.data, _ = librosa.load(self.input, sr=self.sample_rate, mono=True)
        self.time = range(len(self.data))
        self.time = [x / self.sample_rate for x in self.time]

    def _mel_filter(self, frame_pow):
        """
        convert spectral to mel-spectral
        --> from [https://zhuanlan.zhihu.com/p/130926693]
        :param frame_pow: spectral
        :return: mel spectral
        mel = 2595 * log10(1 + f/700)
        f = 700 * (10^(m/2595) - 1
        """
        fs = self.sample_rate
        n_filter = self.n_window
        nfft = self.n_window
        # lowest hearing frequency
        mel_min = 16
        mel_max = 2595 * np.log10(1 + fs / 2.0 / 700)
        mel_points = np.linspace(mel_min, mel_max, n_filter + 2)
        hz_points = 700 * (10 ** (mel_points / 2595.0) - 1)
        filter_edge = np.floor(hz_points * (nfft + 1) / fs)

        f_bank = np.zeros((n_filter, int(nfft / 2 + 1)))
        for m in range(1, 1 + n_filter):
            f_left = int(round(filter_edge[m - 1]))
            f_center = int(round(filter_edge[m]))
            # `+1` to avoid broken image
            f_right = int(round(filter_edge[m + 1])) + 1

            for k in range(f_left, f_center):
                f_bank[m - 1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                f_bank[m - 1, k] = (f_right - k) / (f_right - f_center)

        filter_banks = np.dot(frame_pow, f_bank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)

        filter_banks = np.power(filter_banks, self.spectral_power_transform_coefficient)
        filter_banks -= np.min(filter_banks)
        if np.max(filter_banks) == 0:
            raise ValueError('silence audio')
        filter_banks /= np.max(filter_banks)
        return filter_banks

    def _calc_sp(self, audio):
        """
        Calculate spectrogram.
        :param audio: list(float): audio data
        :return: list(list(float)), list(list(complex)): the real and complex part of clipped sound
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

    def data_prepare(self, starting_time, ending_time):
        """ prepare partition of audios """
        if starting_time < 0:
            starting_time = 0
            print('<!> reset starting time')
        if ending_time < 0:
            ending_time = len(self.data) / self.sample_rate
            print('<!> reset ending time')
        if starting_time >= ending_time:
            print('<!> starting time >= ending time ~~> reset all')
            starting_time = 0
            ending_time = len(self.data) / self.sample_rate
        # extract starting & ending sample
        starting_sample = max(int(self.sample_rate * starting_time), 0)
        ending_sample = min(int(self.sample_rate * ending_time), len(self.data))
        # make clip
        data_ = self.data[starting_sample:ending_sample]
        time_ = self.time[starting_sample:ending_sample]
        sf.write(self.audio_part_path, data_, self.sample_rate)
        return data_, time_, starting_time, ending_time

    def plot_wave(self, data_, time_, grid):
        """ plot wave """
        # plot audio wave
        fig1 = plt.subplot(grid[0, 0])
        fig1.plot(time_, data_, linewidth=self.line_width, color=self.wave_color)
        fig1.set_xlim(left=time_[0], right=time_[-1])
        fig1.axes.get_yaxis().set_ticks([])
        fig1.spines['left'].set_visible(False)
        fig1.spines['right'].set_visible(False)
        fig1.spines['top'].set_visible(False)
        fig1.spines['bottom'].set_color(self.axis_color)
        fig1.tick_params(axis='x', colors=self.axis_color)

    def plot_spectral(self, data, grid):
        """ plot spectral """
        # plot spectral
        spectral = self._calc_sp(data)
        if self.spectral_mode == 'fbank':
            spectral = self._mel_filter(spectral)
        elif self.spectral_mode == 'fft':
            pass
        else:
            raise ValueError(f'spectral mode [{self.spectral_mode}] unrecognized')
        spectral = np.flip(spectral, axis=1)
        spectral = np.transpose(spectral)

        # plot
        fig2 = plt.subplot(grid[1:self.graphics_ratio, 0])
        fig2.imshow(spectral, aspect='auto', cmap=self.spectral_color)
        fig2.axis('off')

    def prepare_graph_audio(self, starting_time, ending_time):
        """ prepare graphics and audio files """
        # default settings
        grid = plt.GridSpec(self.graphics_ratio, 1, wspace=0, hspace=0)
        plt.figure(figsize=self.figure_size)
        plt.style.use('dark_background')

        data_, time_, starting_time, ending_time = self.data_prepare(starting_time, ending_time)
        self.plot_spectral(data_, grid)
        self.plot_wave(data_, time_, grid)

        # save figure
        plt.savefig(self.graphics_path, dpi=self.dpi, bbox_inches='tight')
        return starting_time, ending_time

    def terminal_plot(self):
        """ plot in terminal function """
        command = ['timg', self.graphics_path]
        # noinspection PyBroadException
        try:
            subprocess.call(command)
        except Exception:
            print(f'<!> please fix problem:\n<?> {" ".join(command)}')

    def terminal_play(self, start, end):
        """ play in terminal function """
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
    def is_number(string):
        if string[0] == '-':
            string = string[1:]
        return string.isdigit()

    def main(self):
        """ main function """
        last_starting = 0
        last_ending = len(self.data) / self.sample_rate
        # first run
        self.prepare_graph_audio(last_starting, last_ending)
        self.terminal_plot()
        while True:
            print('-' * 50)
            input_ = input('</> ').strip()
            if ' ' in input_:
                input_split = input_.split()
                if len(input_split) != 2:
                    print('<!> please check number of input')
                    continue
                if self.is_number(input_split[0]) and self.is_number(input_split[1]):
                    last_starting, last_ending = self.prepare_graph_audio(float(input_split[0]), float(input_split[1]))
                    self.terminal_plot()
                else:
                    print('<!> input time should be numbers')
                    continue
            elif input_ == 'p':
                if os.path.exists(self.audio_part_path):
                    self.terminal_play(last_starting, last_ending)
                else:
                    print('<!> temp folder empty')
            elif input_ == '':
                self.terminal_plot()
            elif input_ == 'q':
                break
            else:
                print('<!> unknown command!')
                continue
        shutil.rmtree(self.temp_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='audio plot & play')
    parser.add_argument('--input', '-i', type=str, help='the input is or contains sound file(s)',
                        default='demo/june.ogg')
    parser.add_argument('--sample_rate', '-sr', type=int, help='the sample rate of output mix sound', default=8000)
    parser.add_argument('--temp_folder', '-tmp', type=str, help='the output temp directory for files', default='tmp')
    parser.add_argument('--mode', '-m', type=str, help='the mode of spectral to plot [fft/fbank]', default='fbank')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print(f'path [{args.input}] does not exist!')
    else:
        tsa = TerminalSeeAudio(args)
        tsa.main()
