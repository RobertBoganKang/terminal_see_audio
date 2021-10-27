import numpy as np
import peakutils

from utils.analyze_common import AnalyzeCommon


class PeakAnalyzer(AnalyzeCommon):
    def __init__(self):
        super().__init__()
        self.peak_analyze_coefficient = 80
        self.peak_tuning_coefficient = 40
        self.peak_analyze_min_threshold = 0.1

    def _peak_fft_to_pk_info(self, fft_data):
        """ extract peak from fft data """
        minimum_distance = int(self.analyze_n_window / self.sample_rate * self.peak_analyze_coefficient)
        indexes = peakutils.indexes(fft_data, thres=self.peak_analyze_min_threshold, min_dist=minimum_distance)
        raw_peak_info = []
        for i in indexes:
            raw_peak_info.append([i, fft_data[i]])
        raw_peak_info.sort(key=lambda x: x[-1], reverse=True)
        return raw_peak_info

    def _peak_weighted_sum_fix_tuning_pk(self, log_fft_data, raw_peak_info, fft_data):
        """ use weighted sum to fix peak center frequency """
        tuning_range = self.analyze_n_window / self.sample_rate * self.peak_tuning_coefficient
        half_tuning_range = int(tuning_range / 2)
        modified_peak_info = []
        for fft_position, _ in raw_peak_info:
            if fft_position > half_tuning_range:
                position_range = [fft_position - half_tuning_range, fft_position + half_tuning_range]
                around_peak_data = log_fft_data[position_range[0]:position_range[1] + 1]
                around_peak_data = np.array(around_peak_data)
                if np.sum(around_peak_data) != 0:
                    around_peak_data /= np.sum(around_peak_data)
                around_peak_position = np.array(range(position_range[0], position_range[1] + 1))
                position_weighted_sum = np.sum(around_peak_data * around_peak_position)
                raw_peak_power = fft_data[fft_position]
                modified_peak_info.append([fft_position, position_weighted_sum, raw_peak_power])
        return modified_peak_info

    @staticmethod
    def _add_sign(number):
        if number > 0:
            return '+'
        else:
            return ''

    def _peak_show_peak_information(self, peak_info):
        for fft_position, modified_position, peak_power in peak_info:
            frequency = self._fft_position_to_frequency(fft_position)
            modified_frequency = self._fft_position_to_frequency(modified_position)
            key_name, key_octave, remainder_in_cent = self._translate_frequency_to_music_note(frequency)
            key_name_1, key_octave_1, remainder_in_cent_1 = self._translate_frequency_to_music_note(
                modified_frequency)
            print(
                f' | {round(frequency, 3)}Hz '
                f'({key_name}{key_octave}{self._add_sign(remainder_in_cent)}{round(remainder_in_cent, 2)}c)'
                f' --> {round(modified_frequency, 3)}Hz '
                f'({key_name_1}{key_octave_1}{self._add_sign(remainder_in_cent_1)}{round(remainder_in_cent_1, 2)}c)'
                f' ~~> {round(peak_power * 100, 2)}%')

    def _prepare_audio_peak_info(self, starting_time):
        valid = self._check_analyze_duration(starting_time)
        if not valid:
            return False
        else:
            # get starting sample index
            starting_sample = int(starting_time * self.sample_rate)
            # get data for spectral
            audio_data_combine_channel = np.mean(self.data, axis=0)[
                                         starting_sample:starting_sample + self.analyze_n_window]
            audio_data_separate_channel = [x[starting_sample:starting_sample + self.analyze_n_window] for x in
                                           self.data]
            fft_data = [self._fft_data_transform_single(x)[0] for x in audio_data_separate_channel]
            fft_data_combine_channel = self._analyze_log_min_max_transform(
                self._fft_data_transform_single(audio_data_combine_channel)[0], log=False)
            log_fft_data_combine_channel = self._analyze_log_min_max_transform(
                self._fft_data_transform_single(audio_data_combine_channel)[0])
            if len(self.data) > 1:
                fft_data_multiple = [fft_data_combine_channel] + list(
                    self._analyze_log_min_max_transform(fft_data, log=False))
                log_fft_data_multiple = [log_fft_data_combine_channel] + list(
                    self._analyze_log_min_max_transform(fft_data))
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
                # calculate entropy
                entropy = self._get_shannon_entropy(fft_data_multiple[i])
                print(f'<*> channel ({channel_name}) with entropy `{round(entropy, 5)}`:')
                modified_peak_info = self._peak_weighted_sum_fix_tuning_pk(log_fft_data_multiple[i],
                                                                           raw_peak_info_multiple[i],
                                                                           fft_data_multiple[i])
                # show information
                self._peak_show_peak_information(modified_peak_info)
            print('<*> ...')

            # prepare ifft play
            self._ifft_audio_export(fft_data)
            return True
