import numpy as np
import soundfile as sf

from utils.analyze_common import AnalyzeCommon


class PlayPitch(AnalyzeCommon):
    def __init__(self):
        super().__init__()
        self.pitch_play_time = 0.8

    def _pitch_generate_wave(self, frequency):
        """ get wave data for specific frequency """
        wave_data = []
        for i in range(int(self.pitch_play_time * self.sample_rate)):
            wave_data.append(np.sin(frequency * 2 * np.pi * (i / self.sample_rate)))
        return np.array(wave_data)

    def _pitch_export_wave_frequency(self, string):
        frequency = self._translate_string_to_frequency(string)
        if frequency is None:
            print(f'<!> pitch input `{string}` unknown or frequency too high')
            return False
        else:
            show_frequency = round(frequency, 3)
            if frequency > self.max_hearing_frequency:
                print(f'<!> higher than hearing frequency `{show_frequency}Hz` (>`{self.max_hearing_frequency}Hz`)')
                return False
            print(f'<*> `{show_frequency}Hz`')
            wave_data = self._pitch_generate_wave(frequency)
            sf.write(self.pitch_audio_path, wave_data, samplerate=self.sample_rate)
            return True
