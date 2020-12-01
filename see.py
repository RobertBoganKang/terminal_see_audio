import os
import shutil
import subprocess
import sys

from utils import *

sys.path.append(os.path.abspath(__file__))


class TerminalSeeAudio(WaveSpectral, SpiralAnalyzer, PianoAnalyzer, PianoRoll, PeakAnalyzer, PlayPitch, TuningAnalyzer,
                       ShellUtils, SourceAnalyzer, PhaseAnalyzer):
    """
    this class will show audio information in many aspects.
    """

    def __init__(self):
        WaveSpectral.__init__(self)
        SpiralAnalyzer.__init__(self)
        PianoAnalyzer.__init__(self)
        PianoRoll.__init__(self)
        PeakAnalyzer.__init__(self)
        PlayPitch.__init__(self)
        TuningAnalyzer.__init__(self)
        ShellUtils.__init__(self)
        SourceAnalyzer.__init__(self)
        PhaseAnalyzer.__init__(self)

        # prepare
        self.last_starting = self.last_ending = None
        self.last_analyze_starting = self.last_analyze_ending = None

    def _main_analyzer_1_or_2_input(self, command, prepare_graph_function, prepare_video_function, temp_analyzer_path,
                                    analyzer_name):
        # x.x.1 number as staring time
        if self._is_float(command):
            status = prepare_graph_function(float(command))
            if status:
                self._terminal_plot(temp_analyzer_path + '.png')
        # x.x.2 plot last image
        elif command == '':
            self._terminal_plot(temp_analyzer_path + '.png')
        # x.x.3 two numbers
        elif ' ' in command:
            inputs = command.split()
            if len(inputs) == 2 and self._is_float(inputs[0]) and self._is_float(inputs[1]):
                inputs = [float(x) for x in inputs]
                self.last_analyze_starting, self.last_analyze_ending, status = prepare_video_function(
                    inputs[0],
                    inputs[1])
                if status:
                    self._terminal_video(self.last_analyze_starting, self.last_analyze_ending,
                                         temp_analyzer_path + '.wav',
                                         temp_analyzer_path + '.mp4')
            else:
                print(f'<!> `{analyzer_name}` analyzer inputs unknown')
        # x.x.4 plot last piano video
        elif command == '*':
            self._terminal_video(self.last_analyze_starting, self.last_analyze_ending,
                                 temp_analyzer_path + '.wav',
                                 temp_analyzer_path + '.mp4')
        else:
            print(f'<!> `{analyzer_name}` analyzer inputs unknown')

    def _main_analyzer_2_input(self, command, _prepare_graph_function, temp_analyzer_path, analyzer_name):
        # x.x.1 two numbers
        if ' ' in command:
            inputs = command.split()
            if len(inputs) == 2 and self._is_float(inputs[0]) and self._is_float(
                    inputs[1]):
                inputs = [float(x) for x in inputs]
                status = _prepare_graph_function(inputs[0], inputs[1])
                if status:
                    self._terminal_plot(temp_analyzer_path)
            else:
                print(f'<!> `{analyzer_name}` analyzer inputs unknown')
        # x.x.2 plot last image
        elif command == '':
            self._terminal_plot(temp_analyzer_path)
        else:
            print(f'<!> `{analyzer_name}` analyzer inputs unknown')

    def main(self, in_path):
        """ main function """
        self._get_and_fix_input_path(in_path)
        # initialization
        self._initialization()
        self._initialize_temp()
        # prepare
        self.last_starting = 0
        self.last_ending = self._get_audio_time()
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
                        self._prepare_graph_audio(self.last_starting, self.last_ending)
                        print(f'<*> executed `{command}`')
                        command_success = True
                        continue
                    except Exception as e:
                        command_result.append(e)
                if not command_success:
                    print(f'<!> evaluate error message: `{command_result[0]}`')
                    print(f'<!> execute error message: `{command_result[1]}`')
                    continue
            # 1.2 get spiral (`@`) analyzer
            elif input_.startswith('@'):
                self._main_analyzer_1_or_2_input(command, self._prepare_graph_spiral, self._prepare_video_spiral,
                                                 self.spiral_analyzer_path, 'spiral')
                continue
            # 1.3 get piano roll (`##`) analyzer
            elif input_.startswith('##'):
                command = command[1:].strip()
                self._main_analyzer_2_input(command, self._prepare_graph_piano_roll, self.piano_roll_graphics_path,
                                            'piano roll')
                continue
            # 1.4 get piano (`#`) analyzer
            elif input_.startswith('#'):
                self._main_analyzer_1_or_2_input(command, self._prepare_graph_piano, self._prepare_video_piano,
                                                 self.piano_analyzer_path, 'piano')
                continue
            # 1.5 get tuning analyzer (`^`)
            elif input_.startswith('^'):
                # 1.5.1 number as staring time
                if self._is_float(command):
                    self._prepare_audio_peak_info(float(command))
                # 1.5.2 two numbers: starting time + frequency
                elif ' ' in command:
                    command_split = command.split()
                    if len(command_split) == 2:
                        if self._is_float(command_split[0]):
                            start_timing = float(command_split[0])
                        else:
                            print('<!> tuning starting time error')
                            continue
                        status = self._prepare_graph_tuning(start_timing, command_split[1])
                        if status:
                            self._terminal_plot(self.tuning_graphics_path)
                        else:
                            print('<!> tuning frequency error')
                            continue
                    else:
                        print('<!> please check number of tuning input')
                else:
                    print('<!> tuning inputs unknown')
                continue
            # 1.6 play frequency or music notes
            elif input_.startswith('>'):
                status = self._pitch_export_wave_frequency(command)
                if status:
                    self._terminal_play(0, 1, self.pitch_audio_path)
                continue
            # 1.7 get source analyzer
            elif input_.startswith('*-*'):
                command = command[2:].strip()
                self._main_analyzer_1_or_2_input(command, self._prepare_graph_source, self._prepare_video_source,
                                                 self.source_analyzer_path, 'source stellar map')
                continue
            # 1.8 get source angle analyzer
            elif input_.startswith('*<'):
                command = command[1:].strip()
                self._main_analyzer_1_or_2_input(command, self._prepare_graph_source_angle,
                                                 self._prepare_video_source_angle,
                                                 self.source_angle_analyzer_path, 'source angle')
                continue
            # 1.9 get phase analyzer
            elif input_.startswith('*%'):
                command = command[1:].strip()
                self._main_analyzer_1_or_2_input(command, self._prepare_graph_phase, self._prepare_video_phase,
                                                 self.source_analyzer_path, 'phase')
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
                        self.last_starting, self.last_ending, valid = self._prepare_graph_audio(float(input_split[0]),
                                                                                                float(input_split[1]))
                        if valid:
                            self._terminal_plot(self.wave_spectral_graphics_path)
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
                            elif input_split[1] in ['spectral', 'phase']:
                                self.spectral_phase_mode = input_split[1]
                            # recalculating
                            self._prepare_graph_audio(self.last_starting, self.last_ending)
                            print(f'<+> mode `{input_split[1]}` set')
                        else:
                            print(
                                f'<?> mode `{input_split[1]}` unknown\n<!> modes are within {self.graphics_modes}')
                    # 2.2.2 set sample rate
                    elif input_split[0] == 'sr':
                        if self._is_int(input_split[1]):
                            if int(input_split[1]) >= self.min_sample_rate:
                                self.sample_rate = int(input_split[1])
                                self._initialization()
                                # recalculating
                                self._prepare_graph_audio(self.last_starting, self.last_ending)
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
                                self.last_starting = 0
                                self.last_ending = self._get_audio_time()
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
                    else:
                        print('<!> two inputs case unknown')
                    continue
                else:
                    print('<!> please check number of input')
                    continue

            # 3. single input
            # 3.1 `p` to play last audio
            elif input_ == 'p':
                self._terminal_play(self.last_starting, self.last_ending, self.audio_part_path)
            # 3.2 `p*` to play last short period audio analyzed by spectral analyzer
            elif input_ == 'pp':
                self._terminal_play(0, 0.1, self.ifft_audio_path)
            # 3.3 `` to show last image
            elif input_ == '':
                self._terminal_plot(self.wave_spectral_graphics_path)
            # 3.4 `q` to quit program
            elif input_ == 'q':
                # remove temp folder at quit
                shutil.rmtree(self.temp_folder)
                break
            # 3.4' `q!` to quit program but do not remove temp folder
            elif input_ == 'q!':
                break
            # 3.5 `r` to reset all
            elif input_ == 'r':
                print('<!> reset all')
                self._initial_or_restore_running()
                # reset time
                self.last_starting = 0
                self.last_ending = self._get_audio_time()
            # 3.6 `h` to print help file
            elif input_ == 'h':
                self.print_help()
            # 3.7 `testing` to test experimental functions
            # TODO: test functions here
            elif input_ == 'test':
                print('<!> no experimental function now')
            else:
                print('<!> unknown command')
                continue


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
