import os
import shutil
import subprocess
import sys

from utils import *

sys.path.append(os.path.abspath(__file__))


class TerminalSeeAudio(WaveSpectral, SpiralAnalyzer, PianoAnalyzer, PianoRoll, PeakAnalyzer, PlayPitch, TuningAnalyzer,
                       ShellUtils, SourceAnalyzer, PhaseAnalyzer, StringsAnalyzer, TonnetzAnalyzer):
    """
    this class will show audio information in many aspects.
    """

    def __init__(self):
        super().__init__()
        # prepare
        self.last_starting = self.last_ending = None
        self.last_analyze_starting = self.last_analyze_ending = None

    def _main_get_temp_analyzer_path(self, analyzer_name):
        analyzer_name = analyzer_name.strip().replace(' ', '_')
        path = os.path.join(self.temp_folder, analyzer_name)
        return path

    def _main_reset_time(self):
        self.last_starting = 0
        self.last_ending = self._get_audio_time()

    def _main_analyzer_1_or_2_input(self, command, prepare_graph_function, prepare_video_function, analyzer_name,
                                    **kwargs):
        temp_analyzer_path = self._main_get_temp_analyzer_path(analyzer_name)
        # x.x.1 number as staring time
        if self._is_float(command):
            status = prepare_graph_function(float(command), temp_analyzer_path + '.png', **kwargs)
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
                    inputs[1],
                    temp_analyzer_path,
                    **kwargs)
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

    def _main_analyzer_2_input(self, command, _prepare_graph_function, analyzer_name, **kwargs):
        temp_analyzer_path = self._main_get_temp_analyzer_path(analyzer_name)
        # x.x.1 two numbers
        if ' ' in command:
            inputs = command.split()
            if len(inputs) == 2 and self._is_float(inputs[0]) and self._is_float(
                    inputs[1]):
                inputs = [float(x) for x in inputs]
                status = _prepare_graph_function(inputs[0], inputs[1], temp_analyzer_path, **kwargs)
                if status:
                    self._terminal_plot(temp_analyzer_path + '.png')
            else:
                print(f'<!> `{analyzer_name}` analyzer inputs unknown')
        # x.x.2 plot last image
        elif command == '':
            self._terminal_plot(temp_analyzer_path + '.png')
        else:
            print(f'<!> `{analyzer_name}` analyzer inputs unknown')

    def _main_python_script(self, command):
        command_success = False
        command_result = []
        # x.x.1 to evaluate
        if not command_success:
            try:
                return_string = eval(command)
                print(f'<*> {return_string}')
                command_success = True
                return
            except Exception as e:
                command_result.append(e)
        # x.x.2 to execute
        if not command_success:
            try:
                exec(command)
                self._initialization()
                self._prepare_graph_wave_spectral(self.last_starting, self.last_ending)
                print(f'<*> executed `{command}`')
                command_success = True
                return
            except Exception as e:
                command_result.append(e)
        if not command_success:
            print(f'<!> evaluate error message: `{command_result[0]}`')
            print(f'<!> execute error message: `{command_result[1]}`')
            return

    def _main_tuning_analyzer_script(self, command, analyzer_name):
        tuning_graphics_path = self._main_get_temp_analyzer_path(analyzer_name) + '.png'
        # x.x.1 number as staring time
        if self._is_float(command):
            self._prepare_audio_peak_info(float(command))
        # x.x.2 two numbers: starting time + frequency
        elif ' ' in command:
            command_split = command.split()
            if len(command_split) == 2:
                if self._is_float(command_split[0]):
                    start_timing = float(command_split[0])
                else:
                    print(f'<!> {analyzer_name} starting time error')
                    return
                status = self._prepare_graph_tuning(start_timing, command_split[1], tuning_graphics_path)
                if status:
                    self._terminal_plot(tuning_graphics_path)
                else:
                    print(f'<!> {analyzer_name} frequency error')
                    return
            else:
                print(f'<!> please check number of inputs')
        else:
            print(f'<!> {analyzer_name} inputs unknown')

    def _main_shell_script(self, input_split):
        # noinspection PyBroadException
        try:
            sh_output = subprocess.check_output(input_split[1].replace('\\s', '\\ '), shell=True,
                                                stderr=subprocess.STDOUT, timeout=self.sh_timeout)
            print(f'<*> {str(sh_output.decode("utf-8").strip())}')
        except Exception as e:
            print(f'<!> error message: `{e}`')

    def _main_spectral_wave_script(self, input_split):
        # 2.2.0 set time parameters for wave spectral plot
        if self._is_float(input_split[0]) and self._is_float(input_split[1]):
            self.last_starting, self.last_ending, valid = self._prepare_graph_wave_spectral(float(input_split[0]),
                                                                                            float(input_split[1]))
            if valid:
                self._terminal_plot(self.wave_spectral_graphics_path)
        # 2.2.1 set modes
        elif input_split[0] == 'm':
            known_mode = True
            prepare_wave = False
            if input_split[1] in ['fft', 'fbank']:
                self.ws_spectral_transform_y = input_split[1]
                prepare_wave = True
            elif input_split[1] in ['power', 'log']:
                self.ws_spectral_transform_v = input_split[1]
                prepare_wave = True
            elif input_split[1] in ['mono', 'stereo']:
                self.channel_type = input_split[1]
                self._initialize_audio()
                prepare_wave = True
            elif input_split[1] in ['spectral', 'entropy']:
                self.ws_spectral_mode = input_split[1]
                prepare_wave = True
            elif input_split[1] in ['color', 'nocolor']:
                if input_split[1] == 'color':
                    self.colorful_theme = True
                else:
                    self.colorful_theme = False
            else:
                known_mode = False
            if known_mode:
                if prepare_wave:
                    # recalculating
                    self._prepare_graph_wave_spectral(self.last_starting, self.last_ending)
                print(f'<+> mode `{input_split[1]}` set')
            else:
                print(
                    f'<?> mode `{input_split[1]}` unknown\n<!> please read help')
        # 2.2.2 set sample rate
        elif input_split[0] == 'sr':
            if self._is_int(input_split[1]):
                if int(input_split[1]) >= self.min_sample_rate:
                    self.sample_rate = int(input_split[1])
                    self._initialization()
                    # recalculating
                    self._prepare_graph_wave_spectral(self.last_starting, self.last_ending)
                    print(f'<+> sample rate `{input_split[1]}` set')
                else:
                    print(f'<!> sample rate `{input_split[1]}` (< {self.min_sample_rate}) too low')
            else:
                print(f'<!> sample rate `{input_split[1]}` unknown')
        # 2.2.3 switch file to open
        elif input_split[0] == 'o':
            try_input_path = self._get_try_path(input_split[1]).replace('\\s', ' ')
            if os.path.exists(try_input_path):
                if self.input == try_input_path:
                    print('<!> same file path')
                else:
                    self.input = os.path.abspath(try_input_path)
                    self._initialize_audio()
                    print('<+> file path changed')
                    # reset time
                    self._main_reset_time()
            else:
                print(f'<!> file path `{try_input_path}` does not exist')
        # 2.2.4 ear nonlinear model transform
        elif input_split[0] == 'nonlinear':
            if self._is_float(input_split[1]) and float(input_split[1]) >= 0:
                self.ear_nonlinear = float(input_split[1])
                self._ws_initial_or_restore_running(starting_time=self.last_starting, ending_time=self.last_ending,
                                                    plot=False)
                print(f'<+> `ear/cochlear nonlinear parameter` is set to `{self.ear_nonlinear}`')
            else:
                print('<!> `ear/cochlear nonlinear parameter` should be float and `>0`')
        # 2.2.5 set video frame rate
        elif input_split[0] == 'fr':
            if self._is_int(input_split[1]):
                if int(input_split[1]) >= self.min_frame_rate:
                    self.analyze_video_frame_rate = int(input_split[1])
                    print(f'<+> video frame rate `{input_split[1]}` set')
                else:
                    print(f'<!> video frame rate `{input_split[1]}` (< {self.min_frame_rate}) too low')
            else:
                print(f'<!> video frame rate `{input_split[1]}` unknown')
        else:
            print('<!> two inputs case unknown')

    def _main_multiple_input_script(self, input_):
        to_continue = True
        # 1.1 `=` for advanced script
        # watch: space ` ` will be replaced by `\s`
        if input_.startswith('='):
            command = input_[1:].strip()
            self._main_python_script(command)
        # 1.2 get spiral (`@`) analyzer
        elif input_.startswith('@'):
            command = input_[1:].strip()
            self._main_analyzer_1_or_2_input(command, self._prepare_graph_spiral, self._prepare_video_spiral,
                                             'spiral')
        # 1.3* get piano roll chroma-gram (`##=`) analyzer
        elif input_.startswith('##='):
            command = input_[3:].strip()
            self._main_analyzer_2_input(command, self._prepare_graph_piano_roll, 'piano roll chroma', chroma=True)
        # 1.3 get piano roll (`##`) analyzer
        elif input_.startswith('##'):
            command = input_[2:].strip()
            self._main_analyzer_2_input(command, self._prepare_graph_piano_roll, 'piano roll')
        # 1.4* get piano chroma (`#=`) analyzer
        elif input_.startswith('#='):
            command = input_[2:].strip()
            self._main_analyzer_1_or_2_input(command, self._prepare_graph_piano, self._prepare_video_piano,
                                             'piano chroma', chroma=True)
        # 1.4 get piano (`#`) analyzer
        elif input_.startswith('#'):
            command = input_[1:].strip()
            self._main_analyzer_1_or_2_input(command, self._prepare_graph_piano, self._prepare_video_piano, 'piano')
        # 1.5 get tuning analyzer (`^`)
        elif input_.startswith('^'):
            command = input_[1:].strip()
            self._main_tuning_analyzer_script(command, 'tuning')
        # 1.6 play frequency or music notes
        elif input_.startswith('>'):
            command = input_[1:].strip()
            to_continue = self._pitch_export_wave_frequency(command)
            if to_continue:
                self._terminal_play(0, 1, self.pitch_audio_path)
        # 1.7 get source analyzer
        elif input_.startswith('*-*'):
            command = input_[3:].strip()
            self._main_analyzer_1_or_2_input(command, self._prepare_graph_source, self._prepare_video_source,
                                             'source stellar')
        # 1.8 get source angle analyzer
        elif input_.startswith('*<'):
            command = input_[2:].strip()
            self._main_analyzer_1_or_2_input(command, self._prepare_graph_source_angle,
                                             self._prepare_video_source_angle, 'source angle')
        # 1.9 get phase analyzer
        elif input_.startswith('*%'):
            command = input_[2:].strip()
            self._main_analyzer_1_or_2_input(command, self._prepare_graph_phase, self._prepare_video_phase, 'phase')
        # 1.10 get strings analyzer
        elif input_.startswith('|'):
            command = input_[1:].strip()
            self._main_analyzer_1_or_2_input(command, self._prepare_graph_strings, self._prepare_video_string,
                                             'strings')
        # 1.11 get tonnetz analyzer
        elif input_.startswith('&'):
            command = input_[1:].strip()
            self._main_analyzer_1_or_2_input(command, self._prepare_graph_tonnetz, self._prepare_video_tonnetz,
                                             'tonnetz')
        else:
            to_continue = False
        return to_continue

    def _main_contain_space_input_script(self, input_):
        # 2.0 prepare
        space_idx = input_.find(' ')
        input_split = [input_[:space_idx].strip(), input_[space_idx + 1:].strip()]
        # 2.1 shell command
        if input_split[0] == 'sh':
            self._main_shell_script(input_split)
            return

        # 2.2 two input functions
        if ' ' not in input_split[1] or self._path_input_check(input_split[1]):
            self._main_spectral_wave_script(input_split)
            return
        else:
            print('<!> please check number of input')
            return

    def _main_single_input_script(self, input_):
        to_break = False
        # 3.1 `p` to play last audio
        if input_ == 'p':
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
            shutil.rmtree(self.temp_folder, ignore_errors=True)
            parent_folder = os.path.dirname(self.temp_folder)
            if os.path.exists(parent_folder) and len(os.listdir(parent_folder)) == 0:
                shutil.rmtree(parent_folder)
            to_break = True
        # 3.4' `q!` to quit program but do not remove temp folder
        elif input_ == 'q!':
            to_break = True
        # 3.5 `r` to reset all
        elif input_ == 'r':
            print('<!> reset all')
            self._ws_initial_or_restore_running()
            # reset time
            self._main_reset_time()
        # 3.6 `h` to print help file
        elif input_ == 'h':
            self.print_help()
        # 3.7 `testing` to test experimental functions
        # TODO: test functions here
        elif input_ == 'test':
            print('<!> no experimental function now')
        else:
            print('<!> unknown command')
        return to_break

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
        self._ws_initial_or_restore_running()
        # loop to get inputs
        while True:
            print('-' * 50)
            input_ = input('</> ').strip()

            # 1. multiple input function (calculation)
            to_continue = self._main_multiple_input_script(input_)
            if to_continue:
                continue

            # 2. contain space case
            if ' ' in input_:
                self._main_contain_space_input_script(input_)
                continue

            # 3. single input
            to_break = self._main_single_input_script(input_)
            if to_break:
                break
            else:
                continue


if __name__ == '__main__':
    tsa = TerminalSeeAudio()

    # demo mode
    if len(sys.argv) == 1:
        tsa.main(None)
    # argument error
    elif len(sys.argv) > 2:
        print('<!> argument error, please check number of arguments')
    # default mode
    else:
        input_path = sys.argv[1]
        if input_path in ['-h', '--help']:
            TerminalSeeAudio().print_help()
        elif not os.path.exists(input_path):
            print(f'<!> path [{input_path}] does not exist!')
        else:
            tsa.main(input_path)
