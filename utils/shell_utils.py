import glob
import os
import readline

from utils.common import Common


class ShellUtils(Common):
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
