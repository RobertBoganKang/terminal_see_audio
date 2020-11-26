import matplotlib.pyplot as plt
import numpy as np

from utils.piano_common import PianoCommon


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

    def _piano_roll_indicator(self, ax):
        """ piano roll base """
        # plot cover & frame
        top_most, _ = self._piano_roll_generate_key_position(self.piano_key_range[0])
        bottom_most, _ = self._piano_roll_generate_key_position(self.piano_key_range[1] - 1)
        cover_x_positions = [0, 0, - self.piano_roll_cover_width, - self.piano_roll_cover_width]
        frame_x_positions = [0, 0, self.piano_roll_length, self.piano_roll_length]
        cover_y_positions = [top_most[0, 0], bottom_most[1, 0], bottom_most[1, 0], top_most[0, 0]]
        ax.fill(cover_x_positions, cover_y_positions, edgecolor=self.piano_roll_base_color,
                facecolor=self.piano_base_color,
                linewidth=self.piano_line_width, zorder=5, alpha=0.9)
        ax.fill(frame_x_positions, cover_y_positions, edgecolor=self.piano_roll_base_color,
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
            ax.fill(positions[:, 1], positions[:, 0], facecolor=key_color, edgecolor=self.piano_roll_base_color,
                    linewidth=self.piano_line_width, zorder=bw + 1)
            # plot piano roll base
            ax.fill(base_x_positions, base_y_positions, facecolor=roll_color, zorder=1, alpha=alpha)

            # plot grid
            ax.plot([base_x_positions[0], base_x_positions[1]], [base_y_positions[-1], base_y_positions[-1]],
                    c=self.piano_roll_base_color, linewidth=self.piano_line_width, alpha=0.5, zorder=4)
            # makeup edge line
            if k == self.piano_key_range[0]:
                ax.plot([base_x_positions[0], base_x_positions[1]], [base_y_positions[0], base_y_positions[0]],
                        c=self.piano_roll_base_color, linewidth=self.piano_line_width, alpha=0.5, zorder=4)

    def _piano_roll_generate_frequency_graph_single(self, ax, key_dict, step, all_step_number):
        # get key position
        step_size = self.piano_roll_length / all_step_number
        for k, v in key_dict.items():
            key_0, key_1 = self._piano_roll_key_to_location_range(k)
            x_positions = [step * step_size, (step + 1) * step_size, (step + 1) * step_size, step * step_size]
            y_positions = [key_0, key_0, key_1, key_1]
            freq_alpha = v ** self.piano_key_color_transform_power
            if freq_alpha > self.figure_minimum_alpha:
                ax.fill(x_positions, y_positions, facecolor=self.piano_roll_color, zorder=3, alpha=freq_alpha)

    def _prepare_graph_piano_roll(self, starting_time, ending_time):
        # fix time first
        starting_time, ending_time = self._fix_input_starting_ending_time(starting_time, ending_time)
        if ending_time - starting_time < self.analyze_min_duration:
            print('<!> audio too short to show piano roll')
            return False
        else:
            self._initialize_spectral(starting_time, ending_time)
            # prepare spectrum
            # extract starting & ending sample
            starting_sample = int(self.sample_rate * starting_time)
            ending_sample = int(self.sample_rate * ending_time)
            # to `mono` for piano roll
            data_ = np.mean(self.data, axis=0)
            log_fft_data = self._analyze_log_min_max_transform(
                self._calc_sp(data_[starting_sample:ending_sample], self.analyze_n_window, self.n_analyze_overlap))
            # plot
            fig = plt.figure(figsize=self.piano_roll_figure_size)
            ax = plt.subplot(111)
            ax.set_ylim([self._piano_roll_generate_key_position(self.piano_key_range[0])[0][0, 0] - 0.5,
                         self._piano_roll_generate_key_position(self.piano_key_range[1] - 1)[0][1, 0] + 0.5])
            ax.set_xlim([-self.piano_roll_key_length - 0.2, self.piano_roll_length + 0.2])
            # plot piano base
            self._piano_roll_indicator(ax)
            # plot piano roll
            for i, data in enumerate(log_fft_data):
                key_dict, raw_key, key_fft = self._piano_key_spectral_data(data)
                self._piano_roll_generate_frequency_graph_single(ax, key_dict, i, len(log_fft_data))
            # set plot ratio
            plt.gca().set_aspect(1)
            plt.axis('off')
            fig.savefig(self.piano_roll_graphics_path, dpi=self.piano_roll_dpi, bbox_inches='tight')
            self._matplotlib_clear_memory(fig)
            return True
