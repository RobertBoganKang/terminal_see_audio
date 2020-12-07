import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Arc

from utils.flower_common import FlowerCommon


class SourceAnalyzer(FlowerCommon):
    def __init__(self):
        super().__init__()
        # figure space limit
        self.source_figure_space_limit = 2

        self.source_line_width = 1.5
        self.source_figure_size = (12, 12)
        self.source_axis_color = 'snow'
        self.source_human_color = '#222'

    @staticmethod
    def _source_x_position(d, e, rt):
        if rt != 1:
            return (d ** 2 * (1 + rt)) / (4 * e * (-1 + rt))
        else:
            return 0

    @staticmethod
    def _source_y_square_position(d, e, rt):
        if rt != 1:
            return -((d ** 2 - 4 * e ** 2) * (-4 * e ** 2 * (-1 + rt) ** 2 + d ** 2 * (1 + rt) ** 2)) / (
                    16 * e ** 2 * (-1 + rt) ** 2)
        else:
            return 0

    @staticmethod
    def _source_xy_fake_position(e, rt, tana):
        sqrt_term = e ** 2 * (4 * rt ** 2 - tana ** 2 * (-1 + rt ** 2) ** 2)
        if sqrt_term > 0:
            x_position = (e + e * rt ** 2 + np.sqrt(sqrt_term)) / ((1 + tana ** 2) * (-1 + rt ** 2))
            y_position = ((tana * (e + e * rt ** 2 + np.sqrt(sqrt_term))) / ((1 + tana ** 2) * (-1 + rt ** 2)))
            return x_position, y_position
        else:
            return None, None

    def _source_distance_difference_from_phase(self, p0, p1, frequency, cycle=0):
        """ distance difference is negative phase difference """
        wave_length = self._analyze_sound_speed() / frequency
        phase_portion = (p0 - p1) / (2 * np.pi)
        phase_portion = (phase_portion + 0.5) % 1 - 0.5 + cycle
        return phase_portion * wave_length

    @staticmethod
    def _source_angle_norm(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _source_power(a_x, d_x):
        """
        power model:
        --------------------------------------------------
        aX: amplitude of sine wave at position X;
        pX: power of sine wave at position X;
        pU: power of sine wave at position U where at unit ball;
        --------------------------------------------------
        {4} power function:
            power of sine wave:
                MATHEMATICA:> 
                    Integrate[(aX*Sin[x])^2, {x, 0, 2 Pi}]/(2 Pi)
                {4.1} --> pX=aX^2/2;
            from {2.3}, {4.1}:
                {4.2} --> pX/pU=(dX/dU)^(-2);
                {4.3} --> dU=1;
                {4.4} --> pU=(1/2)*aX^2*dX^2;
        """
        return (1 / 2) * a_x ** 2 * d_x ** 2

    def _source_asymptote_angle(self, e, d, f, v, phase_diff):
        """
        {5} asymptote of hyperbola
        --------------------------------------------------
        a & b: default parameters in hyperbola
        c: eccentricity of hyperbola
        `alpha`: angle of approximate sound source position in rad
        function:
            x^2/a^2 - y^2/b^2=1
            c^2=a^2 + b^2
            e=c
            a=d/2
        from functions above:
            {5.1} --> y/x=(+-)*(4e^2/d^2-1)^(1/2)
        then:
            {5.2} --> alpha=arc_tan(y/x)
        --------------------------------------------------
        {6} fake angle
        --------------------------------------------------
        margin angle `phi_margin` is when y/x = 0
            {6.1} --> phi_margin=2*e*f/v
        set the angle evenly on negative region for excessive phase
        """
        if d != 0:
            y_div_x_pow_2 = 4 * e ** 2 / d ** 2 - 1
            if y_div_x_pow_2 >= 0:
                sign = np.sign(d)
                y_div_x = sign * np.sqrt(y_div_x_pow_2)
                angle = np.arctan(y_div_x)
                angle %= np.pi
                return True, angle
            else:
                phase_diff = self._source_angle_norm(phase_diff)
                sign = -np.sign(phase_diff)
                margin_phase = 2 * e * f / v
                # rotate quarter circle
                fake_angle = sign * (
                        np.pi / 2 * (abs(phase_diff) - margin_phase) / (np.pi - margin_phase) + np.pi / 2) + np.pi / 2
                return False, fake_angle
        else:
            return True, np.pi / 2

    def _source_position_transform(self, arrays, log_arrays, phases):
        """
        position model:
        --------------------------------------------------
              (xt,yt)
              -  T  -
             ;  /|  :
            ;  / |  :
          dA  /  |  dB   ~~> dA-dB=d;
          ;  /   |  :
         ;  /    |  :
        -  A  .  B  -
           |-2*e-|
        (xa,ya)  (xb,yb) ~~> ya=yb=0; xb=-xa=e;
        --------------------------------------------------
        (xt, yt): position of audio source
        dA, dB: distance between receiver A or B, and target T;
        e: half ear distance;
            2*e=0.215
        d: distance of two signal difference in spatial position;
        rt: distance ratio, predicted by energy ratio (ball model: described bellow);
        --------------------------------------------------
        position prediction functions:
        {1} distance difference & phase:
            `lambda`: wave length;
            `phi`: phase in rad;
            v: sound speed;
            f: frequency of sound;
            Tc: temperature in Celcius;
                {1.1} --> `lambda`=v/f;
                {1.2} --> v=331*(1+Tc/273)^(1/2);
                {1.3} ~~> dA-dB=d=`lambda`*`phi`/(2*pi);
        {2} ball model & energy:
            * pA, pB: power of receiver received from A and B;
            * target T radiated sound wave around, but energy weaken by the distance;
            * ignore the effect of sound reflection,
                the receiver will receive same amount of energy at the same distance to T;
                received the same energy on the surface of ball;
            * the energy received by receiver will be considered as a small area `delta`
                that the receiver received from the target;
                `delta`: area of receiver;
                {2.1} --> pA=`delta`/`area_A`; pB=`delta`/`area_B`;
                {2.2} --> `area_A`=4*pi*(dA^2); `area_B`=4*pi*(dB^2);
                {2.3} --> rt=dA/dB;
                {2.4} ~~> pA/pB=(dA/dB)^(-2);
            * aA, aB: amplitude of receiver received from A and B;
                then the power of sine wave from {4.1};
                {2.5} --> pA = aA^2 / 2; pB = aB^2 / 2;
                {2.6} ~~> rt = dA/dB=aB/aA;
        --------------------------------------------------
        calculate source position:
        MATHEMATICA:>
            dA = ((xt + e)^2 + yt^2)^(1/2);
            dB = ((xt - e)^2 + yt^2)^(1/2);
            result = Solve[{dA - dB == d, dA/dB == rt}, {xt, yt}];
            FullSimplify[{xt, yt^2} /. result[[2]]]
        --------------------------------------------------
        {3} output real:
            {3.1} --> xt=(d^2 * (1 + rt))/(4 * e * (-1 + rt));
            {3.2} --> yt^2=-((d^2 - 4 * e^2)*(-4 * e^2 * (-1 + rt)^2 + d^2 * (1 + rt)^2))/(16 * e^2 * (-1 + rt)^2);
            {3.3} --> yt^2 >= 0;
        --------------------------------------------------
        fake position:
            alpha calculated from asymptote angle;
        MATHEMATICA:>
            dA = ((xt + e)^2 + yt^2)^(1/2);
            dB = ((xt - e)^2 + yt^2)^(1/2);
            result = Solve[{yt/xt == tana, dA/dB == rt}, {xt, yt}];
            FullSimplify[{xt, yt} /. result[[2]]]
        --------------------------------------------------
        {7} output fake:
            tana: tan(alpha);
            alpha: fake angle;
            {7.1} --> xt=(e + e*rt^2 + sqrt(e^2*(4*rt^2 - (-1 + rt^2)^2*tana^2)))/((-1 + rt^2)*(1 + tana^2))
            {7.2} --> yt=(tana*(e + e*rt^2 + sqrt(e^2*(4*rt^2 - (-1 + rt^2)^2*tana^2))))/((-1 + rt^2)*(1 + tana^2))
        """
        array_0 = arrays[0]
        array_1 = arrays[1]
        log_array_0 = log_arrays[0]
        log_array_1 = log_arrays[1]
        phase_0 = phases[0]
        phase_1 = phases[1]
        x_array = []
        y_array = []
        p_array = []
        e_array = []
        pitches = []
        ratio_array = []
        i_s = []
        real_fake_array = []
        e = self.ear_distance / 2
        sound_speed = self._analyze_sound_speed()
        for i in range(len(array_0)):
            t0 = array_0[i]
            t1 = array_1[i]
            lt0 = log_array_0[i]
            lt1 = log_array_1[i]
            p0 = phase_0[i]
            p1 = phase_1[i]
            # skip low frequency part
            if i > 0:
                frequency = self._fft_position_to_frequency(i)
                pitch = self._frequency_to_pitch(frequency)
                energy = (lt0 + lt1) / 2
                if pitch > 0 and energy > self.flower_min_source_power:
                    # pitch to `C` as ticks
                    pitch = (pitch * 12 - 3) / 12
                    # from eq. {2.6}
                    rt = t1 / t0
                    d = self._source_distance_difference_from_phase(p0, p1, frequency)
                    y_square_position = self._source_y_square_position(d, e, rt)
                    append_status = False
                    if y_square_position > 0:
                        # real angle position
                        x_position = self._source_x_position(d, e, rt)
                        y_position = y_square_position ** (1 / 2)
                        append_status = True
                        real_fake_array.append(True)
                    else:
                        # get fake angle positions
                        real_fake_status, fake_angle = self._source_asymptote_angle(e, d, frequency, sound_speed,
                                                                                    p1 - p0)
                        if not real_fake_status:
                            tana = np.tan(fake_angle)
                            x_position, y_position = self._source_xy_fake_position(e, rt, tana)
                            if x_position is not None:
                                if y_position > 0:
                                    x_position = -x_position
                                    y_position = -y_position
                                append_status = True
                                real_fake_array.append(False)
                    if append_status:
                        # append positions
                        # noinspection PyUnboundLocalVariable
                        power = self._source_power(t0, (x_position ** 2 + y_position ** 2) ** (1 / 2))
                        x_array.append(x_position)
                        y_array.append(y_position)
                        p_array.append(power)
                        e_array.append(energy)
                        pitches.append(pitch)
                        ratio_array.append(self._amplitude_ratio(lt0, lt1))
                        i_s.append(i)
        return x_array, y_array, p_array, e_array, pitches, ratio_array, real_fake_array, i_s

    def _prepare_graph_source(self, starting_time, save_path=None, dynamic_max_value=False):
        valid = self._check_analyze_duration(starting_time)
        if not valid:
            return False
        else:
            fft_data, phase_data = self._analyze_get_audio_fft_data(starting_time, phase=True)
            log_fft_data = self._analyze_log_min_max_transform(fft_data, dynamic_max_value=dynamic_max_value)
            (x_positions, y_positions,
             powers, energies,
             pitches, ratio_array,
             rf_array, i_s) = self._source_position_transform(fft_data, log_fft_data, phase_data)
            # making plots
            fig = plt.figure(figsize=self.source_figure_size)
            ax = fig.add_subplot(111)
            # plot human ear
            cir_head = Circle((0, 0), radius=self.ear_distance / 2,
                              edgecolor=self.source_human_color,
                              facecolor=self.source_human_color, linewidth=self.source_line_width, zorder=2, alpha=0.9)
            ax.add_patch(cir_head)
            for x_position in [-self.ear_distance / 2, self.ear_distance / 2]:
                cir_ear = Circle((x_position, 0), radius=self.ear_distance / 6,
                                 edgecolor=self.source_human_color,
                                 facecolor=self.source_human_color, linewidth=self.source_line_width, zorder=1,
                                 alpha=0.9)
                ax.add_patch(cir_ear)

            # plot stars
            for i in range(len(x_positions) - 1):
                pitch = pitches[i]
                energy = energies[i]
                # power = powers[i]
                root_energy = energy ** (1 / 2)
                mean_energy = ((energies[i] + energies[i + 1]) / 2) ** 0.8
                color = self._hsb_to_rgb(pitch % 1, 1, 1)
                if i_s[i + 1] - i_s[i] == 1:
                    ax.plot([x_positions[i], x_positions[i + 1]],
                            [y_positions[i], y_positions[i + 1]], c=color, alpha=mean_energy, zorder=3,
                            linewidth=self.source_line_width)
                if rf_array[i]:
                    face_color = color
                else:
                    face_color = 'black'
                cir_head = Circle((x_positions[i], y_positions[i]), radius=energy / 15, edgecolor=color,
                                  facecolor=face_color, linewidth=self.source_line_width, zorder=4, alpha=root_energy)
                ax.add_patch(cir_head)

            # set figure ratio
            self._set_1to1_ratio_figure(axis=True)
            ax.spines['left'].set_color(self.source_axis_color)
            ax.spines['right'].set_color(self.source_axis_color)
            ax.spines['top'].set_color(self.source_axis_color)
            ax.spines['bottom'].set_color(self.source_axis_color)
            ax.tick_params(axis='x', colors=self.source_axis_color)
            ax.tick_params(axis='y', colors=self.source_axis_color)
            ax.set_ylim(bottom=-self.source_figure_space_limit, top=self.source_figure_space_limit)
            ax.set_xlim(left=-self.source_figure_space_limit, right=self.source_figure_space_limit)

            # save figure
            if save_path is None:
                save_path = self.source_analyzer_path + '.png'
            fig.savefig(save_path, dpi=self.flower_dpi, bbox_inches='tight')
            self._matplotlib_clear_memory(fig)

            # prepare ifft play
            if not dynamic_max_value:
                self._ifft_audio_export(self._analyze_log_min_max_transform(fft_data, log=False))
            return True

    def _source_angle_position_transform(self, arrays, phases):
        sound_speed = self._analyze_sound_speed()
        array_0 = arrays[0]
        array_1 = arrays[1]
        phase_0 = phases[0]
        phase_1 = phases[1]
        x_array = []
        y_array = []
        x_peak = []
        y_peak = []
        e_array = []
        pitches = []
        ratio_array = []
        rf_angles = []
        i_s = []
        e = self.ear_distance / 2
        for i in range(len(array_0)):
            t0 = array_0[i]
            t1 = array_1[i]
            p0 = phase_0[i]
            p1 = phase_1[i]
            # skip low frequency part
            if i > 0:
                frequency = self._fft_position_to_frequency(i)
                pitch = self._frequency_to_pitch(frequency)
                energy = (t0 + t1) / 2
                if pitch > 0 and energy > self.flower_min_analyze_power:
                    # pitch to `C` as ticks
                    pitch = (pitch * 12 - 3) / 12
                    d = self._source_distance_difference_from_phase(p0, p1, frequency)
                    real_fake_status, angle = self._source_asymptote_angle(e, d, frequency, sound_speed, p0 - p1)
                    x_position = pitch * np.cos(angle)
                    y_position = pitch * np.sin(angle)
                    x_peak_position = (pitch + np.power(energy,
                                                        self.flower_stem_power_coefficient)) * np.cos(angle)
                    y_peak_position = (pitch + np.power(energy,
                                                        self.flower_stem_power_coefficient)) * np.sin(angle)
                    x_array.append(x_position)
                    y_array.append(y_position)
                    x_peak.append(x_peak_position)
                    y_peak.append(y_peak_position)
                    e_array.append(energy)
                    pitches.append(pitch)
                    ratio_array.append(self._amplitude_ratio(t0, t1))
                    rf_angles.append(real_fake_status)
                    i_s.append(i)

        return x_array, y_array, x_peak, y_peak, e_array, pitches, ratio_array, rf_angles, i_s

    def _prepare_graph_source_angle(self, starting_time, save_path=None, dynamic_max_value=False):
        valid = self._check_analyze_duration(starting_time)
        if not valid:
            return False
        else:
            fft_data, phase_data = self._analyze_get_audio_fft_data(starting_time, phase=True)
            log_fft_data = self._analyze_log_min_max_transform(fft_data, dynamic_max_value=dynamic_max_value)
            (x_positions, y_positions,
             x_peaks, y_peaks,
             energies, pitches,
             ratio_array, rf_angles, i_s) = self._source_angle_position_transform(log_fft_data, phase_data)
            # making plots
            fig = plt.figure(figsize=self.flower_figure_size)
            ax = fig.add_subplot(111)
            # plot baseline
            max_baseline_circle = int(self._frequency_to_pitch(self.sample_rate / 2)) - 1
            for i in range(1, max_baseline_circle + 1)[::-1]:
                alpha = self.flower_baseline_transform_alpha ** (-i)
                above_arc = Arc((0, 0), 2 * i, 2 * i, theta1=0, theta2=180, zorder=1,
                                linewidth=self.flower_baseline_width, edgecolor=self.flower_baseline_color, alpha=alpha)
                bottom_arc = Arc((0, 0), 2 * i, 2 * i, theta1=180, theta2=360, zorder=1,
                                 linewidth=self.flower_baseline_width, edgecolor=self.flower_baseline_color,
                                 alpha=alpha, linestyle='dashed')
                ax.add_patch(above_arc)
                ax.add_patch(bottom_arc)
                ax.plot([-i, -i + 1], [0, 0], c=self.flower_baseline_color, linewidth=self.flower_baseline_width,
                        alpha=alpha, ls='dashed')
                ax.plot([i - 1, i], [0, 0], c=self.flower_baseline_color, linewidth=self.flower_baseline_width,
                        alpha=alpha, ls='dashed')
            # plot grass
            for i in range(len(x_positions)):
                pitch = pitches[i]
                energy = energies[i]
                color = self._hsb_to_rgb(pitch % 1,
                                         ratio_array[i],
                                         1)
                if rf_angles[i]:
                    face_color = color
                else:
                    face_color = 'black'
                ax.plot([x_positions[i], x_peaks[i]],
                        [y_positions[i], y_peaks[i]], c=color, linewidth=self.flower_line_width, alpha=energy, zorder=2)
                if i != 0 and self._flower_get_angle(x_positions[i],
                                                     x_positions[i - 1],
                                                     y_positions[i],
                                                     y_positions[i - 1]) \
                        < self.flower_min_angle_connection / 180 * np.pi:
                    if i_s[i] - i_s[i - 1] == 1:
                        ax.plot([x_positions[i], x_positions[i - 1]],
                                [y_positions[i], y_positions[i - 1]], c=color,
                                linewidth=self.flower_ground_line_width,
                                alpha=(energies[i] + energies[i - 1]) / 2,
                                zorder=2)
                above_arc = Circle((x_peaks[i], y_peaks[i]), radius=energy / 5, zorder=3, facecolor=face_color,
                                   linewidth=self.flower_line_width, edgecolor=color, alpha=energy)
                ax.add_patch(above_arc)

            # set figure ratio
            ax.set_ylim(bottom=-max_baseline_circle - 1, top=max_baseline_circle + 1)
            ax.set_xlim(left=-max_baseline_circle - 1, right=max_baseline_circle + 1)
            self._set_1to1_ratio_figure()

            # save figure
            if save_path is None:
                save_path = self.source_angle_analyzer_path + '.png'
            fig.savefig(save_path, dpi=self.flower_dpi, bbox_inches='tight')
            self._matplotlib_clear_memory(fig)

            # prepare ifft play
            if not dynamic_max_value:
                self._ifft_audio_export(self._analyze_log_min_max_transform(fft_data, log=False))
            return True

    def _prepare_video_source_angle(self, starting_time, ending_time):
        """ save video for spiral """
        (starting_time,
         ending_time,
         status) = self._prepare_video_analyzer(starting_time, ending_time,
                                                save_analyzer_path=self.source_angle_analyzer_path,
                                                analyzer_function=self._prepare_graph_source_angle)
        return starting_time, ending_time, status

    def _prepare_video_source(self, starting_time, ending_time):
        """ save video for spiral """
        (starting_time,
         ending_time,
         status) = self._prepare_video_analyzer(starting_time, ending_time,
                                                save_analyzer_path=self.source_analyzer_path,
                                                analyzer_function=self._prepare_graph_source)
        return starting_time, ending_time, status
