import numpy as np

from utils.analyze_common import AnalyzeCommon


class FlowerCommon(AnalyzeCommon):
    def __init__(self):
        super().__init__()

        # source & phase analyzer (flowers)
        # figure config
        self.flower_figure_size = (12, 12)
        self.flower_dpi = 200
        self.flower_line_width = 2
        self.flower_ground_line_width = 2.5
        self.flower_baseline_width = 1.2
        self.flower_baseline_transform_alpha = 1.3
        self.flower_stem_power_coefficient = 1.5

        # color & theme
        self.flower_baseline_color = 'dimgray'

        # minimum source power
        self.flower_min_source_power = 0.005
        self.flower_min_analyze_power = 0.05
        self.flower_min_angle_connection = 45

    @staticmethod
    def _flower_get_angle(x0, x1, y0, y1):
        angle_0 = np.arctan2(x0, y0)
        angle_1 = np.arctan2(x1, y1)
        return abs((angle_0 - angle_1 + np.pi) % (2 * np.pi) - np.pi)
