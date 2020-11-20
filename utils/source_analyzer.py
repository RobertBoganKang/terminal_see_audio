from utils.analyze_common import AnalyzeCommon


class SourceAnalyzer(AnalyzeCommon):
    def __init__(self):
        super().__init__()

    def _source_x(self):
        pass

    def _source_position_single(self):
        """
        MATHEMATICA code (calculate source position):
        --------------------------------------------------
        A = ((xt - e)^2 + yt^2)^(1/2);
        B = ((xt + e)^2 + yt^2)^(1/2);
        result = Solve[{A - B == d, A/B == rt}, {xt, yt}];
        FullSimplify[{xt, yt^2} /. result[[2]]]
        --------------------------------------------------
        [xt, yt]: position of audio source
        e: half ear distance
        d: distance of two signal difference in spatial position
        rt: distance ratio, predicted by energy ratio (ball model)
        --------------------------------------------------

        """
        pass
