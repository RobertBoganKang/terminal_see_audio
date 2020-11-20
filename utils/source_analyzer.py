from utils.analyze_common import AnalyzeCommon


class SourceAnalyzer(AnalyzeCommon):
    def __init__(self):
        super().__init__()

    def _source_x(self):
        pass

    def _source_position_single(self):
        """
        --------------------------------------------------
              (xt,yt)
              -  T  -
             ;  /|  :
            ;  / |  :
          dA  /  |  dB   ~~> dA-dB=d;
          ;  /   |  :
         ;  /    |  :
        -  A  !  B  -
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
        functions:
        {1} distance difference & phase:
            `lambda`: wave length;
            `phi`: phase in rad;
            v: sound speed;
            f: frequency of sound;
            Tc: temperature in Celcius;
                --> `lambda`=v/f;
                --> v=331*(1+Tc/273)^(1/2);
                ~~> dA-dB=d=`lambda`*`phi`/(2*pi);
        {2} ball model & energy:
            * pA, pB: power of receiver received from A and B;
            * target T radiated sound wave around, but energy weaken by the distance;
            * ignore the effect of sound reflection,
                the receiver will receive same amount of energy at the same distance to T;
                received the same energy on the surface of ball;
            * the energy received by receiver will be considered as a small area `delta`
                that the receiver received from the target;
                --> pA=`delta`/`area_A`; pB=`delta`/`area_B`;
                --> `area_A`=4*pi*(dA^2); `area_B`=4*pi*(dB^2);
                ~~> rt=pA/pB=(dA/dB)^(-1/2);
        --------------------------------------------------
        MATHEMATICA code (calculate source position):
            dA = ((xt - e)^2 + yt^2)^(1/2);
            dB = ((xt + e)^2 + yt^2)^(1/2);
            result = Solve[{dA - dB == d, dA/dB == rt}, {xt, yt}];
            FullSimplify[{xt, yt^2} /. result[[2]]]
        --------------------------------------------------
        output:
            xt=(d^2 + d^2*rt)/(4*e - 4*e*rt)
            yt^2=((d^2 - 4*e^2)*(-4*e^2*(-1 + rt)^2 + d^2*(1 + rt)^2))/(16*e^2*(-1 + rt)^2)
        """
        pass
