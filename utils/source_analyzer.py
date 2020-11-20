from utils.analyze_common import AnalyzeCommon


class SourceAnalyzer(AnalyzeCommon):
    def __init__(self):
        super().__init__()

    def _source_x_position(self):
        pass

    def _source_y_square_position(self):
        pass

    def _source_power(self):
        pass

    def _source_position_single(self):
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
                {2.3} ~~> rt=pA/pB=(dA/dB)^(-2);
        --------------------------------------------------
        calculate source position:
        MATHEMATICA:>
            dA = ((xt - e)^2 + yt^2)^(1/2);
            dB = ((xt + e)^2 + yt^2)^(1/2);
            result = Solve[{dA - dB == d, dA/dB == rt}, {xt, yt}];
            FullSimplify[{xt, yt^2} /. result[[2]]]
        --------------------------------------------------
        {3} output:
            {3.1} --> xt=(d^2 + d^2 * rt)/(4*e - 4 * e * rt)
            {3.2} --> yt^2=((d^2 - 4 * e^2)*(-4 * e^2 * (-1 + rt)^2 + d^2 * (1 + rt)^2))/(16 * e^2 * (-1 + rt)^2)
            {3.3} --> yt^2 >= 0
        """
        pass

    def _source_power_single(self):
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
        pass
