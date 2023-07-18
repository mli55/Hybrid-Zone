function MicroParam = InitMicroParam(SignalParam,FigureParam)
    % �� x�������ϵ�mic1Ϊԭ��

    MicroParam.MicroTotal = SignalParam.ChannelNum;
    % ��˷���һ�㵽���ĵľ��루��λ���ף�
    MicroParam.toEdge = 0.0475;
    % ��˷��豸���������λ��
    MicroParam.Position = 0;
    % ������˷�нǲ�ֵ
    MicroParam.Theta = 2 * pi / MicroParam.MicroTotal;
    % ��˷����и�����˷��λ��
    MicroParam.Angles=-MicroParam.Theta*(0:1:MicroParam.MicroTotal-1)/pi*180+210;
    MicroParam.Positions = MicroParam.toEdge * exp(1i*MicroParam.Angles/180*pi) + MicroParam.Position;
    
    if (FigureParam.MicMap)
        scatter(real(MicroParam.Positions),imag(MicroParam.Positions));
        axis([-0.5,0.5,-0.5,0.5]);
        title('MicroPhone Layout');
        grid on;
    end
end