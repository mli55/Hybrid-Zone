function [b_value,a_value] = DesignLPF(Filter_Pass,Filter_Stop,fs)
    %��ͨ�˲�
    %ʹ��ע�����ͨ��������Ľ�ֹƵ�ʵ�ѡȡ��Χ�ǲ��ܳ��������ʵ�һ��
    rp=0.1;rs=30;%ͨ����˥��DBֵ�������˥��DBֵ
    wp=2*pi*Filter_Pass/fs;
    ws=2*pi*Filter_Stop/fs;
    % ����б�ѩ���˲�����
    [n,~]=cheb1ord(wp/pi,ws/pi,rp,rs);
    [b_value,a_value]=cheby1(n,rp,wp/pi,'low');
    % �鿴Ƶ����Ӧ����Ҫ�鿴�Լ������Ƶ�ʶ��ڲ��Ƿ��ȶ����ⲿ˥���Ƿ��㹻
%     if (FigureParam.LPFfig)
%         figure
%         W = -fs:1:fs;
%         [Hb,wb]=freqz(b_value,a_value,W,fs);
%         plot(wb,20*log10(abs(Hb)),'b');
%         xlabel('Hz');
%         ylabel('��ֵ/dB');
%     end
end