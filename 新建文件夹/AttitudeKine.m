% wbx,wby,wbz -- ��������ϵ�µ����˻����ٶ�
% theta,phi,psi -- ŷ����
% theta_dot,phi_dot,psi_dot -- ŷ���ǽ�����
% ����������ϵ�µĽ�����ת��Ϊ��������ϵ�µ���̬�任��
function [theta_dot, phi_dot, psi_dot] = AttitudeKine(wbx, wby, wbz, theta, phi)
    theta_dot = wbx + tan(theta)*sin(phi)*wby + tan(theta)*cos(phi)*wbz;
    phi_dot = cos(phi)*wby - sin(phi)*wbz;
    psi_dot = sin(phi)/cos(phi)*wby + cos(phi)/cos(theta)*wbz;
end