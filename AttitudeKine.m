% wbx,wby,wbz -- 机体坐标系下的无人机角速度
% theta,phi,psi -- 欧拉角
% theta_dot,phi_dot,psi_dot -- 欧拉角角速率
% 将机体坐标系下的角速率转化为世界坐标系下的姿态变换率
function [theta_dot, phi_dot, psi_dot] = AttitudeKine(wbx, wby, wbz, theta, phi)
    theta_dot = wbx + tan(theta)*sin(phi)*wby + tan(theta)*cos(phi)*wbz;
    phi_dot = cos(phi)*wby - sin(phi)*wbz;
    psi_dot = sin(phi)/cos(phi)*wby + cos(phi)/cos(theta)*wbz;
end