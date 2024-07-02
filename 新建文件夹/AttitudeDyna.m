function [wbx_dot, wby_dot, wbz_dot] = AttitudeDyna(wbx, wby, wbz, t_phi, t_theta, t_psi, phi_dot, theta_dot, psi_dot)
    d = 1000; % 水密度 1000kg/m3
%     c1 = ; % 阻力系数
%     c2 = ;
%     c3 = ;
%     I1 = ; % 转动惯量
%     I2 = ;
%     I3 = ;
    wbx_dot = t_phi/I1 - (I3-I2)*wby*wbz/I1 - d*c1*abs(phi_dot)*wbx/I1;
    wby_dot = t_theta/I2 - (I1-I2)*wbx*wbz/I2 - d*c2*abs(theta_dot)*wby/I2;
    wbz_dot = t_psi/I3 - (I2-I1)*wbx*wbz/I3 - d*c2*abs(psi_dot)*wby/I3;
end