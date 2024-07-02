function [T] = TorqueModel(w1, w2, w3, w4, w5, w6, w7, w8, alpha, beta)
    l = 330; % 轴距
    u = 1; % 转速与推力的比例系数
    k = 1; % 反扭系数
    t_theta = l*u*(w1^2 - w3^2 + w5^2 - w7^2);
    t_phi = l*u*(sin(alpha)*w4^2 - sin(beta)*w2^2 + sin(alpha)*w8^2 - sin(beta)*w6^2) + k(cos(alpha)*w4^2 - cos(alpha)*w8^2 + cos(beta)*w2^2 - cos(beta)*w6^2);
    t_psi = k(-w1^2 + w5^2 - w3^2 + w7^2 + sin(alpha)*w4^2 - sin(alpha)*w8^2 + sin(beta)*w2^2 - sin(beta)*w6^2);
    T = [t_theta; t_phi; t_psi];
end

