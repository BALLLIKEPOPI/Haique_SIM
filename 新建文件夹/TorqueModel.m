function [T] = TorqueModel(w1, w2, w3, w4, w5, w6, w7, w8, alpha, beta)
    l = 330; % ���
    u = 1; % ת���������ı���ϵ��
    k = 1; % ��Ťϵ��
    t_theta = l*u*(w1^2 - w3^2 + w5^2 - w7^2);
    t_phi = l*u*(sin(alpha)*w4^2 - sin(beta)*w2^2 + sin(alpha)*w8^2 - sin(beta)*w6^2) + k(cos(alpha)*w4^2 - cos(alpha)*w8^2 + cos(beta)*w2^2 - cos(beta)*w6^2);
    t_psi = k(-w1^2 + w5^2 - w3^2 + w7^2 + sin(alpha)*w4^2 - sin(alpha)*w8^2 + sin(beta)*w2^2 - sin(beta)*w6^2);
    T = [t_theta; t_phi; t_psi];
end

