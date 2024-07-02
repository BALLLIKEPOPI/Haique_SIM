function [t0, x0, u0] = shift(T, t0, x0, u,f)
st = x0;
con = u(1,:)';
k1 = f(st, con);   % new 
k2 = f(st + h/2*k1, con); % new
k3 = f(st + h/2*k2, con); % new
k4 = f(st + h*k3, con); % new
st_next_RK4=st +h/6*(k1 +2*k2 +2*k3 +k4); % new   
x0 = full(st_next_RK4);

t0 = t0 + T;
u0 = [u(2:size(u,1),:);u(size(u,1),:)];
end
