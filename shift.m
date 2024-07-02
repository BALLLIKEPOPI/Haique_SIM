function [t0, x0, u0] = shift(T, t0, x0, u,f, x_dot)
st = x0;
con = u(1,:)';
k1 = f(st, con, abs(x_dot));   % new 
k2 = f(st + T/2*k1, con, abs(x_dot)); % new
k3 = f(st + T/2*k2, con, abs(x_dot)); % new
k4 = f(st + T*k3, con, abs(x_dot)); % new
st_next_RK4=st +T/6*(k1 +2*k2 +2*k3 +k4); % new   
x0 = full(st_next_RK4);

t0 = t0 + T;
u0 = [u(2:size(u,1),:);u(size(u,1),:)];
end
