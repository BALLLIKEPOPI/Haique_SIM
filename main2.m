clear all
close all
clc

addpath('E:\program\MATLAB\casadi-3.6.5-windows64-matlab2018b')
import casadi.*

h = 0.2; % step[s]
N = 20; % prediction horizon
I1 = 0.103; I2 = 0.104; I3 = 0.161; 
m = 4.8; % kg
V = 0.00285; % m3
rou = 1000; % kg/m3
G = 9.8; % m/s2
l = 0.6; % m
c1 = 0.01; c2 = 0.01; c3 = 0.01;
k = 0.0000005; % 推力系数
c = 0.0000001; % 反扭系数

% --------------------------------Attitude--------------------------------------%

phi = SX.sym('phi'); theta = SX.sym('theta'); psi = SX.sym('psi');
p = SX.sym('p'); q = SX.sym('q'); r = SX.sym('r');
x = SX.sym('x'); y = SX.sym('y'); z = SX.sym('z');
u = SX.sym('u'); v = SX.sym('v'); w = SX.sym('w');
states = [phi; theta; psi; p; q; r; x; y; z; u; v; w]; n_states = length(states);

T1 = SX.sym('T1'); T2 = SX.sym('T2'); T3 = SX.sym('T3'); T4 = SX.sym('T4');
T5 = SX.sym('T5'); T6 = SX.sym('T6'); T7 = SX.sym('T7'); T8 = SX.sym('T8');
alpha = SX.sym('alpha'); beta = SX.sym('beta');
controls = [T1; T2; T3; T4; T5; T6; T7; T8; alpha; beta]; n_controls = length(controls);

M1 = c*T1/k; M2 = c*T2/k; M3 = c*T3/k; M4 = c*T4/k;
M5 = c*T5/k; M6 = c*T6/k; M7 = c*T7/k; M8 = c*T8/k;

Mx = (-M2+M6)*cos(beta)+(-M4+M8)*cos(alpha)+l*((T2+T6)*sin(beta)-(T4+T8)*sin(alpha));
My = l*(-T1-T5+T3+T7);
Mz = M1+M3-M5-M7+(-M2+M6)*sin(beta)+(-M4+M8)*sin(alpha);
Tx = (T2+T6)*cos(beta)+(T4+T8)*cos(alpha);
Ty = 0;
Tz = T1+T3+T5+T7+(T2+T6)*sin(beta)+(T4+T8)*sin(alpha);

rhs = [ p ; ...
        q ; ...
        r; ...
        (Mx - 0)/I1; ...
        (My - 0)/I2; ...
        (Mz - 0)/I3; ...
        u; ...
        v; ...
        w; ...
        1/m*(cos(theta)*cos(psi)*Tx+(cos(psi)*sin(theta)*cos(phi)+sin(psi)*sin(phi))*Tz); ...
        1/m*(cos(phi)*sin(psi)*Tx+(sin(psi)*sin(theta)*cos(phi)-cos(psi)*sin(phi)*Tz)); ...
        1/m*(-sin(theta)*Tx+cos(phi)*cos(theta)*Tz+(rou*V-m)*G)];

f = Function('f', {states, controls}, {rhs}); % nonlinear mapping function f(x,u)

U = SX.sym('U', n_controls, N); % Decision variables (controls)
P = SX.sym('P', 2*n_states); 
% parameters (which include the initial state and the reference state)

X = SX.sym('X',n_states,(N+1));
% A vector that represents the states over the optimization problem.

obj = 0; % objective function
g = []; % constraints vector

Q = zeros(12,12); Q(1,1) = 5; Q(2,2) = 5; Q(3,3) = 5; Q(4,4) = 10; Q(5,5) = 10; Q(6,6) = 10; % weighing matrices (states)
                 Q(7,7) = 5; Q(8,8) = 5; Q(9,9) = 5; Q(10,10) = 10; Q(11,11) = 10; Q(12,12) = 10;
R = zeros(10,10); R(1,1) = 1; R(2,2) = 1; R(3,3) = 1; R(4,4) = 1; R(5,5) = 1; % weighing matrices (controls)
                 R(6,6) = 1; R(7,7) = 1; R(8,8) = 1; R(9,9) = 10; R(10,10) = 10;

st = X(:,1); % initial state
con_last = U(:,1);
g = [g;st-P(1:12)]; % initial condition constraints
for k = 1:N
    st = X(:,k);  con = U(:,k); 
    obj = obj+(st-P(13:24))'*Q*(st-P(13:24)) + con'*R*con; % calculate obj
    con_last = con;
    st_next = X(:,k+1);
    k1 = f(st, con);   % new 
    k2 = f(st + h/2*k1, con); % new
    k3 = f(st + h/2*k2, con); % new
    k4 = f(st + h*k3, con); % new
    st_next_RK4=st +h/6*(k1 +2*k2 +2*k3 +k4); % new  
%     f_value = f(st,con, abs_st_dot);
%     st_next_euler = st+ (h*f_value);
%     g = [g;st_next-st_next_euler]; % compute constraints
    g = [g;st_next-st_next_RK4]; % compute constraints % new
    g = [g;U(2,k)-U(6,k);U(4,k)-U(8,k);U(9,k)*U(10,k)];
end
% obj =+ obj + st_next_RK4'*S*st_next_RK4;
% make the decision variable one column  vector
OPT_variables = [reshape(X,12*(N+1),1);reshape(U,10*N,1)];

nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 99;
opts.ipopt.print_level =1;%0,3
opts.print_time = 1;
opts.ipopt.acceptable_tol =1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob, opts);

args = struct;

args.lbg(1:12*(N+1)+3*N) = 0;  % -1e-20  % Equality constraints
args.ubg(1:12*(N+1)+3*N) = 0;  % 1e-20   % Equality constraints

args.lbx(1:12:12*(N+1),1) = -pi/2; %state phi lower bound
args.ubx(1:12:12*(N+1),1) = pi/2; %state phi upper bound
args.lbx(2:12:12*(N+1),1) = -pi/2; %state theta lower bound
args.ubx(2:12:12*(N+1),1) = pi/2; %state theta upper bound
args.lbx(3:12:12*(N+1),1) = -pi/2; %state psi lower bound
args.ubx(3:12:12*(N+1),1) = pi/2; %state psi upper bound
args.lbx(4:12:12*(N+1),1) = -inf; %state p lower bound
args.ubx(4:12:12*(N+1),1) = inf; %state p upper bound
args.lbx(5:12:12*(N+1),1) = -inf; %state q lower bound
args.ubx(5:12:12*(N+1),1) = inf; %state q upper bound
args.lbx(6:12:12*(N+1),1) = -inf; %state r lower bound
args.ubx(6:12:12*(N+1),1) = inf; %state r upper bound
args.lbx(7:12:12*(N+1),1) = -inf; %state x lower bound
args.ubx(7:12:12*(N+1),1) = inf; %state x upper bound
args.lbx(8:12:12*(N+1),1) = -inf; %state y lower bound
args.ubx(8:12:12*(N+1),1) = inf; %state y upper bound
args.lbx(9:12:12*(N+1),1) = -inf; %state z lower bound
args.ubx(9:12:12*(N+1),1) = inf; %state z upper bound
args.lbx(10:12:12*(N+1),1) = -inf; %state u lower bound
args.ubx(10:12:12*(N+1),1) = inf; %state u upper bound
args.lbx(11:12:12*(N+1),1) = -inf; %state v lower bound
args.ubx(11:12:12*(N+1),1) = inf; %state v upper bound
args.lbx(12:12:12*(N+1),1) = -inf; %state w lower bound
args.ubx(12:12:12*(N+1),1) = inf; %state w upper bound

args.lbx(12*(N+1)+1:10:12*(N+1)+10*N,1) = 0; %T1 lower bound
args.ubx(12*(N+1)+1:10:12*(N+1)+10*N,1) = 2; %T1 upper bound
args.lbx(12*(N+1)+2:10:12*(N+1)+10*N,1) = -2; %T2 lower bound 
args.ubx(12*(N+1)+2:10:12*(N+1)+10*N,1) = 2; %T2 upper bound
args.lbx(12*(N+1)+3:10:12*(N+1)+10*N,1) = 0; %T3 lower bound 
args.ubx(12*(N+1)+3:10:12*(N+1)+10*N,1) = 2; %T3 upper bound
args.lbx(12*(N+1)+4:10:12*(N+1)+10*N,1) = -2; %T4 lower bound 
args.ubx(12*(N+1)+4:10:12*(N+1)+10*N,1) = 2; %T4 upper bound
args.lbx(12*(N+1)+5:10:12*(N+1)+10*N,1) = 0; %T5 lower bound 
args.ubx(12*(N+1)+5:10:12*(N+1)+10*N,1) = 2; %T5 upper bound
args.lbx(12*(N+1)+6:10:12*(N+1)+10*N,1) = -2; %T6 lower bound 
args.ubx(12*(N+1)+6:10:12*(N+1)+10*N,1) = 2; %T6 upper bound
args.lbx(12*(N+1)+7:10:12*(N+1)+10*N,1) = 0; %T7 lower bound 
args.ubx(12*(N+1)+7:10:12*(N+1)+10*N,1) = 2; %T7 upper bound
args.lbx(12*(N+1)+8:10:12*(N+1)+10*N,1) = -2; %T8 lower bound 
args.ubx(12*(N+1)+8:10:12*(N+1)+10*N,1) = 2; %T8 upper bound
args.lbx(12*(N+1)+9:10:12*(N+1)+10*N,1) = -pi/2; %alpha lower bound 
args.ubx(12*(N+1)+9:10:12*(N+1)+10*N,1) = pi/2; %alpha upper bound
args.lbx(12*(N+1)+10:10:12*(N+1)+10*N,1) = -pi/2; %beta lower bound 
args.ubx(12*(N+1)+10:10:12*(N+1)+10*N,1) = pi/2; %beta upper bound
% --------------------------------Attitude--------------------------------------%


t0 = 0;
t01 = 0;
x0 = [0 ; 0 ; 0.0; 0; 0; 0; 0 ; 0 ; 0.0; 0; 0; 0];    % initial condition.
xs = [0 ; 0 ; 0; 0; 0; 0; 0 ; 1 ; 0.0; 0; 0; 0]; % Reference posture. % 这里要改

obj_h = [];

xx(:,1) = x0; % xx contains the history of states
t(1) = t0;

u0 = zeros(N,10);        % two control inputs for each robot
X0 = repmat(x0,1,N+1)'; % initialization of the states decision variables

u_out = [0; 0; 0];
% sim_tim = 20; % Maximum simulation time

% Start MPC
mpciter = 0;
xx1 = [];
u_cl=[];

figure;
subplot(3,4,1); % 上半部分用于显示状态
h1 = plot(t, xx(1,:), 'b', 'LineWidth', 2);
title('phi');
xlabel('t');
ylabel('phi');

subplot(3,4,5); % 上半部分用于显示状态
h2 = plot(t, xx(2,:), 'b', 'LineWidth', 2);
title('theta');
xlabel('t');
ylabel('theta');

subplot(3,4,9); % 上半部分用于显示状态
h3 = plot(t, xx(3,:), 'b', 'LineWidth', 2);
title('psi');
xlabel('t');
ylabel('psi');

subplot(3,4,2); % 上半部分用于显示状态
h4 = plot(t, xx(3,:), 'b', 'LineWidth', 2);
title('p');
xlabel('t');
ylabel('p');

subplot(3,4,6); % 上半部分用于显示状态
h5 = plot(t, xx(3,:), 'b', 'LineWidth', 2);
title('q');
xlabel('t');
ylabel('q');

subplot(3,4,10); % 上半部分用于显示状态
h6 = plot(t, xx(3,:), 'b', 'LineWidth', 2);
title('r');
xlabel('t');
ylabel('r');

u = [0; 0; 0];
subplot(3,4,3); % 下半部分用于显示控制输入
h7 = plot(t, u', 'r', 'LineWidth', 2);
title('x');
xlabel('t');
ylabel('x');

subplot(3,4,7); % 下半部分用于显示控制输入
h8 = plot(t, u', 'r', 'LineWidth', 2);
title('y');
xlabel('t');
ylabel('y');

subplot(3,4,11); % 下半部分用于显示控制输入
h9 = plot(t, u', 'r', 'LineWidth', 2);
title('z');
xlabel('t');
ylabel('z');

subplot(3,4,4); % 下半部分用于显示控制输入
h10 = plot(t, u', 'r', 'LineWidth', 2);
title('u');
xlabel('t');
ylabel('u');

subplot(3,4,8); % 下半部分用于显示控制输入
h11 = plot(t, u', 'r', 'LineWidth', 2);
title('v');
xlabel('t');
ylabel('v');

subplot(3,4,12); % 下半部分用于显示控制输入
h12 = plot(t, u', 'r', 'LineWidth', 2);
title('w');
xlabel('t');
ylabel('w');

figure;
subplot(3,4,1); % 上半部分用于显示状态
h01 = plot(t, xx(1,:), 'b', 'LineWidth', 2);
title('T1');
xlabel('t');
ylabel('T1');

subplot(3,4,2); % 上半部分用于显示状态
h02 = plot(t, xx(1,:), 'b', 'LineWidth', 2);
title('T2');
xlabel('t');
ylabel('T2');

subplot(3,4,3); % 上半部分用于显示状态
h03 = plot(t, xx(1,:), 'b', 'LineWidth', 2);
title('T3');
xlabel('t');
ylabel('T3');

subplot(3,4,4); % 上半部分用于显示状态
h04 = plot(t, xx(1,:), 'b', 'LineWidth', 2);
title('T4');
xlabel('t');
ylabel('T4');

subplot(3,4,5); % 上半部分用于显示状态
h05 = plot(t, xx(1,:), 'b', 'LineWidth', 2);
title('T5');
xlabel('t');
ylabel('T5');

subplot(3,4,6); % 上半部分用于显示状态
h06 = plot(t, xx(1,:), 'b', 'LineWidth', 2);
title('T6');
xlabel('t');
ylabel('T6');

subplot(3,4,7); % 上半部分用于显示状态
h07 = plot(t, xx(1,:), 'b', 'LineWidth', 2);
title('T7');
xlabel('t');
ylabel('T7');

subplot(3,4,8); % 上半部分用于显示状态
h08 = plot(t, xx(1,:), 'b', 'LineWidth', 2);
title('T8');
xlabel('t');
ylabel('T8');

subplot(3,4,9); % 上半部分用于显示状态
h09 = plot(t, xx(1,:), 'b', 'LineWidth', 2);
title('alpha');
xlabel('t');
ylabel('alpha');

subplot(3,4,10); % 上半部分用于显示状态
h010 = plot(t, xx(1,:), 'b', 'LineWidth', 2);
title('beta');
xlabel('t');
ylabel('beta');

while mpciter < 100
    % --------------------------------Attitude--------------------------------------%
%     xs = control_sliders();
    args.p = [x0;xs]; % set the values of the parameters vector
    % initial value of the optimization variables
    args.x0  = [reshape(X0',12*(N+1),1);reshape(u0',10*N,1)];
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);
    u = reshape(full(sol.x(12*(N+1)+1:end))',10,N)'; % get controls only from the solution
    obj_h = [obj_h; full(sol.f)];
    xx1(:,1:12,mpciter+1)= reshape(full(sol.x(1:12*(N+1)))',12,N+1)'; % get solution TRAJECTORY
    u_cl= [u_cl ; u(1,:)];
%     t(mpciter+1) = t0;
    % Apply the control and shift the solution
    [t0, x0, u0] = shift1(h, t0, x0, u, f);
    xx(:,mpciter+2) = x0;
%     x0(1:3,:) = x0(1:3,:) + unifrnd(-pi/5,pi/5,3,1);
    X0 = reshape(full(sol.x(1:12*(N+1)))',12,N+1)'; % get solution TRAJECTORY
    % Shift trajectory to initialize the next step
    X0 = [X0(2:end,:);X0(end,:)];
    % --------------------------------Attitude--------------------------------------%
 
    t(mpciter+1) = t0;
    mpciter
    mpciter = mpciter + 1;
    
    % ----------------------------------Draw----------------------------------------%
    % 更新图形数据
    set(h1, 'XData', t, 'YData', xx(1,1:end-1)); % 更新状态图像
    set(h2, 'XData', t, 'YData', xx(2,1:end-1)); % 更新状态图像
    set(h3, 'XData', t, 'YData', xx(3,1:end-1)); % 更新状态图像
    set(h4, 'XData', t, 'YData', xx(4,1:end-1)); % 更新状态图像
    set(h5, 'XData', t, 'YData', xx(5,1:end-1)); % 更新状态图像
    set(h6, 'XData', t, 'YData', xx(6,1:end-1)); % 更新状态图像
    set(h7, 'XData', t, 'YData', xx(7,1:end-1)); % 更新状态图像
    set(h8, 'XData', t, 'YData', xx(8,1:end-1)); % 更新状态图像
    set(h9, 'XData', t, 'YData', xx(9,1:end-1)); % 更新状态图像
    set(h10, 'XData', t, 'YData', xx(10,1:end-1)); % 更新状态图像
    set(h11, 'XData', t, 'YData', xx(11,1:end-1)); % 更新状态图像
    set(h12, 'XData', t, 'YData', xx(12,1:end-1)); % 更新状态图像
    
    set(h01, 'XData', t, 'YData', u_cl(1:end,1)); % 更新状态图像
    set(h02, 'XData', t, 'YData', u_cl(1:end,2)); % 更新状态图像
    set(h03, 'XData', t, 'YData', u_cl(1:end,3)); % 更新状态图像
    set(h04, 'XData', t, 'YData', u_cl(1:end,4)); % 更新状态图像
    set(h05, 'XData', t, 'YData', u_cl(1:end,5)); % 更新状态图像
    set(h06, 'XData', t, 'YData', u_cl(1:end,6)); % 更新状态图像
    set(h07, 'XData', t, 'YData', u_cl(1:end,7)); % 更新状态图像
    set(h08, 'XData', t, 'YData', u_cl(1:end,8)); % 更新状态图像
    set(h09, 'XData', t, 'YData', u_cl(1:end,9)); % 更新状态图像
    set(h010, 'XData', t, 'YData', u_cl(1:end,10)); % 更新状态图像

    drawnow; % 刷新图形窗口以实时显示最新图像
    % ----------------------------------Draw----------------------------------------%
end
