clear all
close all
clc

addpath('E:\program\MATLAB\casadi-3.6.5-windows64-matlab2018b')
import casadi.*

h = 0.2; % step[s]
N = 20; % prediction horizon
I1 = 0.103; I2 = 0.104; I3 = 0.161; 
m = 4.8; % kg
V = 0.0285; % m3
rou = 1000; % kg/m3
G = 9.8; % m/s2
l = 0.6; % m
c1 = 0.01; c2 = 0.01; c3 = 0.01;

% --------------------------------Attitude--------------------------------------%

phi = SX.sym('phi'); theta = SX.sym('theta'); psi = SX.sym('psi');
p = SX.sym('p'); q = SX.sym('q'); r = SX.sym('r');
states = [phi; theta; psi; p; q; r]; n_states = length(states);

t_phi = SX.sym('t_phi'); t_theta = SX.sym('t_theta'); t_psi = SX.sym('t_psi');
controls = [t_phi; t_theta; t_psi]; n_controls = length(controls);

abs_phi_dot = SX.sym('abs_phi_dot');
abs_theta_dot = SX.sym('abs_theta_dot');
abs_psi_dot = SX.sym('abs_psi_dot');
abs_states_dot = [abs_theta_dot; abs_phi_dot; abs_psi_dot];

% rhs = [ p + tan(theta)*sin(phi)*q + tan(theta)*cos(phi)*r; ...
%         cos(phi)*q - sin(phi)*r; ...
%         sin(phi)*q/cos(theta) + cos(phi)*r/cos(theta); ...
%         (t_phi - (I3-I2)*q*r - 0)/I1; ...
%         (t_theta - (I1-I3)*p*r - 0)/I2; ...
%         (t_psi - (I2-I1)*p*q - 0)/I3];
    
rhs = [ p ; ...
        q ; ...
        r; ...
        (t_phi - (I3-I2)*q*r - 0)/I1; ...
        (t_theta - (I1-I3)*p*r - 0)/I2; ...
        (t_psi - (I2-I1)*p*q - 0)/I3];


f = Function('f', {states, controls, abs_states_dot}, {rhs}); % nonlinear mapping function f(x,u)
% f = Function('f', {states, controls}, {rhs}); % nonlinear mapping function f(x,u)

U = SX.sym('U', n_controls, N); % Decision variables (controls)
P = SX.sym('P', 3*n_states-3); 
% parameters (which include the initial state and the reference state)

X = SX.sym('X',n_states,(N+1));
% A vector that represents the states over the optimization problem.

obj = 0; % objective function
g = []; % constraints vector

Q = zeros(6,6); Q(1,1) = 10; Q(2,2) = 10; Q(3,3) = 10; % weighing matrices (states)
                Q(4,4) = 10; Q(5,5) = 10; Q(6,6) = 10;
R = zeros(3,3); R(1,1) = 0; R(2,2) = 0; R(3,3) = 0; % weighing matrices (controls)
S = zeros(3,3); S(1,1) = 0; S(2,2) = 0; S(3,3) = 0;
Z = zeros(3,3); Z(1,1) = 0; Z(2,2) = 0; Z(3,3) = 0;

st = X(:,1); % initial state
abs_st_dot = abs(P(13:15));
con_last = U(:,1);
g = [g;st-P(1:6)]; % initial condition constraints
for k = 1:N
    st = X(:,k);  con = U(:,k); 
    obj = obj+(st-P(7:12))'*Q*(st-P(7:12)) + con'*R*con ...
            + (con - con_last)'*S*(con - con_last) + abs_st_dot'*Z*abs_st_dot; % calculate obj
    con_last = con;
    st_next = X(:,k+1);
    k1 = f(st, con, abs_st_dot);   % new 
    k2 = f(st + h/2*k1, con, abs_st_dot); % new
    k3 = f(st + h/2*k2, con, abs_st_dot); % new
    k4 = f(st + h*k3, con, abs_st_dot); % new
    st_next_RK4=st +h/6*(k1 +2*k2 +2*k3 +k4); % new  
    abs_st_dot(1) = abs((st_next(4) - st(4))/h);
    abs_st_dot(2) = abs((st_next(5) - st(5))/h);
    abs_st_dot(3) = abs((st_next(6) - st(6))/h);
%     f_value = f(st,con, abs_st_dot);
%     st_next_euler = st+ (h*f_value);
%     g = [g;st_next-st_next_euler]; % compute constraints
    g = [g;st_next-st_next_RK4]; % compute constraints % new
end
% obj =+ obj + st_next_RK4'*S*st_next_RK4;
% make the decision variable one column  vector
OPT_variables = [reshape(X,6*(N+1),1);reshape(U,3*N,1)];

nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 99;
opts.ipopt.print_level =1;%0,3
opts.print_time = 1;
opts.ipopt.acceptable_tol =1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob, opts);

args = struct;

args.lbg(1:6*(N+1)) = 0;  % -1e-20  % Equality constraints
args.ubg(1:6*(N+1)) = 0;  % 1e-20   % Equality constraints

args.lbx(1:6:6*(N+1),1) = -pi/2; %state phi lower bound
args.ubx(1:6:6*(N+1),1) = pi/2; %state phi upper bound
args.lbx(2:6:6*(N+1),1) = -pi/2; %state theta lower bound
args.ubx(2:6:6*(N+1),1) = pi/2; %state theta upper bound
args.lbx(3:6:6*(N+1),1) = -pi/2; %state psi lower bound
args.ubx(3:6:6*(N+1),1) = pi/2; %state psi upper bound
args.lbx(4:6:6*(N+1),1) = -inf; %state p lower bound
args.ubx(4:6:6*(N+1),1) = inf; %state p upper bound
args.lbx(5:6:6*(N+1),1) = -inf; %state q lower bound
args.ubx(5:6:6*(N+1),1) = inf; %state q upper bound
args.lbx(6:6:6*(N+1),1) = -inf; %state r lower bound
args.ubx(6:6:6*(N+1),1) = inf; %state r upper bound

args.lbx(6*(N+1)+1:3:6*(N+1)+3*N,1) = -15; %t_theta lower bound
args.ubx(6*(N+1)+1:3:6*(N+1)+3*N,1) = 15; %t_theta upper bound
args.lbx(6*(N+1)+2:3:6*(N+1)+3*N,1) = -15; %t_phi lower bound 
args.ubx(6*(N+1)+2:3:6*(N+1)+3*N,1) = 15; %t_phi upper bound
args.lbx(6*(N+1)+3:3:6*(N+1)+3*N,1) = -15; %t_psi lower bound 
args.ubx(6*(N+1)+3:3:6*(N+1)+3*N,1) = 15; %t_psi upper bound

% --------------------------------Attitude--------------------------------------%

% ---------------------------------Torque---------------------------------------%

% t_x = SX.sym('t_x'); t_y = SX.sym('t_y'); t_z = SX.sym('t_z'); 
% T1 = SX.sym('T1'); T2 = SX.sym('T2');
% states1 = [t_x; t_y; t_z; T1; T2]; n_states1 = length(states1);
% 
% f1 = SX.sym('f1'); f2 = SX.sym('f2'); f3 = SX.sym('f3'); f4 = SX.sym('f4');
% f5 = SX.sym('f5'); f6 = SX.sym('f6'); f7 = SX.sym('f7'); f8 = SX.sym('f8');
% alpha = SX.sym('alpha'); beta = SX.sym('beta'); 
% controls1 = [f1; f2; f3; f4; f5; f6; f7; f8; alpha; beta]; n_controls1 = length(controls1);
% rhs1 = [ -2*l*sin(alpha)*f4 + 2*l*sin(beta)*f2; ...
%         l*(f3+f7-f1-f5); ...
%         l*k*(f1-f5+f3-f7) + 2*l*(f4*cos(alpha)-f2*cos(beta)); ...
%         f1+f5+f3+f7+2*f4*sin(alpha)+2*f2*sin(beta); ...
%         2*f2*cos(beta) + 2*f4*cos(alpha)]; % system r.h.s
% 
% ft = Function('ft', {states1, controls1}, {rhs1}); % nonlinear mapping function f(x,u)
% U1 = SX.sym('U1', n_controls1, N); % Decision variables (controls)
% P1 = SX.sym('P1', n_states1 + n_states1);
% % parameters (which include the initial state and the reference state)
% 
% X1 = SX.sym('X1',n_states1,(N+1));
% % A vector that represents the states over the optimization problem.
% 
% obj1 = 0; % objective function
% g1 = []; % constraints vector
% 
% Q1 = zeros(5,5);
% Q1(1,1) = 1; Q1(2,2) = 5; Q1(3,3) = 0.1; Q1(4,4) = 0.1; Q1(5,5) = 0.1; % weighing matrices (states)
% R1 = zeros(10,10); 
% R1(1,1) = 0.5; R1(2,2) = 0.05; R1(3,3) = 0.5; R1(4,4) = 0.5; R1(5,5) = 0.5;
% R1(6,6) = 0.5; R1(7,7) = 0.05; R1(8,8) = 0.5; R1(9,9) = 0.5; R1(10,10) = 0.5;% weighing matrices (controls)
% 
% st1  = X1(:,1); % initial state
% g1 = [g1;st1-P1(1:5)]; % initial condition constraints
% for k = 1:N
%     st1 = X1(:,k);  con1 = U1(:,k);
%     obj1 = obj1+(st1-P1(6:10))'*Q1*(st1-P1(6:10)) + con1'*R1*con1; % calculate obj
%     st_next1 = X1(:,k+1);
%     st_next_f = ft(st1, con1);
% %     k1 = f(st1, con1);   % new 
% %     k2 = f(st1 + h/2*k1, con1); % new
% %     k3 = f(st1 + h/2*k2, con1); % new
% %     k4 = f(st1 + h*k3, con1); % new
% %     st_next_RK4=st1 +h/6*(k1 +2*k2 +2*k3 +k4); % new    
%     % f_value = f(st,con);
%     % st_next_euler = st+ (h*f_value);
%     % g = [g;st_next-st_next_euler]; % compute constraints
%     g1 = [g1;st_next1-st_next_f]; % compute constraints % new
% end
% % make the decision variable one column  vector
% OPT_variables1 = [reshape(X1,5*(N+1),1);reshape(U1,10*N,1)];
% 
% nlp_prob = struct('f', obj1, 'x', OPT_variables1, 'g', g1, 'p', P1);
% 
% opts = struct;
% opts.ipopt.max_iter = 2000;
% opts.ipopt.print_level =0;%0,3
% opts.print_time = 0;
% opts.ipopt.acceptable_tol =1e-8;
% opts.ipopt.acceptable_obj_change_tol = 1e-6;
% 
% solver1 = nlpsol('solver', 'ipopt', nlp_prob,opts);
% 
% args1 = struct;
% 
% args1.lbg(1:5*(N+1)) = 0;  % -1e-20  % Equality constraints
% args1.ubg(1:5*(N+1)) = 0;  % 1e-20   % Equality constraints
% 
% args1.lbx(1:5:5*(N+1),1) = -inf; %state t_x lower bound
% args1.ubx(1:5:5*(N+1),1) = inf; %state t_x upper bound
% args1.lbx(2:5:5*(N+1),1) = -inf; %state t_y lower bound
% args1.ubx(2:5:5*(N+1),1) = inf; %state t_y upper bound
% args1.lbx(3:5:5*(N+1),1) = -inf; %state t_z lower bound
% args1.ubx(3:5:5*(N+1),1) = inf; %state t_z upper bound
% args1.lbx(4:5:5*(N+1),1) = -inf; %state T1 lower bound
% args1.ubx(4:5:5*(N+1),1) = inf; %state T1 upper bound
% args1.lbx(5:5:5*(N+1),1) = -inf; %state T2 lower bound
% args1.ubx(5:5:5*(N+1),1) = inf; %state T2 upper bound
% 
% args1.lbx(5*(N+1)+1:10:5*(N+1)+10*N,1) = 0; %f1 lower bound
% args1.ubx(5*(N+1)+1:10:5*(N+1)+10*N,1) = inf; %f1 upper bound
% args1.lbx(5*(N+1)+2:10:5*(N+1)+10*N,1) = -inf; %f2 lower bound 
% args1.ubx(5*(N+1)+2:10:5*(N+1)+10*N,1) = inf; %f2 upper bound
% args1.lbx(5*(N+1)+3:10:5*(N+1)+10*N,1) = 0; %f3 lower bound 
% args1.ubx(5*(N+1)+3:10:5*(N+1)+10*N,1) = inf; %f3 upper bound
% args1.lbx(5*(N+1)+4:10:5*(N+1)+10*N,1) = -inf; %f4 lower bound 
% args1.ubx(5*(N+1)+4:10:5*(N+1)+10*N,1) = inf; %f4 upper bound
% args1.lbx(5*(N+1)+5:10:5*(N+1)+10*N,1) = 0; %f5 lower bound 
% args1.ubx(5*(N+1)+5:10:5*(N+1)+10*N,1) = inf; %f5 upper bound
% args1.lbx(5*(N+1)+6:10:5*(N+1)+10*N,1) = -inf; %f6 lower bound 
% args1.ubx(5*(N+1)+6:10:5*(N+1)+10*N,1) = inf; %f6 upper bound
% args1.lbx(5*(N+1)+7:10:5*(N+1)+10*N,1) = 0; %f7 lower bound 
% args1.ubx(5*(N+1)+7:10:5*(N+1)+10*N,1) = inf; %f7 upper bound
% args1.lbx(5*(N+1)+8:10:5*(N+1)+10*N,1) = -inf; %f8 lower bound 
% args1.ubx(5*(N+1)+8:10:5*(N+1)+10*N,1) = inf; %f8 upper bound
% args1.lbx(5*(N+1)+9:10:5*(N+1)+10*N,1) = -pi/2; %alpha lower bound 
% args1.ubx(5*(N+1)+9:10:5*(N+1)+10*N,1) = pi/2; %alpha upper bound
% args1.lbx(5*(N+1)+10:10:5*(N+1)+10*N,1) = -pi/2; %beta lower bound 
% args1.ubx(5*(N+1)+10:10:5*(N+1)+10*N,1) = pi/2; %beta upper bound

% ---------------------------------Torque---------------------------------------%
t0 = 0;
t01 = 0;
x0 = [0 ; 0 ; 0.0; 0; 0; 0];    % initial condition.
x01 = [0; 0; 0; 0; 0];
xs = [0 ; pi/6 ; pi/5; 0; 0; 0]; % Reference posture. % 这里要改
xs1 = [0; 0; 0; 0; 0];
x_dot = [0; 0; 0];
x_doth(:,1) = x_dot;
obj_h = [];

xx(:,1) = x0; % xx contains the history of states
xx01(:,1) = x01; % xx contains the history of states
t(1) = t0;

u0 = zeros(N,3);        % two control inputs for each robot
u01 = zeros(N,10); 
X0 = repmat(x0,1,N+1)'; % initialization of the states decision variables
X01 = repmat(x01,1,N+1)'; % initialization of the states decision variables

u_out = [0; 0; 0];
% sim_tim = 20; % Maximum simulation time

% Start MPC
mpciter = 0;
xx1 = [];
u_cl=[];
u_cl1=[];

figure;
subplot(3,3,1); % 上半部分用于显示状态
h1 = plot(t, xx(1,:), 'b', 'LineWidth', 2);
title('phi');
xlabel('t');
ylabel('phi');

subplot(3,3,4); % 上半部分用于显示状态
h2 = plot(t, xx(2,:), 'b', 'LineWidth', 2);
title('theta');
xlabel('t');
ylabel('theta');

subplot(3,3,7); % 上半部分用于显示状态
h3 = plot(t, xx(3,:), 'b', 'LineWidth', 2);
title('psi');
xlabel('t');
ylabel('psi');

subplot(3,3,2); % 上半部分用于显示状态
h4 = plot(t, xx(3,:), 'b', 'LineWidth', 2);
title('p');
xlabel('t');
ylabel('p');

subplot(3,3,5); % 上半部分用于显示状态
h5 = plot(t, xx(3,:), 'b', 'LineWidth', 2);
title('q');
xlabel('t');
ylabel('q');

subplot(3,3,8); % 上半部分用于显示状态
h6 = plot(t, xx(3,:), 'b', 'LineWidth', 2);
title('r');
xlabel('t');
ylabel('r');

u = [0; 0; 0];
subplot(3,3,3); % 下半部分用于显示控制输入
h7 = plot(t, u', 'r', 'LineWidth', 2);
title('t_x');
xlabel('t');
ylabel('t_x');

subplot(3,3,6); % 下半部分用于显示控制输入
h8 = plot(t, u', 'r', 'LineWidth', 2);
title('t_y');
xlabel('t');
ylabel('t_y');

subplot(3,3,9); % 下半部分用于显示控制输入
h9 = plot(t, u', 'r', 'LineWidth', 2);
title('t_z');
xlabel('t');
ylabel('t_z');


while mpciter < 100
    % --------------------------------Attitude--------------------------------------%
%     xs = control_sliders();
    args.p = [x0;xs;x_dot]; % set the values of the parameters vector
    % initial value of the optimization variables
    args.x0  = [reshape(X0',6*(N+1),1);reshape(u0',3*N,1)];
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);
    u = reshape(full(sol.x(6*(N+1)+1:end))',3,N)'; % get controls only from the solution
    obj_h = [obj_h; full(sol.f)];
    xx1(:,1:6,mpciter+1)= reshape(full(sol.x(1:6*(N+1)))',6,N+1)'; % get solution TRAJECTORY
    u_cl= [u_cl ; u(1,:)];
%     t(mpciter+1) = t0;
    % Apply the control and shift the solution
    [t0, x0, u0] = shift(h, t0, x0, u, f, x_dot);
    xx(:,mpciter+2) = x0;
%     x0(1:3,:) = x0(1:3,:) + unifrnd(-pi/5,pi/5,3,1);
    x_dot = (xx(4:6, mpciter+2) - xx(4:6, mpciter+1))/h;
    x_doth(:,mpciter+2) = x_dot;
    X0 = reshape(full(sol.x(1:6*(N+1)))',6,N+1)'; % get solution TRAJECTORY
    % Shift trajectory to initialize the next step
    X0 = [X0(2:end,:);X0(end,:)];
    % --------------------------------Attitude--------------------------------------%

    % ---------------------------------Torque---------------------------------------%
%     xs1 = [u(1,:)'; m*G - rou*V*G; 0];
%     args1.p = [x01;xs1]; % set the values of the parameters vector
%     % initial value of the optimization variables
%     args1.x0  = [reshape(X01',5*(N+1),1);reshape(u01',10*N,1)];
%     sol1 = solver1('x0', args1.x0, 'lbx', args1.lbx, 'ubx', args1.ubx,...
%         'lbg', args1.lbg, 'ubg', args1.ubg,'p',args1.p);
%     u1 = reshape(full(sol1.x(5*(N+1)+1:end))',10,N)'; % get controls only from the solution
%     xx11(:,1:3,mpciter+1)= reshape(full(sol1.x(1:3*(N+1)))',3,N+1)'; % get solution TRAJECTORY
%     u_cl1= [u_cl1 ; u1(1,:)];
% %     t(mpciter+1) = t0;
%     % Apply the control and shift the solution
%     [t01, x01, u01] = shift1(h, t01, x01, u1,ft);
%     xx01(:,mpciter+2) = x01;
%     X01 = reshape(full(sol1.x(1:5*(N+1)))',5,N+1)'; % get solution TRAJECTORY
%     % Shift trajectory to initialize the next step
%     X01 = [X01(2:end,:);X01(end,:)];
    % ---------------------------------Torque---------------------------------------%
 
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
    set(h7, 'XData', t, 'YData', u_cl(:,1)); % 更新控制输入图像
    set(h8, 'XData', t, 'YData', u_cl(:,2)); % 更新控制输入图像
    set(h9, 'XData', t, 'YData', u_cl(:,3)); % 更新控制输入图像

    drawnow; % 刷新图形窗口以实时显示最新图像
    % ----------------------------------Draw----------------------------------------%
end
