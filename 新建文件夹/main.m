clear all
close all
clc

addpath('D:\Program\MATLAB\casadi-3.6.5-windows64-matlab2018b')
import casadi.*

h = 0.2; % step[s]
N = 5; % prediction horizon

I1 = 50; I2 = 50; I3 = 50;

theta = SX.sym('theta'); phi = SX.sym('phi'); psi = SX.sym('psi');
states = [theta; phi; psi]; n_states = length(states);

t_theta = SX.sym('t_theta'); t_phi = SX.sym('t_phi'); t_psi = SX.sym('t_psi');
controls = [t_theta; t_phi; t_psi]; n_control = length(controls);
rhs = [(t_theta - (I3-I2)*phi*psi - r*c1*abs(theta_dot)*theta)/I1; ...
        (t_phi - (I1-I3)*theta*psi - r*c2*abs(phi_dot)*phi)/I2; ...
        (t_psi - (I2-I1)*theta*phi - r*c3*abs(psi_dot)*psi)/I3]; % system r.h.s

f = Function('f', {states, controls}, {rhs}); % nonlinear mapping function f(x,u)
U = SX.sym('U', n_controls, N); % Decision variables (controls)
P = SX.sym('P', n_states + n_states);
% parameters (which include the initial state and the reference state)

X = SX.sym('X',n_states,(N+1));
% A vector that represents the states over the optimization problem.

obj = 0; % objective function
g = []; % constraints vector

Q = zeros(3,3); Q(1,1) = 1;Q(2,2) = 5;Q(3,3) = 0.1; % weighing matrices (states)
R = zeros(2,2); R(1,1) = 0.5; R(2,2) = 0.05; % weighing matrices (controls)

st  = X(:,1); % initial state
g = [g;st-P(1:3)]; % initial condition constraints
for k = 1:N
    st = X(:,k);  con = U(:,k);
    obj = obj+(st-P(4:6))'*Q*(st-P(4:6)) + con'*R*con; % calculate obj
    st_next = X(:,k+1);
    k1 = f(st, con);   % new 
    k2 = f(st + h/2*k1, con); % new
    k3 = f(st + h/2*k2, con); % new
    k4 = f(st + h*k3, con); % new
    st_next_RK4=st +h/6*(k1 +2*k2 +2*k3 +k4); % new    
    % f_value = f(st,con);
    % st_next_euler = st+ (h*f_value);
    % g = [g;st_next-st_next_euler]; % compute constraints
    g = [g;st_next-st_next_RK4]; % compute constraints % new
end
% make the decision variable one column  vector
OPT_variables = [reshape(X,3*(N+1),1);reshape(U,3*N,1)];

nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 2000;
opts.ipopt.print_level =0;%0,3
opts.print_time = 0;
opts.ipopt.acceptable_tol =1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob,opts);

args = struct;

args.lbg(1:3*(N+1)) = 0;  % -1e-20  % Equality constraints
args.ubg(1:3*(N+1)) = 0;  % 1e-20   % Equality constraints

args.lbx(1:3:3*(N+1),1) = 0; %state theta lower bound
args.ubx(1:3:3*(N+1),1) = 90; %state theta upper bound
args.lbx(2:3:3*(N+1),1) = 0; %state phi lower bound
args.ubx(2:3:3*(N+1),1) = 90; %state phi upper bound
args.lbx(3:3:3*(N+1),1) = 0; %state psi lower bound
args.ubx(3:3:3*(N+1),1) = 90; %state psi upper bound

args.lbx(3*(N+1)+1:3:3*(N+1)+3*N,1) = -inf; %t_theta lower bound
args.ubx(3*(N+1)+1:3:3*(N+1)+3*N,1) = inf; %t_theta upper bound
args.lbx(3*(N+1)+2:3:3*(N+1)+3*N,1) = -inf; %t_phi lower bound 
args.ubx(3*(N+1)+2:3:3*(N+1)+3*N,1) = inf; %t_phi upper bound
args.lbx(3*(N+1)+3:3:3*(N+1)+3*N,1) = -inf; %t_psi lower bound 
args.ubx(3*(N+1)+3:3:3*(N+1)+3*N,1) = inf; %t_psi upper bound


t0 = 0;
x0 = [0 ; 0 ; 0.0];    % initial condition.
xs = [0.0 ; 0.0 ; 0.0]; % Reference posture. % 这里要改

xx(:,1) = x0; % xx contains the history of states
t(1) = t0;

u0 = zeros(N,3);        % two control inputs for each robot
X0 = repmat(x0,1,N+1)'; % initialization of the states decision variables

% sim_tim = 20; % Maximum simulation time

% Start MPC
mpciter = 0;
xx1 = [];
u_cl=[];

while 1
    xs = control_sliders();
    args.p   = [x0;xs]; % set the values of the parameters vector
    % initial value of the optimization variables
    args.x0  = [reshape(X0',3*(N+1),1);reshape(u0',3*N,1)];
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);
    u = reshape(full(sol.x(3*(N+1)+1:end))',3,N)'; % get controls only from the solution
    xx1(:,1:3,mpciter+1)= reshape(full(sol.x(1:3*(N+1)))',3,N+1)'; % get solution TRAJECTORY
    u_cl= [u_cl ; u(1,:)];
    t(mpciter+1) = t0;
    % Apply the control and shift the solution
    [t0, x0, u0] = shift(h, t0, x0, u,f);
    xx(:,mpciter+2) = x0;
    X0 = reshape(full(sol.x(1:3*(N+1)))',3,N+1)'; % get solution TRAJECTORY
    % Shift trajectory to initialize the next step
    X0 = [X0(2:end,:);X0(end,:)];
    mpciter
    mpciter = mpciter + 1;
    
    
end
