function [args, solver] = AttitudeMPCInit(h, N)
I1 = 50; I2 = 50; I3 = 50;

theta = SX.sym('theta'); phi = SX.sym('phi'); psi = SX.sym('psi');
states = [theta; phi; psi]; n_states = length(states);

t_theta = SX.sym('t_theta'); t_phi = SX.sym('t_phi'); t_psi = SX.sym('t_psi');
controls = [t_theta; t_phi; t_psi]; n_controls = length(controls);
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

end
