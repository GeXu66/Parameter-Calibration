
% Parameters_initialize defines the parameters used in simulation.

% Inputs:
%    - no parameters need to be passed in.

% Outputs:
%    - p is a MATLAB struct containing the initial parameters.

function p = Parameters_initialize

% α_a and α_c: transfer coefficient of surface reaction [-]
p.alpha_a_neg = 0.5;               % anodic charge transfer coefficent in the negative electrode [-]
p.alpha_c_neg = 1 - p.alpha_a_neg; % cathodic charge transfer coefficent in the negative electrode [-]
p.alpha_a_pos = 0.5;               % anodic charge transfer coefficent in the positive electrode [-]
p.alpha_c_pos = 1 - p.alpha_a_pos; % cathodic charge transfer coefficent in the positive electrode [-]
% Note that the cathodic transfer coefficient alpha_c is automatically computed from alpha_a+alpha_c=1.

% σ: solid-phase conductivity [S/m]
p.sigma_neg = 10.0;
p.sigma_pos = 3.8;

% ε_e: porosity (electrolyte volume fraction) [-]
p.epse_neg = 0.485;
p.epse_pos = 0.385;
p.epse_sep = 0.724;

% ε_f: fill volume fraction [-]
p.epsf_neg = 0.0326;
p.epsf_pos = 0.0250;

% Bruggeman constant [-]
p.brug_neg = 1.5;
p.brug_pos = 1.5;
p.brug_sep = 1.5;

% solid-phase Li diffusion coefficient [m^2/s]
p.Ds_neg_0 = 1.2e-14;
p.Ds_pos_0 = 1.0e-14;

% particle radius [m]
p.rs_neg = 2.0e-6;
p.rs_pos = 2.0e-6;

% length of region of cell [m]
p.L_neg = 88e-6;
p.L_pos = 80e-6;
p.L_sep = 25e-6;

% rate constant for the electrochemical reaction [mol / (m^2·s) / (mol/m^3)^(1+p.alpha_a)]
p.k_neg_0 = 5.031e-11;
p.k_pos_0 = 2.334e-11;

% double-layer capacitance [F/m^2]
p.Cdl_neg = 0.1;
p.Cdl_pos = 0.1;

p.delta_sei_neg = 2.0e-6/50;   % δ_sei: sei thickness of the negative electrode [m]
p.rou_sei_neg_0 = 1.4025e5;    % ρ_sei: sei resistivity of the negative electrode [Ω·m]
p.epse_sei_neg = 3.9216e-10;   % ε_sei: sei permittivity of the negative electrode [F/m]

%% Electrolyte parameters
p.kappa_0 = 1.2049;     % κ: conductivity of Li+ in the electrolyte phase [S/m]
p.dlnf_ce = 1.1319;     % dlnf±/dlnce: an electrolyte activity coefficient term [-]
p.t_plus = 0.38;        % t^0_+: Li+ transference number [-]
p.ce_0 = 1000;          % initial bulk electrolyte concentration [mol/m^3]

p.F = 96487;            % Faraday's constant [C/mol]
p.R = 8.314;            % ideal gas constant [J/mol/K]
p.T_0 = 298.15;         % reference temperature [K]

% diffusion coefficient of the salt in the electrolyte phase [m^2/s]: p.De_0 = 3.2228e-10
p.De_0 = 2 * p.R * p.T_0 / p.F^2 / p.ce_0 * (1+p.dlnf_ce) * p.t_plus * (1-p.t_plus) * p.kappa_0;
% Note that the Nernst-Einstein relation in porous media does not hold when the diffusion coefficient De and temperature T change.

%% Thermodynamic data
% Graphite is chosen as the negative electrode.
% LiCoO2   is chosen as the positive electrode.
% θ: stoichiometries [-]
p.s0_neg   = 0.01429;   % at   0% SoC in the negative electrode of a fresh cell [-]
p.s100_neg = 0.85510;   % at 100% SoC in the negative electrode of a fresh cell [-]
p.s0_pos   = 0.99174;   % at   0% SoC in the positive electrode of a fresh cell [-]
p.s100_pos = 0.49950;   % at 100% SoC in the positive electrode of a fresh cell [-]

% the open-circuit potential function [V]
p.U_neg_0 = @(theta_n) 0.7222 + 0.1387*theta_n + 0.029*theta_n.^0.5 - 0.0172./theta_n + 0.0019./theta_n.^1.5 ...
                     + 0.2808*exp(0.9-15*theta_n) - 0.7984*exp(0.4465*theta_n-0.4108);
p.U_pos_0 = @(theta_p) (- 4.656 + 88.669*theta_p.^2 - 401.119*theta_p.^4 + 342.909*theta_p.^6 ...
                        - 462.471*theta_p.^8 + 433.434*theta_p.^10) ./...
                       (- 1 + 18.933*theta_p.^2 - 79.532*theta_p.^4 + 37.311*theta_p.^6 - 73.083*theta_p.^8 + 95.96*theta_p.^10);

% the variation of open-circuit potential with respect to temperature variations [V/K]
p.dUdT_neg = @(theta_n) 0.001 * (0.005269056 + 3.299265709*theta_n - 91.79325798*theta_n.^2 ...
         + 1004.911008*theta_n.^3 - 5812.278127*theta_n.^4 + 19329.7549*theta_n.^5 ...
         - 37147.8947*theta_n.^6 + 38379.18127*theta_n.^7 - 16515.05308*theta_n.^8)...
       ./ (1 - 48.09287227*theta_n + 1017.234804*theta_n.^2 - 10481.80419*theta_n.^3 + 59431.3*theta_n.^4 ...
         - 195881.6488*theta_n.^5 + 374577.3152*theta_n.^6 - 385821.1607*theta_n.^7 + 165705.8597*theta_n.^8);
p.dUdT_pos = @(theta_p) - 0.001 * (0.199521039 - 0.928373822*theta_p + 1.364550689000003*theta_p.^2 - 0.6115448939999998*theta_p.^3)...
                      ./ (1 - 5.661479886999997*theta_p + 11.47636191*theta_p.^2 ...
                        - 9.82431213599998*theta_p.^3 + 3.048755063*theta_p.^4);

% maximum solid-phase concentration [mol/m^3]
p.cs_max_neg = 30555;
p.cs_max_pos = 51554;

%% Activation energy [J/mol]
p.Ea_Ds_neg = 42.77e3;  % activation energy for negative electrode solid-phase diffusion [J/mol]
p.Ea_Ds_pos = 18.55e3;  % activation energy for positive electrode solid-phase diffusion [J/mol]

p.Ea_k_neg = 39.57e3;   % activation energy for negative electrode reaction constant [J/mol]
p.Ea_k_pos = 37.48e3;   % activation energy for positive electrode reaction constant [J/mol]

p.Ea_De = 37.04e3;      % activation energy for electrolyte diffusion [J/mol]
p.Ea_kappa = 34.70e3;   % activation energy for electrolyte conductivity [J/mol]

p.Ea_rou_sei = 33.26e3; % activation energy for negative electrode sei resistivity [J/mol]

end
