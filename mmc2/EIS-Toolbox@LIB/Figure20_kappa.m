%% Script description
% Script used to obtain impedance datas shown in Figures 20 and 22(c and d).

%% clear past variables and plots
clc; clear; close all
addpath('Functions')          % add path
addpath('Functions\plot')     % add path

%% settings for parametric studies
change = 'kappa';
Multiple(:,1) = [0.7 0.7:0.3:1.6];  % κ

%% simulation frequency and initial parameters
f(:,1) = logspace(-3,5,100);  % frequency [Hz]

p = Parameters_initialize;    % the parameters used in simulation
T = p.T_0;                    % ambient temperature [K]
SOC = 1;                      % the state of charge (SOC) of the full cell in present work [-]

%% impedance simulation
kappa_0 = p.kappa_0 * Multiple;   % settings for parametric studies
for m = 1:length(Multiple)
    p.kappa_0 = kappa_0(m);       % settings for parametric studies
    p.De_0 = 2 * p.R * p.T_0 / p.F^2 / p.ce_0 * (1+p.dlnf_ce) * p.t_plus * (1-p.t_plus) * p.kappa_0;   % settings for parametric studies
    p = Parameters_update(p, T, SOC);
    out{m,1} = EIS(p, f);
end
clc; clear p f m

%% Sensitivity analysis
for n = 1:length(out)
    Z_Ds(:,n) = out{n,1}.ZD.Ds_neg;
    Z_De(:,n) = out{n,1}.ZD.De_neg;
end
clear n

SD = [Sensitivity_calculate(Z_Ds) Sensitivity_calculate(Z_De)];

%% make plot
Origin = polt_electrode_scale(out, Multiple, change);
Origin{4,1} = polt_diffusion_impedance(out, Multiple, change);
