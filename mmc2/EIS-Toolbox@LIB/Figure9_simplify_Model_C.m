%% Script description
% Script used to obtain impedance datas shown in Figure 9.

%% clear past variables and plots
clc; clear; close all
addpath('Functions')          % add path
addpath('Functions\plot')     % add path

%% settings for parametric studies
change = 'sigma_eff';
Multiple(:,1) = [1 3 30 1e3];  % De,eff

%% simulation frequency and initial parameters
f(:,1) = logspace(-3,5,100);  % frequency [Hz]

p = Parameters_initialize;    % the parameters used in simulation
T = p.T_0;                    % ambient temperature [K]
SOC = 1;                      % the state of charge (SOC) of the full cell in present work [-]

%% impedance simulation
p = Parameters_update(p, T, SOC);
sigma_eff_neg = p.sigma_eff_neg * Multiple;   % settings for parametric studies
sigma_eff_pos = p.sigma_eff_pos * Multiple;   % settings for parametric studies
for m = 1:length(Multiple)
    p.sigma_eff_neg = sigma_eff_neg(m);       % settings for parametric studies
    p.sigma_eff_pos = sigma_eff_pos(m);       % settings for parametric studies
    out{m,1} = EIS(p, f);

    Beta_neg(m,1) = p.kappa_eff_neg / p.sigma_eff_neg;
    Beta_pos(m,1) = p.kappa_eff_pos / p.sigma_eff_pos;
end
clc; clear p f m

%% make plot
Origin = polt_model_simplify(out, Multiple, change);
