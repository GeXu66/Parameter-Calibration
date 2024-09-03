%% Script description
% Script used to obtain impedance datas shown in Figure 8.

%% clear past variables and plots
clc; clear; close all
addpath('Functions')          % add path
addpath('Functions\plot')     % add path

%% settings for parametric studies
change = 'De_eff';
Multiple(:,1) = [1 2 12 1e3];  % De,eff

%% simulation frequency and initial parameters
f(:,1) = logspace(-3,5,100);  % frequency [Hz]

p = Parameters_initialize;    % the parameters used in simulation
T = p.T_0;                    % ambient temperature [K]
SOC = 1;                      % the state of charge (SOC) of the full cell in present work [-]

%% impedance simulation
p = Parameters_update(p, T, SOC);
De_eff_neg = p.De_eff_neg * Multiple;   % settings for parametric studies
De_eff_pos = p.De_eff_pos * Multiple;   % settings for parametric studies
for m = 1:length(Multiple)
    p.De_eff_neg = De_eff_neg(m);       % settings for parametric studies
    p.De_eff_pos = De_eff_pos(m);       % settings for parametric studies
    out{m,1} = EIS(p, f);

    Pi_III_neg(m,1) = - p.kappa_D_eff_neg / p.De_eff_neg / p.ce_0 / p.F;   % similar to Equation (63)
    Pi_III_pos(m,1) = - p.kappa_D_eff_pos / p.De_eff_pos / p.ce_0 / p.F;   % similar to Equation (63)
end
clc; clear p f m

%% make plot
Origin = polt_model_simplify(out, Multiple, change);
