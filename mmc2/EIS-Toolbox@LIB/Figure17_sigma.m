%% Script description
% Script used to obtain impedance datas shown in Figures 17 and 18(e and f).

%% clear past variables and plots
clc; clear; close all
addpath('Functions')          % add path
addpath('Functions\plot')     % add path

%% settings for parametric studies
change = 'sigma_neg';
Multiple(:,1) = [0.6 0.6 1 1.8 3.2];  % Ds, k, σ, De

%% simulation frequency and initial parameters
f(:,1) = logspace(-3,5,100);  % frequency [Hz]

p = Parameters_initialize;    % the parameters used in simulation
T = p.T_0;                    % ambient temperature [K]
SOC = 1;                      % the state of charge (SOC) of the full cell in present work [-]

%% impedance simulation
sigma_neg = p.sigma_neg * Multiple;   % settings for parametric studies
sigma_pos = p.sigma_pos * Multiple;   % settings for parametric studies
for m = 1:length(Multiple)
    p.sigma_neg = sigma_neg(m);       % settings for parametric studies
    p.sigma_pos = sigma_pos(m);       % settings for parametric studies
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
