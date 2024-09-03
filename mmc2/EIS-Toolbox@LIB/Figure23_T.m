%% Script description
% Script used to obtain impedance datas shown in Figures 23 and 25.

%% clear past variables and plots
clc; clear; close all
addpath('Functions')          % add path
addpath('Functions\plot')     % add path

%% settings for parametric studies
change = 'T';
Multiple(:,1) = [-10 -10 0 10 20];  % T

%% simulation frequency and initial parameters
f(:,1) = logspace(-3,5,100);  % frequency [Hz]

p = Parameters_initialize;    % the parameters used in simulation
SOC = 1;                      % the state of charge (SOC) of the full cell in present work [-]

%% impedance simulation
T = p.T_0 + Multiple;         % settings for parametric studies
for m = 1:length(Multiple)
    p = Parameters_update(p, T(m), SOC);
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
Origin = polt_multi_scale(out, Multiple, change);
Origin{4,1} = polt_diffusion_impedance(out, Multiple, change);
