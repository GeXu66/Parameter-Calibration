%% Script description
% Script used to obtain impedance datas shown in Figures 15 and 18(a and b).

%% clear past variables and plots
clc; clear; close all
addpath('Functions')          % add path
addpath('Functions\plot')     % add path

%% settings for parametric studies
change = 'L_neg';
Multiple(:,1) = [0.8 0.8:0.2:1.4];  % L

%% simulation frequency and initial parameters
f(:,1) = logspace(-3,5,100);  % frequency [Hz]

p = Parameters_initialize;    % the parameters used in simulation
T = p.T_0;                    % ambient temperature [K]
SOC = 1;                      % the state of charge (SOC) of the full cell in present work [-]

%% impedance simulation
L_neg = p.L_neg * Multiple;   % settings for parametric studies
L_pos = p.L_pos * Multiple;   % settings for parametric studies
L_sep = p.L_sep * Multiple;   % settings for parametric studies
for m = 1:length(Multiple)
    p.L_neg = L_neg(m);       % settings for parametric studies
    p.L_pos = L_pos(m);       % settings for parametric studies
    p.L_sep = L_sep(m);       % settings for parametric studies
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
