%% Script description
% Script used to obtain impedance datas shown in Figures 16 and 18(c and d).

%% clear past variables and plots
clc; clear; close all
addpath('Functions')          % add path
addpath('Functions\plot')     % add path

%% settings for parametric studies
change = 'epse_neg';
Multiple(:,1) = [0.8 0.8:0.1:1.1];  % εe

%% simulation frequency and initial parameters
f(:,1) = logspace(-3,5,100);  % frequency [Hz]

p = Parameters_initialize;    % the parameters used in simulation
T = p.T_0;                    % ambient temperature [K]
SOC = 1;                      % the state of charge (SOC) of the full cell in present work [-]

%% impedance simulation
epse_neg = p.epse_neg * Multiple;   % settings for parametric studies
epse_pos = p.epse_pos * Multiple;   % settings for parametric studies
epse_sep = p.epse_sep * Multiple;   % settings for parametric studies
for m = 1:length(Multiple)
    p.epse_neg = epse_neg(m);       % settings for parametric studies
    p.epse_pos = epse_pos(m);       % settings for parametric studies
    p.epse_sep = epse_sep(m);       % settings for parametric studies
    p.epss_neg = (1-p.epse_neg)*0.4824/(0.4824+0.0326);  % settings for parametric studies
    p.epsf_neg = 1-p.epse_neg-p.epss_neg;                % settings for parametric studies
    p.epss_pos = (1-p.epse_pos)*0.59/(0.59+0.025);       % settings for parametric studies
    p.epsf_pos = 1-p.epse_pos-p.epss_pos;                % settings for parametric studies

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
