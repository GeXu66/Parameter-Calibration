%% Script description
% Script used to obtain impedance datas shown in Figures 2, 5, and 6 in [1].

% Author: Yuxuan Bai; Qiu-An Huang
% Affliction: Shanghai University, China
% Email address: yuxuan_bai@shu.edu.cn; hqahqahqa@163.com
% Created December 2, 2023

% References:
% [1] Bai et al., Decouple charge transfer reactions in the Li-ion battery,
% submitted to Journal of Energy Chemistry, 2024

% Doyle-Fuller-Newman (DFN) model:
% [2] Marc Doyle et al., Modeling of galvanostatic charge and discharge 
% of the lithium/polymer/insertion cell,
% Journal of The Electrochemical Society, 140 (6) 1526-1533 (1993)
% [3] Thomas F. Fuller et al., Simulation and optimization of the dual
% lithium ion insertion cell,
% Journal of The Electrochemical Society, 141 (1) 1-10 (1994)
% [4] Marc Doyle et al., Comparison of modeling predictions
% with experimental data from plastic lithium ion cells,
% Journal of The Electrochemical Society, 143 (6) 1890-1903 (1996)

% DFN-like impedance models:
% [5] Godfrey Sikha et al., Analytical expression
% for the impedance response for a lithium-ion cell,
% Journal of The Electrochemical Society, 155 (12) A893-A902 (2008)
% [6] Jun Huang et al., Theory of impedance response of porous electrodes:
% simplifications, inhomogeneities, non-stationarities and applications,
% Journal of The Electrochemical Society, 163 (9) A1983-A2000 (2016)
% [7] Jeremy P. Meyers et al., The impedance response of
% a porous electrode composed of intercalation particles,
% Journal of The Electrochemical Society, 147 (8) 2930-2940 (2000)
% [8] Marvin Cronau et al., Thickness-dependent impedance of composite
% battery electrodes containing ionic liquid-based electrolytes,
% Batteries & Supercaps, 3 (7) 611-618 (2020)
% [9] Sheba Devan et al., Analytical solution for the impedance of a porous electrode,
% Journal of The Electrochemical Society, 151 (6) A905-A913 (2004)
% [10] G. Paasch et al., Theory of the electrochemical impedance
% of macrohomogeneous porous electrodes,
% Electrochimica Acta, 38 (18) 2653-2662 (1993)
% [11] R De Levie, On porous electrodes in electrolyte solutions—IV,
% Electrochimica Acta, 9 (9) 1231-1245 (1964)

% Parameter:
% [12] Torchio et al., LIONSIMBA: a Matlab framework based on a finite volume
% model suitable for li-ion battery design, simulation, and control,
% Journal of The Electrochemical Society, 163 (7) A1192-A1205 (2016)
% [13] Zhang et al., Multi-objective optimization of lithium-ion battery
% model using genetic algorithm approach,
% Journal of Power Sources 270 367-378 (2014)

%% clear past variables and plots
clc; clear; close all
addpath('Functions')          % add path
addpath('Functions\plot')     % add path

%% simulation frequency and initial parameters
f(:,1) = logspace(-3,5,100);  % frequency [Hz]

p = Parameters_initialize;    % the parameters used in simulation
T = p.T_0;                    % ambient temperature [K]
SOC = 1;                      % the state of charge (SOC) of the full cell in present work [-]

%% impedance simulation
p = Parameters_update(p, T, SOC);
out = EIS(p, f);
clear p f

%% make plot
Origin = plot_default(out);
