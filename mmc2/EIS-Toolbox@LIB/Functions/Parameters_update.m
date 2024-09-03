
% Parameters_update calculates extra parameters.

% Inputs:
%    - p is a MATLAB struct containing the initial parameters.
%    - T is the ambient temperature, K.
%    - SOC is the state of charge (SOC) of the full cell in the present work, dimensionless.

% Outputs:
%    - p is a MATLAB struct containing the initial parameters and the calculated extra parameters.

function p = Parameters_update(p, T, SOC)

p.T = T;
p.SOC = SOC;

% the parameters vary with temperature via the Arrhenius relationship:
p.Ds_neg = p.Ds_neg_0 * exp( - p.Ea_Ds_neg / p.R * (1/p.T - 1/p.T_0));
p.Ds_pos = p.Ds_pos_0 * exp( - p.Ea_Ds_pos / p.R * (1/p.T - 1/p.T_0));

p.k_neg = p.k_neg_0 * exp( - p.Ea_k_neg / p.R * (1/p.T - 1/p.T_0));
p.k_pos = p.k_pos_0 * exp( - p.Ea_k_pos / p.R * (1/p.T - 1/p.T_0));

p.De = p.De_0 * exp( - p.Ea_De / p.R * (1/p.T - 1/p.T_0));
p.kappa = p.kappa_0 * exp( - p.Ea_kappa / p.R * (1/p.T - 1/p.T_0));

p.rou_sei_neg = p.rou_sei_neg_0 * exp( - p.Ea_rou_sei / p.R * (1/p.T - 1/p.T_0));

syms theta_n theta_p
p.U_neg = p.U_neg_0(theta_n) + (p.T-p.T_0) * p.dUdT_neg(theta_n);
p.U_pos = p.U_pos_0(theta_p) + (p.T-p.T_0) * p.dUdT_pos(theta_p);

% sei resistance [Ωm^2]
p.Rsei_neg = p.rou_sei_neg * (p.rs_neg * p.delta_sei_neg) / (p.rs_neg + p.delta_sei_neg);
p.Rsei_pos = 0;

% sei capacitance [F/m^2]
p.Csei_neg = p.epse_sei_neg * (p.rs_neg + p.delta_sei_neg) / (p.rs_neg * p.delta_sei_neg);
p.Csei_pos = 0;

% initial stoichiometric coefficient for porous electrode [-]
theta_neg_0 = p.s0_neg + p.SOC * (p.s100_neg - p.s0_neg);
theta_pos_0 = p.s0_pos + p.SOC * (p.s100_pos - p.s0_pos);

% first derivative of open-circuit potential at the electrode [V·m^3/mol]
p.dUdc_neg = double(subs(diff(p.U_neg,theta_n), theta_n, theta_neg_0)/p.cs_max_neg);
p.dUdc_pos = double(subs(diff(p.U_pos,theta_p), theta_p, theta_pos_0)/p.cs_max_pos);

% initial concentration of lithium in the porous electrode [mol/m^3]
p.cs0_neg = p.cs_max_neg * theta_neg_0;
p.cs0_pos = p.cs_max_pos * theta_pos_0;

% exchange current density of the Faradaic reactions (for electrode) [A/m^2]
p.i0_neg = p.F * p.k_neg * p.ce_0^p.alpha_a_neg * (p.cs_max_neg-p.cs0_neg)^p.alpha_a_neg * p.cs0_neg^p.alpha_c_neg;
p.i0_pos = p.F * p.k_pos * p.ce_0^p.alpha_a_pos * (p.cs_max_pos-p.cs0_pos)^p.alpha_a_pos * p.cs0_pos^p.alpha_c_pos;

% ε_s: active material volume fraction at the electrodes [-]
p.epss_neg = 1 - p.epse_neg - p.epsf_neg;
p.epss_pos = 1 - p.epse_pos - p.epsf_pos;

% specific surface area of the porous electrode [m^2/m^3]
p.as_neg = 3 * p.epss_neg / p.rs_neg;
p.as_pos = 3 * p.epss_pos / p.rs_pos;

% effective electrolyte diffusivity [m^2/s]
p.De_eff_neg = p.De * p.epse_neg ^ p.brug_neg;
p.De_eff_pos = p.De * p.epse_pos ^ p.brug_pos;
p.De_eff_sep = p.De * p.epse_sep ^ p.brug_sep;

% κ_eff: effective electrolyte conductivity [S/m]
p.kappa_eff_neg = p.kappa * p.epse_neg ^ p.brug_neg;
p.kappa_eff_pos = p.kappa * p.epse_pos ^ p.brug_pos;
p.kappa_eff_sep = p.kappa * p.epse_sep ^ p.brug_sep;

% κ_D,eff: effective diffusional conductivity [A/m]
p.kappa_D_eff_neg = 2 * p.R * p.T / p.F * p.kappa_eff_neg * (p.t_plus-1) * (1+p.dlnf_ce);
p.kappa_D_eff_pos = 2 * p.R * p.T / p.F * p.kappa_eff_pos * (p.t_plus-1) * (1+p.dlnf_ce);
p.kappa_D_eff_sep = 2 * p.R * p.T / p.F * p.kappa_eff_sep * (p.t_plus-1) * (1+p.dlnf_ce);

% σ_eff: effective solid conductivity [S/m]
p.sigma_eff_neg = p.sigma_neg * p.epss_neg;
p.sigma_eff_pos = p.sigma_pos * p.epss_pos;

% charge transfer resistance at the interface of the particle/electrolyte [Ω·m^2]
p.Rct_neg = p.R * p.T / (p.i0_neg * (p.alpha_a_neg + p.alpha_c_neg) * p.F);
p.Rct_pos = p.R * p.T / (p.i0_pos * (p.alpha_a_pos + p.alpha_c_pos) * p.F);

% the solid-phase diffusion resistance in the particle [Ω·m^2]
p.Rdiff_neg = - p.rs_neg / (p.Ds_neg * p.F) * p.dUdc_neg;
p.Rdiff_pos = - p.rs_pos / (p.Ds_pos * p.F) * p.dUdc_pos;

end
