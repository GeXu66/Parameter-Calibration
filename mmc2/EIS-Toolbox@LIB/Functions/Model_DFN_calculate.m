
% Model_DFN_calculate calculates the impedance of Model DFN for Li-ion batteries.

% Inputs:
%    - p is a MATLAB struct containing the initial parameters and the calculated extra parameters.
%    - w is a MATLAB struct containing the frequency variable s = j*2*π*f and the calculated related variables.
%    - z is a MATLAB struct containing three particle impedances in the single particle model, 
%      such as solid-diffusion impedance z_d, the faradaic behavior impedance z_F, and the interfacial impedance z_int.

% Outputs:
%    - Var is a MATLAB struct containing all internal cell electrochemical variables,
%      such as electrolyte concentration C_e(x,s), solid-electrolyte potential difference Φ_s-e(x,s), reaction flux J(x,s),
%      solid particle surface concentration C_ss(x,s), solid potential Φ_s(x,s), and electrolyte variables Φ_e(x,s).
%    - Z is a MATLAB struct containing the impedance of individual components in the full cell,
%      such as the negative electrode impedance Z_1(s), the positive electrode impedance Z_2(s),
%      the separator impedance Z_3(s), and the full cell impedance Z_4(s).

function [Var, Z] = Model_DFN_calculate(p, w, z)
tic()

%% Frequency domain variables
% dimensionless angular frequency [-]
w.s_neg = p.epse_neg * p.L_neg^2 / p.De_eff_neg * w.s;   % Equation (33-4)
w.s_pos = p.epse_pos * p.L_pos^2 / p.De_eff_pos * w.s;   % Equation (33-4)
w.s_sep = p.epse_sep * p.L_sep^2 / p.De_eff_sep * w.s;   % Equation (36-4)

% Θ_I: the ratio of the electrolyte-related diffusion resistance to z_int [-]
w.Pi_I_neg = - p.kappa_D_eff_neg / p.kappa_eff_neg / p.ce_0 * (1-p.t_plus)...
             * p.as_neg * p.L_neg^2 / p.F / p.De_eff_neg ./ z.zint_neg;   % Equation (59-1)
w.Pi_I_pos = - p.kappa_D_eff_pos / p.kappa_eff_pos / p.ce_0 * (1-p.t_plus)...
             * p.as_pos * p.L_pos^2 / p.F / p.De_eff_pos ./ z.zint_pos;   % Equation (62-1)

% Θ_II: the ratio of the whole electrode-related ohmic resistance to z_int [-]
w.Pi_II_neg = p.as_neg * p.L_neg^2 * (1+p.sigma_eff_neg/p.kappa_eff_neg) / p.sigma_eff_neg ./ z.zint_neg;   % Equation (59-2)
w.Pi_II_pos = p.as_pos * p.L_pos^2 * (1+p.sigma_eff_pos/p.kappa_eff_pos) / p.sigma_eff_pos ./ z.zint_pos;   % Equation (62-2)

% Θ_III: dimensionless parameter is defined in the separator region [-]
w.Pi_III_sep = - p.kappa_D_eff_sep / p.ce_0 / p.De_eff_sep / p.F;   % Equation (63)

% λ: eigenvalues [-]
w.lamda_I_neg = 1/2 * (w.s_neg + w.Pi_I_neg + w.Pi_II_neg...
    + sqrt(w.s_neg.^2 + 2*w.s_neg.*(w.Pi_I_neg-w.Pi_II_neg) + (w.Pi_I_neg+w.Pi_II_neg).^2));   % Equation (59-3)
w.lamda_II_neg = 1/2 * (w.s_neg + w.Pi_I_neg + w.Pi_II_neg...
    - sqrt(w.s_neg.^2 + 2*w.s_neg.*(w.Pi_I_neg-w.Pi_II_neg) + (w.Pi_I_neg+w.Pi_II_neg).^2));   % Equation (59-4)
w.lamda_I_pos = 1/2 * (w.s_pos + w.Pi_I_pos + w.Pi_II_pos...
    + sqrt(w.s_pos.^2 + 2*w.s_pos.*(w.Pi_I_pos-w.Pi_II_pos) + (w.Pi_I_pos+w.Pi_II_pos).^2));   % Equation (62-3)
w.lamda_II_pos = 1/2 * (w.s_pos + w.Pi_I_pos + w.Pi_II_pos...
    - sqrt(w.s_pos.^2 + 2*w.s_pos.*(w.Pi_I_pos-w.Pi_II_pos) + (w.Pi_I_pos+w.Pi_II_pos).^2));   % Equation (62-4)

% Λ_III: coefficients [s/m]
w.Lambda_III_sep = p.L_sep / p.De_eff_sep ./ sqrt(w.s_sep) .* csch(sqrt(w.s_sep));   % Equation (60-5)
% Λ_I: coefficients [mol/m^3/(A/m^2)]
w.Lambda_I_neg = - p.L_neg^3 * p.as_neg * (1-p.t_plus) / p.F ./ z.zint_neg...
        / p.De_eff_neg / p.sigma_eff_neg ./ (w.lamda_I_neg-w.lamda_II_neg)...
     .* ((csch(sqrt(w.lamda_II_neg)) ./ sqrt(w.lamda_II_neg) - csch(sqrt(w.lamda_I_neg)) ./ sqrt(w.lamda_I_neg))...
        + p.sigma_eff_neg/p.kappa_eff_neg...
       * (1 ./ sqrt(w.lamda_II_neg) ./ tanh(sqrt(w.lamda_II_neg))...
        - 1 ./ sqrt(w.lamda_I_neg) ./ tanh(sqrt(w.lamda_I_neg))));                   % Equation (60-1)
w.Lambda_II_neg = p.L_neg / p.De_eff_neg ./ (w.lamda_I_neg-w.lamda_II_neg) .*...
                ((w.s_neg + w.Pi_I_neg - w.lamda_II_neg) ./ sqrt(w.lamda_I_neg) ./ tanh(sqrt(w.lamda_I_neg))...
               - (w.s_neg + w.Pi_I_neg - w.lamda_I_neg) ./ sqrt(w.lamda_II_neg) ./ tanh(sqrt(w.lamda_II_neg)))...
                + p.L_sep / p.De_eff_sep ./ sqrt(w.s_sep) ./ tanh(sqrt(w.s_sep));    % Equation (60-2)
% Λ_II: coefficients [s/m]
w.Lambda_I_pos = - p.L_pos^3 * p.as_pos * (1-p.t_plus) / p.F ./ z.zint_pos...
        / p.De_eff_pos / p.sigma_eff_pos ./ (w.lamda_I_pos-w.lamda_II_pos)...
     .* ((csch(sqrt(w.lamda_II_pos)) ./ sqrt(w.lamda_II_pos) - csch(sqrt(w.lamda_I_pos)) ./ sqrt(w.lamda_I_pos))...
        + p.sigma_eff_pos/p.kappa_eff_pos...
       * (1 ./ sqrt(w.lamda_II_pos) ./ tanh(sqrt(w.lamda_II_pos))...
        - 1 ./ sqrt(w.lamda_I_pos) ./ tanh(sqrt(w.lamda_I_pos))));                   % Equation (60-3)
w.Lambda_II_pos = p.L_pos / p.De_eff_pos ./ (w.lamda_I_pos-w.lamda_II_pos) .*...
                ((w.s_pos + w.Pi_I_pos - w.lamda_II_pos) ./ sqrt(w.lamda_I_pos) ./ tanh(sqrt(w.lamda_I_pos))...
               - (w.s_pos + w.Pi_I_pos - w.lamda_I_pos) ./ sqrt(w.lamda_II_pos) ./ tanh(sqrt(w.lamda_II_pos)))...
                + p.L_sep / p.De_eff_sep ./ sqrt(w.s_sep) ./ tanh(sqrt(w.s_sep));    % Equation (60-4)

% ξ/I: the concentration flux transfer function at the electrode/separator interface [mol/m^2/s/(A/m^2)]
w.zeta_neg = (w.Lambda_I_neg + w.Lambda_I_pos ./ w.Lambda_II_pos .* w.Lambda_III_sep)...
          ./ (w.Lambda_II_neg - w.Lambda_III_sep.^2 ./ w.Lambda_II_pos);   % Equation (59-5)
w.zeta_pos = (w.Lambda_I_pos + w.Lambda_I_neg ./ w.Lambda_II_neg .* w.Lambda_III_sep)...
          ./ (w.Lambda_II_pos - w.Lambda_III_sep.^2 ./ w.Lambda_II_neg);   % Equation (62-5)

% the expressions for the coefficients in the negative electrode region [-]
w.B_I_neg = w.Pi_I_neg ./ sqrt(w.lamda_I_neg) ./ (w.lamda_I_neg-w.lamda_II_neg) ./ tanh(sqrt(w.lamda_I_neg))...
          + w.Pi_I_neg ./ sqrt(w.lamda_I_neg) ./ (w.lamda_I_neg-w.lamda_II_neg) .* csch(sqrt(w.lamda_I_neg))...
        .* (p.sigma_eff_neg/p.kappa_eff_neg...
         - (w.s_neg + w.Pi_I_neg - w.lamda_II_neg) * p.sigma_eff_neg / p.L_neg^2 / p.as_neg / (1-p.t_plus)...
          * p.F .* z.zint_neg .* w.zeta_neg);                                           % Equation (58-3)
w.B_II_neg = - w.Pi_I_neg ./ sqrt(w.lamda_I_neg) ./ (w.lamda_I_neg - w.lamda_II_neg);   % Equation (58-4)

w.B_III_neg = - w.Pi_I_neg ./ sqrt(w.lamda_II_neg) ./ (w.lamda_I_neg-w.lamda_II_neg) ./ tanh(sqrt(w.lamda_II_neg))...
              - w.Pi_I_neg ./ sqrt(w.lamda_II_neg) ./ (w.lamda_I_neg-w.lamda_II_neg) .* csch(sqrt(w.lamda_II_neg))...
            .* (p.sigma_eff_neg/p.kappa_eff_neg...
             - (w.s_neg + w.Pi_I_neg - w.lamda_I_neg) * p.sigma_eff_neg / p.L_neg^2 / p.as_neg / (1-p.t_plus)...
              * p.F .* z.zint_neg .* w.zeta_neg);                                       % Equation (58-5)
w.B_IV_neg = w.Pi_I_neg ./ sqrt(w.lamda_II_neg) ./ (w.lamda_I_neg - w.lamda_II_neg);    % Equation (58-6)

% the expressions for the coefficients in the positive electrode region [-]
w.B_II_pos = w.Pi_I_pos ./ sqrt(w.lamda_I_pos) ./ (w.lamda_I_pos-w.lamda_II_pos) .* (p.sigma_eff_pos/p.kappa_eff_pos...
          - (w.s_pos + w.Pi_I_pos - w.lamda_II_pos) * p.sigma_eff_pos / p.L_pos^2 / p.as_pos / (1-p.t_plus)...
           * p.F .* z.zint_pos .* w.zeta_pos);            % Equation (61-4)
w.B_I_pos = - w.Pi_I_pos ./ sqrt(w.lamda_I_pos) ./ (w.lamda_I_pos-w.lamda_II_pos) .* csch(sqrt(w.lamda_I_pos))...
            - w.B_II_pos ./ tanh(sqrt(w.lamda_I_pos));    % Equation (61-3)

w.B_IV_pos = - w.Pi_I_pos ./ sqrt(w.lamda_II_pos) ./ (w.lamda_I_pos-w.lamda_II_pos) .* (p.sigma_eff_pos/p.kappa_eff_pos...
            - (w.s_pos + w.Pi_I_pos - w.lamda_I_pos) * p.sigma_eff_pos / p.L_pos^2 / p.as_pos / (1-p.t_plus)...
             * p.F .* z.zint_pos .* w.zeta_pos);          % Equation (61-6)
w.B_III_pos = w.Pi_I_pos ./ sqrt(w.lamda_II_pos) ./ (w.lamda_I_pos-w.lamda_II_pos) .* csch(sqrt(w.lamda_II_pos))...
            - w.B_IV_pos ./ tanh(sqrt(w.lamda_II_pos));   % Equation (61-5)

% the expressions for the coefficients in the separator region [-]
w.B_I_sep = w.Pi_III_sep * p.F ./ sqrt(w.s_sep) .* csch(sqrt(w.s_sep)) .* w.zeta_pos...   % Equation (65-2)
          - w.Pi_III_sep * p.F ./ sqrt(w.s_sep) ./ tanh(sqrt(w.s_sep)) .* w.zeta_neg;
w.B_II_sep = w.Pi_III_sep * p.F ./ sqrt(w.s_sep) .* w.zeta_neg;                           % Equation (65-3)

% Γ_I  = B_I   * cosh(sqrt(λ_I))  + B_II * sinh(sqrt(λ_I))
% Γ_II = B_III * cosh(sqrt(λ_II)) + B_IV * sinh(sqrt(λ_II))
w.chi_I_neg = w.Pi_I_neg ./ sqrt(w.lamda_I_neg) ./ (w.lamda_I_neg-w.lamda_II_neg) .* csch(sqrt(w.lamda_I_neg))...
            + w.Pi_I_neg ./ sqrt(w.lamda_I_neg) ./ (w.lamda_I_neg-w.lamda_II_neg) ./ tanh(sqrt(w.lamda_I_neg))...
          .* (p.sigma_eff_neg/p.kappa_eff_neg...           % Equation (58-1)
           - (w.s_neg + w.Pi_I_neg - w.lamda_II_neg) * p.sigma_eff_neg / p.L_neg^2 / p.as_neg / (1-p.t_plus)...
            * p.F .* z.zint_neg .* w.zeta_neg);
w.chi_II_neg = - w.Pi_I_neg ./ sqrt(w.lamda_II_neg) ./ (w.lamda_I_neg-w.lamda_II_neg) .* csch(sqrt(w.lamda_II_neg))...
               - w.Pi_I_neg ./ sqrt(w.lamda_II_neg) ./ (w.lamda_I_neg-w.lamda_II_neg) ./ tanh(sqrt(w.lamda_II_neg))...
             .* (p.sigma_eff_neg/p.kappa_eff_neg...
              - (w.s_neg + w.Pi_I_neg - w.lamda_I_neg) * p.sigma_eff_neg / p.L_neg^2 / p.as_neg / (1-p.t_plus)...
               * p.F .* z.zint_neg .* w.zeta_neg);         % Equation (58-2)

w.chi_I_pos = - w.Pi_I_pos ./ sqrt(w.lamda_I_pos) ./ (w.lamda_I_pos-w.lamda_II_pos) ./ tanh(sqrt(w.lamda_I_pos))...
              - w.B_II_pos .* csch(sqrt(w.lamda_I_pos));   % Equation (61-1)
w.chi_II_pos = w.Pi_I_pos ./ sqrt(w.lamda_II_pos) ./ (w.lamda_I_pos-w.lamda_II_pos) ./ tanh(sqrt(w.lamda_II_pos))...
             - w.B_IV_pos .* csch(sqrt(w.lamda_II_pos));   % Equation (61-2)

% Γ_III = B_I * cosh(sqrt(s_sep)) + B_II * sinh(sqrt(s_sep))
w.chi_III_sep = w.Pi_III_sep * p.F ./ sqrt(w.s_sep) ./ tanh(sqrt(w.s_sep)) .* w.zeta_pos...
              - w.Pi_III_sep * p.F ./ sqrt(w.s_sep) .* csch(sqrt(w.s_sep)) .* w.zeta_neg;   % Equation (65-1)

% the expressions for the coefficients in the negative electrode region [V/(A/m^2)]
w.H_I_neg = - p.L_neg / (p.sigma_eff_neg+p.kappa_eff_neg);                                % Equation (64-1)
w.H_II_neg = - p.L_neg / p.sigma_eff_neg ./ w.Pi_I_neg...
          .* ((w.s_neg - w.lamda_I_neg) .* w.chi_I_neg...
            + (w.s_neg - w.lamda_II_neg) .* w.chi_II_neg)...
             + p.L_neg^3 * p.as_neg / p.sigma_eff_neg^2 ./ z.zint_neg ./ w.Pi_I_neg...
          .* ((w.s_neg - w.lamda_I_neg) ./ w.lamda_I_neg .* w.chi_I_neg...
            + (w.s_neg - w.lamda_II_neg) ./ w.lamda_II_neg .* w.chi_II_neg)...
             - w.H_I_neg + p.L_sep / p.kappa_eff_sep * (1 + w.B_I_sep - w.chi_III_sep);   % Equation (64-2)

% the expressions for the coefficients in the positive electrode region [V/(A/m^2)]
w.H_I_pos = - p.L_pos / (p.sigma_eff_pos+p.kappa_eff_pos);                                % Equation (64-3)
w.H_II_pos = - p.L_pos / p.sigma_eff_pos ./ w.Pi_I_pos...
          .* ((w.s_pos - w.lamda_I_pos) .* w.B_I_pos...
            + (w.s_pos - w.lamda_II_pos) .* w.B_III_pos)...
             + p.L_pos^3 * p.as_pos / p.sigma_eff_pos^2 ./ z.zint_pos ./ w.Pi_I_pos...
          .* ((w.s_pos - w.lamda_I_pos) ./ w.lamda_I_pos .* w.B_I_pos...
            + (w.s_pos - w.lamda_II_pos) ./ w.lamda_II_pos .* w.B_III_pos);               % Equation (64-4)

%% 1. Electrolyte-concentration transfer function [mol/m^3/(A/m^2)]
Var.Ce_neg(:,1) = p.kappa_eff_neg * p.ce_0 / p.kappa_D_eff_neg * p.L_neg / p.sigma_eff_neg * (w.B_I_neg + w.B_III_neg);      % Equation (42-1)
Var.Ce_neg(:,2) = p.kappa_eff_neg * p.ce_0 / p.kappa_D_eff_neg * p.L_neg / p.sigma_eff_neg * (w.chi_I_neg + w.chi_II_neg);   % Equation (42-1)

Var.Ce_sep(:,1) = - p.L_sep * p.ce_0 / p.kappa_D_eff_sep * w.B_I_sep;       % Equation (45)
Var.Ce_sep(:,2) = - p.L_sep * p.ce_0 / p.kappa_D_eff_sep * w.chi_III_sep;   % Equation (45)

Var.Ce_pos(:,1) = p.kappa_eff_pos * p.ce_0 / p.kappa_D_eff_pos * p.L_pos / p.sigma_eff_pos * (w.B_I_pos + w.B_III_pos);      % Equation (42-5)
Var.Ce_pos(:,2) = p.kappa_eff_pos * p.ce_0 / p.kappa_D_eff_pos * p.L_pos / p.sigma_eff_pos * (w.chi_I_pos + w.chi_II_pos);   % Equation (42-5)

%% 2. Solid–electrolyte potential difference transfer function [V/(A/m^2)]
Var.Phise_neg(:,1) = - p.L_neg / p.sigma_eff_neg * ((w.s_neg-w.lamda_I_neg) ./ w.Pi_I_neg .* w.B_I_neg...
                                                  + (w.s_neg-w.lamda_II_neg) ./ w.Pi_I_neg .* w.B_III_neg);    % Equation (51-1)
Var.Phise_neg(:,2) = - p.L_neg / p.sigma_eff_neg * ((w.s_neg-w.lamda_I_neg) ./ w.Pi_I_neg .* w.chi_I_neg...
                                                  + (w.s_neg-w.lamda_II_neg) ./ w.Pi_I_neg .* w.chi_II_neg);   % Equation (51-1)

Var.Phise_pos(:,1) = - p.L_pos / p.sigma_eff_pos * ((w.s_pos-w.lamda_I_pos) ./ w.Pi_I_pos .* w.B_I_pos...
                                                  + (w.s_pos-w.lamda_II_pos) ./ w.Pi_I_pos .* w.B_III_pos);    % Equation (51-2)
Var.Phise_pos(:,2) = - p.L_pos / p.sigma_eff_pos * ((w.s_pos-w.lamda_I_pos) ./ w.Pi_I_pos .* w.chi_I_pos...
                                                  + (w.s_pos-w.lamda_II_pos) ./ w.Pi_I_pos .* w.chi_II_pos);   % Equation (51-2)

%% 3. Reaction flux transfer function [mol/m^2/s/(A/m^2)]
Var.J_neg = Var.Phise_neg / p.F ./ z.zint_neg;   % Equation (23-3)
Var.J_pos = Var.Phise_pos / p.F ./ z.zint_pos;

%% 4. Solid surface concentration transfer function [mol/m^3/(A/m^2)]
Var.Css_neg = - p.rs_neg ./ (p.Ds_neg .* w.Ys_neg) .* w.JF_J_neg .* Var.J_neg;   % Equation (18-1)
Var.Css_pos = - p.rs_pos ./ (p.Ds_pos .* w.Ys_pos) .* w.JF_J_pos .* Var.J_pos;

%% 5. Solid-potential transfer function [V/(A/m^2)]
Var.Phis_neg(:,1) = - p.L_neg^3 * p.as_neg / p.sigma_eff_neg^2 ./ z.zint_neg ./ w.Pi_I_neg...
                 .* ((w.s_neg-w.lamda_I_neg) ./ w.lamda_I_neg .* w.B_I_neg...
                   + (w.s_neg-w.lamda_II_neg) ./ w.lamda_II_neg .* w.B_III_neg) + w.H_II_neg;                  % Equation (54-1)
Var.Phis_neg(:,2) = - p.L_neg^3 * p.as_neg / p.sigma_eff_neg^2 ./ z.zint_neg ./ w.Pi_I_neg...
                 .* ((w.s_neg-w.lamda_I_neg) ./ w.lamda_I_neg .* w.chi_I_neg...
                   + (w.s_neg-w.lamda_II_neg) ./ w.lamda_II_neg .* w.chi_II_neg) + (w.H_I_neg + w.H_II_neg);   % Equation (54-1)

Var.Phis_pos(:,1) = - p.L_pos^3 * p.as_pos / p.sigma_eff_pos^2 ./ z.zint_pos ./ w.Pi_I_pos...
                 .* ((w.s_pos-w.lamda_I_pos) ./ w.lamda_I_pos .* w.B_I_pos...
                   + (w.s_pos-w.lamda_II_pos) ./ w.lamda_II_pos .* w.B_III_pos) + w.H_II_pos;                  % Equation (55-1)
Var.Phis_pos(:,2) = - p.L_pos^3 * p.as_pos / p.sigma_eff_pos^2 ./ z.zint_pos ./ w.Pi_I_pos...
                 .* ((w.s_pos-w.lamda_I_pos) ./ w.lamda_I_pos .* w.chi_I_pos...
                   + (w.s_pos-w.lamda_II_pos) ./ w.lamda_II_pos .* w.chi_II_pos) + (w.H_I_pos + w.H_II_pos);   % Equation (55-1)

%% 6. Electrolyte-potential transfer function [V/(A/m^2)]
Var.Phie_neg = Var.Phis_neg - Var.Phise_neg;
Var.Phie_pos = Var.Phis_pos - Var.Phise_pos;

Var.Phie_sep(:,1) = - p.L_sep / p.kappa_eff_sep * (w.chi_III_sep - w.B_I_sep - 1);   % Equation (51-3)
Var.Phie_sep(:,2) = 0;

% The reference condition Equation (32)
Var.Phie_pos(:,1) = 0;

%% 7. Local average Li concentration transfer function [mol/m^3/(A/m^2)]
Var.Cs_local_neg = - 3 / p.rs_neg ./ w.s .* w.JF_J_neg .* Var.J_neg;
Var.Cs_local_pos = - 3 / p.rs_pos ./ w.s .* w.JF_J_pos .* Var.J_pos;

%% Cell impedance [Ω·m^2]
Cell.Z_neg  = - (Var.Phie_neg(:,end) - Var.Phis_neg(:,1));   % Equation (56-1)
Cell.Z_pos  = - (Var.Phis_pos(:,end) - Var.Phie_pos(:,1));   % Equation (56-2)
Cell.Z_sep  = - (Var.Phie_sep(:,end) - Var.Phie_sep(:,1));   % Equation (56-3)
Cell.Z_cell = - (Var.Phis_pos(:,end) - Var.Phis_neg(:,1));   % Equation (56-4)

%% Impedance expression [Ω·m^2]
Z.Z_neg = p.L_neg / (p.sigma_eff_neg + p.kappa_eff_neg)...
        - p.L_neg / p.sigma_eff_neg ./ w.Pi_I_neg .* ((w.s_neg-w.lamda_I_neg) .* w.chi_I_neg...
                                                    + (w.s_neg-w.lamda_II_neg) .* w.chi_II_neg)...
        + p.L_neg^3 * p.as_neg / p.sigma_eff_neg^2 ./ z.zint_neg ./ w.Pi_I_neg...
     .* ((w.s_neg-w.lamda_I_neg) ./ w.lamda_I_neg .* (w.chi_I_neg - w.B_I_neg)...
       + (w.s_neg-w.lamda_II_neg) ./ w.lamda_II_neg .* (w.chi_II_neg - w.B_III_neg));     % Equation (57-1)

Z.Z_pos = p.L_pos / (p.sigma_eff_pos + p.kappa_eff_pos)...
        + p.L_pos / p.sigma_eff_pos ./ w.Pi_I_pos .* ((w.s_pos-w.lamda_I_pos) .* w.B_I_pos...
                                                    + (w.s_pos-w.lamda_II_pos) .* w.B_III_pos)...
        + p.L_pos^3 * p.as_pos / p.sigma_eff_pos^2 ./ z.zint_pos ./ w.Pi_I_pos...
     .* ((w.s_pos-w.lamda_I_pos) ./ w.lamda_I_pos .* (w.chi_I_pos - w.B_I_pos)...
       + (w.s_pos-w.lamda_II_pos) ./ w.lamda_II_pos .* (w.chi_II_pos - w.B_III_pos));     % Equation (57-2)

Z.Z_sep = p.L_sep / p.kappa_eff_sep + p.L_sep / p.kappa_eff_sep * w.Pi_III_sep * p.F ./ sqrt(w.s_sep) ...
      .* (csch(sqrt(w.s_sep)) - 1 ./ tanh(sqrt(w.s_sep))) .* (w.zeta_neg + w.zeta_pos);   % Equation (57-3)

Z.Z_cell = p.L_pos^3 * p.as_pos / p.sigma_eff_pos^2 ./ z.zint_pos ./ w.Pi_I_pos...
      .* ((w.s_pos-w.lamda_I_pos) ./ w.lamda_I_pos .* w.chi_I_pos...
        + (w.s_pos-w.lamda_II_pos) ./ w.lamda_II_pos .* w.chi_II_pos) - (w.H_I_pos + w.H_II_pos)...
         - p.L_neg^3 * p.as_neg / p.sigma_eff_neg^2 ./ z.zint_neg ./ w.Pi_I_neg...
      .* ((w.s_neg-w.lamda_I_neg) ./ w.lamda_I_neg .* w.B_I_neg...
        + (w.s_neg-w.lamda_II_neg) ./ w.lamda_II_neg .* w.B_III_neg) + w.H_II_neg;        % Equation (57-4)

%%
fprintf('Finished the  DFN   impedance model in %2.5f ms\n',toc()*1e3);

end