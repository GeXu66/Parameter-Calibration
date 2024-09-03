
% Model_C_F_calculate calculates the impedance of Model C or F for Li-ion batteries.

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
%      such as the negative electrode impedance Z_C#1(s) or Z_F#1(s), the positive electrode impedance Z_C#2(s) or Z_F#2(s),
%      the separator impedance Z_C#3(s) or Z_F#3(s), and the full cell impedance Z_C#4(s) or Z_F#4(s).

function [Var, Z] = Model_C_F_calculate(p, w, z)
tic()

%% Frequency domain variables
% Θ_II: the ratio of the whole electrode-related ohmic resistance to z_int [-]
w.Pi_II_neg = p.as_neg * p.L_neg^2 / p.kappa_eff_neg ./ z.zint_neg;   % Equation (79-1)
w.Pi_II_pos = p.as_pos * p.L_pos^2 / p.kappa_eff_pos ./ z.zint_pos;   % Equation (79-2)

% the expressions for the coefficients in the negative electrode region [-]
w.B_III_neg = - 1 ./ sqrt(w.Pi_II_neg) .* csch(sqrt(w.Pi_II_neg));       % Equation (S29-1)
w.B_IV_neg = 0;                                                          % Equation (S29-2)

% the expressions for the coefficients in the positive electrode region [-]
w.B_III_pos = 1 ./ sqrt(w.Pi_II_pos) ./ tanh(sqrt(w.Pi_II_pos));         % Equation (81-2)
w.B_IV_pos = - 1 ./ sqrt(w.Pi_II_pos);                                   % Equation (S29-4)

% Γ_II = B_III * cosh(sqrt(Θ_II)) + B_IV * sinh(sqrt(Θ_II))
w.Lambda_II_neg = - 1 ./ sqrt(w.Pi_II_neg) ./ tanh(sqrt(w.Pi_II_neg));   % Equation (81-1)
w.Lambda_II_pos =   1 ./ sqrt(w.Pi_II_pos);

% the expressions for the coefficients in the electrode region [V/(A/m^2)]
w.H_II_neg = - p.L_neg / p.kappa_eff_neg * w.Lambda_II_neg...
             + p.L_sep / p.kappa_eff_sep;                 % Equation (80-1)
w.H_II_pos = - p.L_pos / p.kappa_eff_pos * w.B_III_pos;   % Equation (80-2)

%% 1. Electrolyte-concentration transfer function [mol/m^3/(A/m^2)]
Var.Ce_neg = 0 * ones(length(w.s),1);   % Equation (S21-1)
Var.Ce_sep = 0 * ones(length(w.s),1);   % Equation (S21-3)
Var.Ce_pos = 0 * ones(length(w.s),1);   % Equation (S21-2)

%% 2. Solid–electrolyte potential difference transfer function [V/(A/m^2)]
Var.Phise_neg(:,1) = - p.L_neg / p.kappa_eff_neg .* w.B_III_neg;       % Equation (S31-1)
Var.Phise_neg(:,2) = - p.L_neg / p.kappa_eff_neg .* w.Lambda_II_neg;   % Equation (S31-1)

Var.Phise_pos(:,1) = - p.L_pos / p.kappa_eff_pos .* w.B_III_pos;       % Equation (S31-2)
Var.Phise_pos(:,2) = - p.L_pos / p.kappa_eff_pos .* w.Lambda_II_pos;   % Equation (S31-2)

%% 3. Reaction flux transfer function [mol/m^2/s/(A/m^2)]
Var.J_neg = Var.Phise_neg ./ p.F ./ z.zint_neg;   % Equation (23-3)
Var.J_pos = Var.Phise_pos ./ p.F ./ z.zint_pos;

%% 4. Solid surface concentration transfer function [mol/m^3/(A/m^2)]
Var.Css_neg = - p.rs_neg ./ (p.Ds_neg * w.Ys_neg) .* w.JF_J_neg .* Var.J_neg;   % Equation (18-1)
Var.Css_pos = - p.rs_pos ./ (p.Ds_pos * w.Ys_pos) .* w.JF_J_pos .* Var.J_pos;

%% 5. Solid-potential transfer function [V/(A/m^2)]
Var.Phis_neg = w.H_II_neg;   % Equation (S34-1)
Var.Phis_pos = w.H_II_pos;   % Equation (S35-1)

%% 6. Electrolyte-potential transfer function [V/(A/m^2)]
Var.Phie_neg = Var.Phis_neg - Var.Phise_neg;
Var.Phie_pos = Var.Phis_pos - Var.Phise_pos;

Var.Phie_sep(:,1) = p.L_sep / p.kappa_eff_sep * ones(length(w.s),1);   % Equation (S30)
Var.Phie_sep(:,2) = 0 * ones(length(w.s),1);

% The reference condition Equation (32)
Var.Phie_pos(:,1) = 0 * ones(length(w.s),1);

%% 7. local average Li concentration transfer function [mol/m^3/(A/m^2)]
Var.Cs_local_neg = - 3 / p.rs_neg ./ w.s .* w.JF_J_neg .* Var.J_neg;
Var.Cs_local_pos = - 3 / p.rs_pos ./ w.s .* w.JF_J_pos .* Var.J_pos;

%% Cell impedance [Ω·m^2]
Cell.Z_neg  = - (Var.Phie_neg(:,end) - Var.Phis_neg(:,1));   % Equation (56-1)
Cell.Z_pos  = - (Var.Phis_pos(:,end) - Var.Phie_pos(:,1));   % Equation (56-2)
Cell.Z_sep  = - (Var.Phie_sep(:,end) - Var.Phie_sep(:,1));   % Equation (56-3)
Cell.Z_cell = - (Var.Phis_pos(:,end) - Var.Phis_neg(:,1));   % Equation (56-4)

%% Impedance expression [Ω·m^2]
Z.Z_neg  = p.L_neg / p.kappa_eff_neg * coth(sqrt(w.Pi_II_neg)) ./ sqrt(w.Pi_II_neg);   % Equation (78-1)
Z.Z_pos  = p.L_pos / p.kappa_eff_pos * coth(sqrt(w.Pi_II_pos)) ./ sqrt(w.Pi_II_pos);   % Equation (78-2)
Z.Z_sep  = p.L_sep / p.kappa_eff_sep * ones(length(w.s),1);                            % Equation (78-3)
Z.Z_cell = (w.H_II_neg - w.H_II_pos);                                                  % Equation (78-4)

%%
fprintf('Finished the C or F impedance model in %2.5f ms\n',toc()*1e3);

end
