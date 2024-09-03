
% Model_particle_calculate calculates three impedances at the particle scale.

% Inputs:
%    - p is a MATLAB struct containing the initial parameters and the calculated extra parameters.
%    - f is frequency, Hz.

% Outputs:
%    - w is a MATLAB struct containing the frequency variable s = j*2*π*f and the calculated related variables.
%    - z is a MATLAB struct containing three particle impedances in the single particle model,
%      such as solid-diffusion impedance z_d, the faradaic behavior impedance z_F, and the interfacial impedance z_int.

function [w, z] = Model_particle_calculate(p, f)

tic()

w.s = i * 2 * pi * f;

% the dimensionless transfer function [-]
w.Ys_neg = (sqrt(w.s * p.rs_neg^2 / p.Ds_neg) - tanh(sqrt(w.s * p.rs_neg^2 / p.Ds_neg)))...
         ./ tanh(sqrt(w.s * p.rs_neg^2 / p.Ds_neg));   % Equation (18-2)
w.Ys_pos = (sqrt(w.s * p.rs_pos^2 / p.Ds_pos) - tanh(sqrt(w.s * p.rs_pos^2 / p.Ds_pos)))...
         ./ tanh(sqrt(w.s * p.rs_pos^2 / p.Ds_pos));

% the complex impedances for the solid-diffusion [Ω·m^2]
z.zd_neg = p.Rdiff_neg ./ w.Ys_neg;   % Equation (23-1)
z.zd_pos = p.Rdiff_pos ./ w.Ys_pos;

% the complex impedances for the faradaic behavior [Ω·m^2]
z.zF_neg = 1 ./ (w.s * p.Cdl_neg + 1 ./ (p.Rct_neg + z.zd_neg));   % Equation (23-2)
z.zF_pos = 1 ./ (w.s * p.Cdl_pos + 1 ./ (p.Rct_pos + z.zd_pos));

% the complex impedances for the whole process with sei film in the particle [Ω·m^2]
z.zint_neg = 1 ./ (w.s * p.Csei_neg + 1 ./ (p.Rsei_neg + z.zF_neg));   % Equation (23-3)
z.zint_pos = 1 ./ (w.s * p.Csei_pos + 1 ./ (p.Rsei_pos + z.zF_pos));

% the calculation of J_F/J [-]
w.JF_J_neg = (1 - w.s * p.Csei_neg .* z.zint_neg) ./ (w.s * p.Cdl_neg .* (p.Rct_neg + z.zd_neg) + 1);
w.JF_J_pos = (1 - w.s * p.Csei_pos .* z.zint_pos) ./ (w.s * p.Cdl_pos .* (p.Rct_pos + z.zd_pos) + 1);

%%
fprintf('Finished the particle impedance in %2.5f ms\n',toc()*1e3);

end
