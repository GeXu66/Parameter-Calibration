
% EIS calculates the impedance of DFN-like impedance models for Li-ion batteries.

% Inputs:
%    - p is a MATLAB struct containing the initial parameters and the calculated extra parameters.
%    - f is frequency, Hz.

% Outputs:
%    - out is a MATLAB struct containing the simulation parameters (p),
%      frequency (f & f_ZD),
%      the particle impedance (z_d, z_F, & z_int),
%      the impedance of individual components in the full cell (Z, Z_A#, Z_B#, Z_C#, Z_D#, Z_E#, Z_F#),
%      and the diffusion impedance (Z_Ds, Z_De, & Z_D).

% Practically, the impedance at high frequencies cannot be calculated
% using a numerical calculator such as Matlab because the hyperbolic
% sine and cosine terms always go infinite in high frequency range.
% A expedient way to cope with this difficulty is to transform
% all the hyperbolic sine and cosine terms to
% hyperbolic tangent terms bounded in the range of −1 ∼ 1.

function out = EIS(p, f)

out.p = p;
out.f = f;

%% include solid phase diffusion
[w, out.z_par] = Model_particle_calculate(p, f);

[~, out.Model_DFN] = Model_DFN_calculate(p, w, out.z_par);
[~, out.Model_A]   = Model_A_D_calculate(p, w, out.z_par);
[~, out.Model_B]   = Model_B_E_calculate(p, w, out.z_par);
[~, out.Model_C]   = Model_C_F_calculate(p, w, out.z_par);

%% neglect solid phase diffusion
p_zd0 = p;
p_zd0.Rdiff_neg = 0;
p_zd0.Rdiff_pos = 0;

[w_zd0, z_par_zd0] = Model_particle_calculate(p_zd0, f);

[~, out.Model_D] = Model_A_D_calculate(p_zd0, w_zd0, z_par_zd0);
[~, out.Model_E] = Model_B_E_calculate(p_zd0, w_zd0, z_par_zd0);
[~, out.Model_F] = Model_C_F_calculate(p_zd0, w_zd0, z_par_zd0);

%% diffusion impedance
[out.f_ZD, out.ZD] = Model_diffusion_calculate(out.Model_DFN, out.Model_B, out.Model_E, f);

%% extract real part and imaginary part
[out.z_par, ~, ~] = Extract_Re_Im(out.z_par, [], []);

[~, out.Model_DFN, ~] = Extract_Re_Im([], out.Model_DFN, []);
[~, out.Model_A, ~]   = Extract_Re_Im([], out.Model_A, []);
[~, out.Model_B, ~]   = Extract_Re_Im([], out.Model_B, []);
[~, out.Model_C, ~]   = Extract_Re_Im([], out.Model_C, []);
[~, out.Model_D, ~]   = Extract_Re_Im([], out.Model_D, []);
[~, out.Model_E, ~]   = Extract_Re_Im([], out.Model_E, []);
[~, out.Model_F, ~]   = Extract_Re_Im([], out.Model_F, []);

[~, ~, out.ZD] = Extract_Re_Im([], [], out.ZD);

end
