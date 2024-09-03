
% Extract_Re_Im extracts the real and imaginary parts of the complex impedance.

% Inputs:
%    - z is a MATLAB struct containing the particle complex impedance in the single particle model.
%    - Z is a MATLAB struct containing the complex impedance of individual components in the full cell.
%    - ZD is a MATLAB struct containing the diffusion complex impedance for Li-ion batteries.

% Outputs:
%    - z is a MATLAB struct containing the particle complex impedance and its real & imaginary parts.
%    - Z is a MATLAB struct containing the complex impedance of individual components and its real & imaginary parts.
%    - ZD is a MATLAB struct containing the diffusion complex impedance and its real & imaginary parts.

function [z, Z, ZD] = Extract_Re_Im(z, Z, ZD)

%% particle impedance [Ω·m^2]
if isempty(z) ~= 1
    % negative electrode
    z.Nyquist_zd_neg   = [real(z.zd_neg)   imag(z.zd_neg)];
    z.Nyquist_zF_neg   = [real(z.zF_neg)   imag(z.zF_neg)];
    z.Nyquist_zint_neg = [real(z.zint_neg) imag(z.zint_neg)];

    % positive electrode
    z.Nyquist_zd_pos   = [real(z.zd_pos)   imag(z.zd_pos)];
    z.Nyquist_zF_pos   = [real(z.zF_pos)   imag(z.zF_pos)];
    z.Nyquist_zint_pos = [real(z.zint_pos) imag(z.zint_pos)];
end

%% component impedance [Ω·m^2]
if isempty(Z) ~= 1
    Z.Nyquist_Z_neg  = [real(Z.Z_neg)  imag(Z.Z_neg)];
    Z.Nyquist_Z_pos  = [real(Z.Z_pos)  imag(Z.Z_pos)];
    Z.Nyquist_Z_sep  = [real(Z.Z_sep)  imag(Z.Z_sep)];
    Z.Nyquist_Z_cell = [real(Z.Z_cell) imag(Z.Z_cell)];
end

%% diffusion impedance [Ω·m^2]
if isempty(ZD) ~= 1
    % solid diffusion
    ZD.Nyquist_Ds_neg  = [real(ZD.Ds_neg)  imag(ZD.Ds_neg)];
    ZD.Nyquist_Ds_pos  = [real(ZD.Ds_pos)  imag(ZD.Ds_pos)];
    ZD.Nyquist_Ds_cell = [real(ZD.Ds_cell) imag(ZD.Ds_cell)];

    % electrolyte diffusion
    ZD.Nyquist_De_neg  = [real(ZD.De_neg)  imag(ZD.De_neg)];
    ZD.Nyquist_De_pos  = [real(ZD.De_pos)  imag(ZD.De_pos)];
    ZD.Nyquist_De_sep  = [real(ZD.De_sep)  imag(ZD.De_sep)];
    ZD.Nyquist_De_cell = [real(ZD.De_cell) imag(ZD.De_cell)];

    % total diffusion
    ZD.Nyquist_D_neg  = [real(ZD.D_neg)  imag(ZD.D_neg)];
    ZD.Nyquist_D_pos  = [real(ZD.D_pos)  imag(ZD.D_pos)];
    ZD.Nyquist_D_sep  = [real(ZD.D_sep)  imag(ZD.D_sep)];
    ZD.Nyquist_D_cell = [real(ZD.D_cell) imag(ZD.D_cell)];
end

end