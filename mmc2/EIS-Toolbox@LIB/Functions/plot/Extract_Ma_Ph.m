
% Extract_Ma_Ph extracts the magnitude and phase of the complex impedance.

% Inputs:
%    - z is a MATLAB struct containing the particle complex impedance in the single particle model.
%    - Z is a MATLAB struct containing the complex impedance of individual components in the full cell.
%    - ZD is a MATLAB struct containing the diffusion complex impedance for Li-ion batteries.

% Outputs:
%    - z is a MATLAB struct containing the particle complex impedance and its magnitude & phase.
%    - Z is a MATLAB struct containing the complex impedance of individual components and its magnitude & phase.
%    - ZD is a MATLAB struct containing the diffusion complex impedance and its magnitude & phase.

% Note: abs(x) = (real(x).^2 + imag(x).^2) .^ 0.5;
%     angle(x) = atan(real(x) ./ imag(x));

function [z, Z, ZD] = Extract_Ma_Ph(z, Z, ZD)

%% particle impedance [Ω·m^2]
if isempty(z) ~= 1
    % negative electrode
    z.Bode_zd_neg   = [abs(z.zd_neg)   angle(z.zd_neg)   / pi * 180];
    z.Bode_zF_neg   = [abs(z.zF_neg)   angle(z.zF_neg)   / pi * 180];
    z.Bode_zint_neg = [abs(z.zint_neg) angle(z.zint_neg) / pi * 180];

    % positive electrode
    z.Bode_zd_pos   = [abs(z.zd_pos)   angle(z.zd_pos)   / pi * 180];
    z.Bode_zF_pos   = [abs(z.zF_pos)   angle(z.zF_pos)   / pi * 180];
    z.Bode_zint_pos = [abs(z.zint_pos) angle(z.zint_pos) / pi * 180];
end

%% component impedance [Ω·m^2]
if isempty(Z) ~= 1
    Z.Bode_Z_neg  = [abs(Z.Z_neg)  angle(Z.Z_neg)  / pi * 180];
    Z.Bode_Z_pos  = [abs(Z.Z_pos)  angle(Z.Z_pos)  / pi * 180];
    Z.Bode_Z_sep  = [abs(Z.Z_sep)  angle(Z.Z_sep)  / pi * 180];
    Z.Bode_Z_cell = [abs(Z.Z_cell) angle(Z.Z_cell) / pi * 180];
end

%% diffusion impedance [Ω·m^2]
if isempty(ZD) ~= 1
    % solid diffusion
    ZD.Bode_Ds_neg  = [abs(ZD.Ds_neg)  angle(ZD.Ds_neg)  / pi * 180];
    ZD.Bode_Ds_pos  = [abs(ZD.Ds_pos)  angle(ZD.Ds_pos)  / pi * 180];
    ZD.Bode_Ds_cell = [abs(ZD.Ds_cell) angle(ZD.Ds_cell) / pi * 180];

    % electrolyte diffusion
    ZD.Bode_De_neg  = [abs(ZD.De_neg)  angle(ZD.De_neg)  / pi * 180];
    ZD.Bode_De_pos  = [abs(ZD.De_pos)  angle(ZD.De_pos)  / pi * 180];
    ZD.Bode_De_sep  = [abs(ZD.De_sep)  angle(ZD.De_sep)  / pi * 180];
    ZD.Bode_De_cell = [abs(ZD.De_cell) angle(ZD.De_cell) / pi * 180];

    % total diffusion
    ZD.Bode_D_neg  = [abs(ZD.D_neg)  angle(ZD.D_neg)  / pi * 180];
    ZD.Bode_D_pos  = [abs(ZD.D_pos)  angle(ZD.D_pos)  / pi * 180];
    ZD.Bode_D_sep  = [abs(ZD.D_sep)  angle(ZD.D_sep)  / pi * 180];
    ZD.Bode_D_cell = [abs(ZD.D_cell) angle(ZD.D_cell) / pi * 180];
end

end
