
% Model_diffusion_calculate calculates the diffusion impedance for Li-ion batteries.

% Inputs:
%    - Model_DFN is a MATLAB struct containing the impedance of individual components in the Model DFN,
%      such as the negative electrode impedance Z_1(s), the positive electrode impedance Z_2(s),
%      the separator impedance Z_3(s), and the full cell impedance Z_4(s).
%    - Model_B is a MATLAB struct containing the impedance of individual components in the Model B,
%      such as the negative electrode impedance Z_B#1(s), the positive electrode impedance Z_B#2(s),
%      the separator impedance Z_B#3(s), and the full cell impedance Z_B#4(s).
%    - Model_E is a MATLAB struct containing the impedance of individual components in the Model E,
%      such as the negative electrode impedance Z_E#1(s), the positive electrode impedance Z_E#2(s),
%      the separator impedance Z_E#3(s), and the full cell impedance Z_E#4(s).
%    - f is frequency, Hz.

% Outputs:
%    - f_ZD is frequency (f_ZD < 1), Hz.
%    - ZD is a MATLAB struct containing the diffusion impedance for Li-ion batteries,
%      such as the solid diffusion impedance Z_Ds(s), the electrolyte diffusion impedance Z_De(s),
%      and the full diffusion impedance Z_D(s).

function [f_ZD, ZD] = Model_diffusion_calculate(Model_DFN, Model_B, Model_E, f)

if f(end) > 10^0
    for m = 1:length(f)
        if f(m) > 10^0
            M = m - 1;
            break
        end
    end
else
    M = length(f);
end

f_ZD = f(1:M,:);

% solid diffusion: Z_Ds = Z_B - Z_E [Ω·m^2]
ZD.Ds_neg  = Model_B.Z_neg(1:M,:)  - Model_E.Z_neg(1:M,:);     % Equation (82-1)
ZD.Ds_pos  = Model_B.Z_pos(1:M,:)  - Model_E.Z_pos(1:M,:);
ZD.Ds_cell = Model_B.Z_cell(1:M,:) - Model_E.Z_cell(1:M,:);

% electrolyte diffusion: Z_De = Z - Z_B [Ω·m^2]
ZD.De_neg  = Model_DFN.Z_neg(1:M,:)  - Model_B.Z_neg(1:M,:);   % Equation (82-2)
ZD.De_pos  = Model_DFN.Z_pos(1:M,:)  - Model_B.Z_pos(1:M,:);
ZD.De_sep  = Model_DFN.Z_sep(1:M,:)  - Model_B.Z_sep(1:M,:);
ZD.De_cell = Model_DFN.Z_cell(1:M,:) - Model_B.Z_cell(1:M,:);

% total diffusion: Z_D = Z - Z_E [Ω·m^2]
ZD.D_neg  = Model_DFN.Z_neg(1:M,:)  - Model_E.Z_neg(1:M,:);    % Equation (82-3)
ZD.D_pos  = Model_DFN.Z_pos(1:M,:)  - Model_E.Z_pos(1:M,:);
ZD.D_sep  = Model_DFN.Z_sep(1:M,:)  - Model_E.Z_sep(1:M,:);
ZD.D_cell = Model_DFN.Z_cell(1:M,:) - Model_E.Z_cell(1:M,:);

end