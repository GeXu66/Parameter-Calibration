
% Sensitivity_calculate calculates the relative sensitivity of the parameter X to the diffusion processes.

% Inputs:
%    - ZD is a MATLAB struct containing the diffusion impedance for Li-ion batteries,
%      such as the solid diffusion impedance Z_Ds(s) or the electrolyte diffusion impedance Z_De(s).

% Outputs:
%    - SD is the relative sensitivity of the solid/electrolyte diffusion impedance to parameter set X.

% Note: Change range of parameter X is set as:
% Multiple(:,1) = logspace(-1,0,9); Multiple = Multiple(3:7) / Multiple(5);
% i.e., Multiple(:,1) = [0.5623; 0.7499; 1.0000; 1.3335; 1.7783];
% T = 25 * Multiple + (p.T_0 - 25).

function SD = Sensitivity_calculate(ZD)

N = length(ZD(1,:));     % N = 5 in the present study
M = length(ZD(:,1));     % M = 38 in the present study

% the average value of Z_fi at frequency f over i = 1~N times simulating results.
% find the average of each row.
ZD_av = mean(ZD(1:M,:) , 2);                    % Equation (87-1)

% the standard derivation of Z_fi over M sampling frequency points.
ZD_M = abs(ZD(1:M,:) - ZD_av);
% find the sum of each row.
ZD_S = (sum(ZD_M .^ 2 , 2) / N) .^ 0.5;         % Equation (87-2)

% the relative sensitivity of the parameter X to the solid/electrolyte diffusion processes.
SD = (sum(ZD_S) / M) / (sum(abs(ZD_av)) / M);   % Equation (87-3)

end
