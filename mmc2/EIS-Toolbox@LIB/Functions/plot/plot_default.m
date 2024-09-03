
% Plot DFN-like impedance models results

function Origin = plot_default(out)

%% Figure 2
% multi-scale
figure(1)
subplot(2,3,1)
plot(1e3 * out.z_par.Nyquist_zd_neg(:,1),...
     1e3 * out.z_par.Nyquist_zd_neg(:,2),'-r','linewidth',3)
% Add a legend.
h=legend('{\itz}_{d1}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',24,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([-0.9 30 -30 0.9])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Set title.
title('Figure 2 (h)')

subplot(2,3,4)
plot(1e3 * out.z_par.Nyquist_zF_neg(:,1),...
     1e3 * out.z_par.Nyquist_zF_neg(:,2),'-b','linewidth',3)
% Add a legend.
h=legend('{\itz}_{F1}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',24,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([-0.9 30 -30 0.9])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Set title.
title('Figure 2 (i)')

subplot(2,3,2)
plot(1e3 * out.z_par.Nyquist_zint_neg(:,1),...
     1e3 * out.z_par.Nyquist_zint_neg(:,2),'-m','linewidth',3)
% Add a legend.
h=legend('{\itz}_{int1}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',24,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([-0.9 30 -30 0.9])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Set title.
title('Figure 2 (j)')

subplot(2,3,5)
plot(1e3 * out.Model_DFN.Nyquist_Z_neg(:,1),...
     1e3 * out.Model_DFN.Nyquist_Z_neg(:,2),'-g','linewidth',3)
% Add a legend.
h=legend('{\itZ}_{1}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',24,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([-0.03 1 -1 0.03])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Set title.
title('Figure 2 (k)')

%% Figure 5
% negative electrode impedance
figure(2)
subplot(2,3,1)
plot(1e3 * out.Model_C.Nyquist_Z_neg(:,1),  1e3 * out.Model_C.Nyquist_Z_neg(:,2)-0.714,  '-c','linewidth',3)
hold on
plot(1e3 * out.Model_F.Nyquist_Z_neg(:,1),  1e3 * out.Model_F.Nyquist_Z_neg(:,2)-0.594,  '-y','linewidth',3)
plot(1e3 * out.Model_B.Nyquist_Z_neg(:,1),  1e3 * out.Model_B.Nyquist_Z_neg(:,2)-0.475,  '-b','linewidth',3)
plot(1e3 * out.Model_E.Nyquist_Z_neg(:,1),  1e3 * out.Model_E.Nyquist_Z_neg(:,2)-0.355,  '-g','linewidth',3)
plot(1e3 * out.Model_A.Nyquist_Z_neg(:,1),  1e3 * out.Model_A.Nyquist_Z_neg(:,2)-0.235,  '-r','linewidth',3)
plot(1e3 * out.Model_D.Nyquist_Z_neg(:,1),  1e3 * out.Model_D.Nyquist_Z_neg(:,2)-0.118,  '-m','linewidth',3)
plot(1e3 * out.Model_DFN.Nyquist_Z_neg(:,1),1e3 * out.Model_DFN.Nyquist_Z_neg(:,2),      '-k','linewidth',3)
% Add a legend.
h = legend('{\itZ}_{C#1}','{\itZ}_{F#1}','{\itZ}_{B#1}','{\itZ}_{E#1}',...
           '{\itZ}_{A#1}','{\itZ}_{D#1}','{\itZ}_{1}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([-0.05 1 -1 0.05])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Set title.
title('Figure 5 (a)')

% negative electrode diffusion impedance
subplot(2,3,2)
plot(1e3 * out.ZD.Nyquist_Ds_neg(:,1),1e3 * out.ZD.Nyquist_Ds_neg(:,2)-0.258,'-m','linewidth',3)
hold on
plot(1e3 * out.ZD.Nyquist_De_neg(:,1),1e3 * out.ZD.Nyquist_De_neg(:,2)-0.129,'-y','linewidth',3)
plot(1e3 * out.ZD.Nyquist_D_neg(:,1), 1e3 * out.ZD.Nyquist_D_neg(:,2),       '-g','linewidth',3)
% Add a legend.
h = legend('{\itZ}_{Ds1}','{\itZ}_{De1}','{\itZ}_{D1}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',18,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([-0.125 0.4 -0.5 0.025])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Set title.
title('Figure 5 (b)')

%% Figure 6
% impedance of different components
figure(3)
subplot(2,3,1)
plot(1e3 * out.Model_DFN.Nyquist_Z_neg(:,1),1e3 * out.Model_DFN.Nyquist_Z_neg(:,2),'-k','linewidth',3)
hold on
plot(1e3 * out.Model_A.Nyquist_Z_neg(:,1),  1e3 * out.Model_A.Nyquist_Z_neg(:,2),  '-r','linestyle','--','linewidth',3)
plot(1e3 * out.Model_B.Nyquist_Z_neg(:,1),  1e3 * out.Model_B.Nyquist_Z_neg(:,2),  '-b','linewidth',3)
plot(1e3 * out.Model_C.Nyquist_Z_neg(:,1),  1e3 * out.Model_C.Nyquist_Z_neg(:,2),  '.m','linestyle','--','linewidth',3)
% Add a legend.
h = legend('{\itZ}_{1}','{\itZ}_{A#1}','{\itZ}_{B#1}','{\itZ}_{C#1}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',18,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([-0.05 1 -1 0.05])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Set title.
title('Figure 6 (a)')

subplot(2,3,4)
plot(1e3 * out.Model_DFN.Nyquist_Z_pos(:,1),1e3 * out.Model_DFN.Nyquist_Z_pos(:,2),'-k','linewidth',3)
hold on
plot(1e3 * out.Model_A.Nyquist_Z_pos(:,1),  1e3 * out.Model_A.Nyquist_Z_pos(:,2),  '-r','linestyle','--','linewidth',3)
plot(1e3 * out.Model_B.Nyquist_Z_pos(:,1),  1e3 * out.Model_B.Nyquist_Z_pos(:,2),  '-b','linewidth',3)
plot(1e3 * out.Model_C.Nyquist_Z_pos(:,1),  1e3 * out.Model_C.Nyquist_Z_pos(:,2),  '.m','linestyle','--','linewidth',3)
% Add a legend.
h = legend('{\itZ}_{2}','{\itZ}_{A#2}','{\itZ}_{B#2}','{\itZ}_{C#2}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',18,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([-0.1 2 -2 0.1])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Set title.
title('Figure 6 (c)')

figure(4)
subplot(2,3,1)
plot(1e3 * out.Model_DFN.Nyquist_Z_sep(:,1),1e3 * out.Model_DFN.Nyquist_Z_sep(:,2),'-k','linewidth',3)
hold on
plot(1e3 * out.Model_A.Nyquist_Z_sep(:,1),  1e3 * out.Model_A.Nyquist_Z_sep(:,2),  'or','linewidth',6)
plot(1e3 * out.Model_B.Nyquist_Z_sep(:,1),  1e3 * out.Model_B.Nyquist_Z_sep(:,2),  'ob','linewidth',3)
plot(1e3 * out.Model_C.Nyquist_Z_sep(:,1),  1e3 * out.Model_C.Nyquist_Z_sep(:,2),  '.m','markersize',8)
% Add a legend.
h = legend('{\itZ}_{3}','{\itZ}_{A#3}','{\itZ}_{B#3}','{\itZ}_{C#3}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',18,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([-0.01 0.2 -0.2 0.01])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Set title.
title('Figure 6 (e)')

subplot(2,3,4)
plot(1e3 * out.Model_DFN.Nyquist_Z_cell(:,1),1e3 * out.Model_DFN.Nyquist_Z_cell(:,2),'-k','linewidth',3)
hold on
plot(1e3 * out.Model_A.Nyquist_Z_cell(:,1),  1e3 * out.Model_A.Nyquist_Z_cell(:,2),  '-r','linestyle','--','linewidth',3)
plot(1e3 * out.Model_B.Nyquist_Z_cell(:,1),  1e3 * out.Model_B.Nyquist_Z_cell(:,2),  '-b','linewidth',3)
plot(1e3 * out.Model_C.Nyquist_Z_cell(:,1),  1e3 * out.Model_C.Nyquist_Z_cell(:,2),  '.m','linestyle','--','linewidth',3)
% Add a legend.
h = legend('{\itZ}_{4}','{\itZ}_{A#4}','{\itZ}_{B#4}','{\itZ}_{C#4}','Location','northwest');
set(h,'FontName','Times New Roman','FontSize',18,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([-0.1 2 -2 0.1])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Set title.
title('Figure 6 (g)')

% diffusion impedance
figure(3)
subplot(4,3,2)
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_De_neg(:,1),'-b','linewidth',3);
hold on
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_Ds_neg(:,1),'-r','linewidth',3);
% Add a legend.
h = legend('{\itZ}_{De1}','{\itZ}_{Ds1}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',18,'FontWeight','normal')
% Set the properties of the coordinate axis.
% xlabel('{\it f} (Hz)','fontsize',20,'fontname','Times')
ylabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([10^-3, 10^0 -0.025, 0.5])
% Flip the Y axis up and down.
% set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Do not display x-axis scale
set(gca,'xticklabel',[]);
% Set title.
title('Figure 6 (b)')

subplot(4,3,5)
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_De_neg(:,2),'-b','linewidth',3);
hold on
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_Ds_neg(:,2),'-r','linewidth',3);
% Add a legend.
h = legend('{\itZ}_{De1}','{\itZ}_{Ds1}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',18,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\it f} (Hz)','fontsize',20,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([10^-3, 10^0 -0.5, 0.025])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Adjust the position of the entire coordinate region in the figure window
set(gca,'position',[0.4108 0.5895 0.2134 0.1577])

subplot(4,3,8)
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_De_pos(:,1),'-b','linewidth',3);
hold on
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_Ds_pos(:,1),'-r','linewidth',3);
% Add a legend.
h = legend('{\itZ}_{De2}','{\itZ}_{Ds2}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',18,'FontWeight','normal')
% Set the properties of the coordinate axis.
% xlabel('{\it f} (Hz)','fontsize',20,'fontname','Times')
ylabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([10^-3, 10^0 -0.1, 2])
% Flip the Y axis up and down.
% set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Do not display x-axis scale
set(gca,'xticklabel',[]);
% Adjust the position of the entire coordinate region in the figure window
set(gca,'position',[0.4108 0.2944 0.2134 0.1577])
% Set title.
title('Figure 6 (d)')

subplot(4,3,11)
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_De_pos(:,2),'-b','linewidth',3);
hold on
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_Ds_pos(:,2),'-r','linewidth',3);
% Add a legend.
h = legend('{\itZ}_{De2}','{\itZ}_{Ds2}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',18,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\it f} (Hz)','fontsize',20,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([10^-3, 10^0 -2, 0.1])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')

figure(4)
subplot(4,3,2)
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_De_sep(:,1),'-b','linewidth',3);
% Add a legend.
h = legend('{\itZ}_{De3}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',18,'FontWeight','normal')
% Set the properties of the coordinate axis.
% xlabel('{\it f} (Hz)','fontsize',20,'fontname','Times')
ylabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([10^-3, 10^0 -0.005, 0.1])
% Flip the Y axis up and down.
% set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Do not display x-axis scale
set(gca,'xticklabel',[]);
% Set title.
title('Figure 6 (f)')

subplot(4,3,5)
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_De_sep(:,2),'-b','linewidth',3);
% Add a legend.
h = legend('{\itZ}_{De3}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',18,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\it f} (Hz)','fontsize',20,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([10^-3, 10^0 -0.1, 0.005])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Adjust the position of the entire coordinate region in the figure window
set(gca,'position',[0.4108 0.5895 0.2134 0.1577])

subplot(4,3,8)
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_De_cell(:,1),'-b','linewidth',3);
hold on
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_Ds_cell(:,1),'-r','linewidth',3);
% Add a legend.
h = legend('{\itZ}_{De4}','{\itZ}_{Ds4}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',18,'FontWeight','normal')
% Set the properties of the coordinate axis.
% xlabel('{\it f} (Hz)','fontsize',20,'fontname','Times')
ylabel('{\itZ}` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([10^-3, 10^0 -0.1, 2])
% Flip the Y axis up and down.
% set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Do not display x-axis scale
set(gca,'xticklabel',[]);
% Adjust the position of the entire coordinate region in the figure window
set(gca,'position',[0.4108 0.2944 0.2134 0.1577])
% Set title.
title('Figure 6 (h)')

subplot(4,3,11)
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_De_cell(:,2),'-b','linewidth',3);
hold on
semilogx(out.f_ZD,1e3 * out.ZD.Nyquist_Ds_cell(:,2),'-r','linewidth',3);
% Add a legend.
h = legend('{\itZ}_{De4}','{\itZ}_{Ds4}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',18,'FontWeight','normal')
% Set the properties of the coordinate axis.
xlabel('{\it f} (Hz)','fontsize',20,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',15,'fontname','Times')
axis([10^-3, 10^0 -2, 0.1])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')

%% data pasted into Origin
% Figure 2 (h)-(k)
Origin{1,1} = [out.z_par.Nyquist_zd_neg...           % z_d1
               out.z_par.Nyquist_zF_neg...           % z_F1
               out.z_par.Nyquist_zint_neg...         % z_int1
               out.Model_DFN.Nyquist_Z_neg] * 1e3;   % Z_1

number_F2 = [10 ...          % z_d1
             63 10 ...       % z_F1
             80 63 10 ...    % z_int1
             81 63 10 14];   % Z_1
Origin{2,1} = [out.z_par.Nyquist_zd_neg(number_F2(1),:)...      % z_d1
               out.z_par.Nyquist_zF_neg(number_F2(2),:)...      % z_F1
               out.z_par.Nyquist_zF_neg(number_F2(3),:)...
               out.z_par.Nyquist_zint_neg(number_F2(4),:)...    % z_int1
               out.z_par.Nyquist_zint_neg(number_F2(5),:)...
               out.z_par.Nyquist_zint_neg(number_F2(6),:)...
               out.Model_DFN.Nyquist_Z_neg(number_F2(7),:)...   % Z_1
               out.Model_DFN.Nyquist_Z_neg(number_F2(8),:)...
               out.Model_DFN.Nyquist_Z_neg(number_F2(9),:)...
               out.Model_DFN.Nyquist_Z_neg(number_F2(10),:)] * 1e3;

% Figure 5 (a) and (b)
Origin{3,1} = [out.Model_DFN.Nyquist_Z_neg...
               out.Model_A.Nyquist_Z_neg...
               out.Model_B.Nyquist_Z_neg...
               out.Model_C.Nyquist_Z_neg...
               out.Model_D.Nyquist_Z_neg...
               out.Model_E.Nyquist_Z_neg...
               out.Model_F.Nyquist_Z_neg] * 1e3;

Origin{4,1} = [out.ZD.Nyquist_Ds_neg...
               out.ZD.Nyquist_De_neg...
               out.ZD.Nyquist_D_neg] * 1e3;

number_F5 = [1 37 100 ...  % Z_1
             1 38];        % Z_D1
Origin{5,1} = [out.Model_DFN.Nyquist_Z_neg(number_F5(1),:)...   % Z_1
               out.Model_DFN.Nyquist_Z_neg(number_F5(2),:)...
               out.Model_DFN.Nyquist_Z_neg(number_F5(3),:)...
               out.ZD.Nyquist_D_neg(number_F5(4),:)...          % Z_D1
               out.ZD.Nyquist_D_neg(number_F5(5),:)] * 1e3;

% Figure 6 (a),(c),(e) and (g)
Origin{6,1} = [out.Model_DFN.Nyquist_Z_neg  out.Model_A.Nyquist_Z_neg  out.Model_B.Nyquist_Z_neg  out.Model_C.Nyquist_Z_neg...
               out.Model_DFN.Nyquist_Z_pos  out.Model_A.Nyquist_Z_pos  out.Model_B.Nyquist_Z_pos  out.Model_C.Nyquist_Z_pos...
               out.Model_DFN.Nyquist_Z_sep  out.Model_A.Nyquist_Z_sep  out.Model_B.Nyquist_Z_sep  out.Model_C.Nyquist_Z_sep...
               out.Model_DFN.Nyquist_Z_cell out.Model_A.Nyquist_Z_cell out.Model_B.Nyquist_Z_cell out.Model_C.Nyquist_Z_cell] * 1e3;

% Figure 6 (b),(d),(f) and (h)
Origin{7,1} = [out.f_ZD / 1e3...
               out.ZD.Nyquist_De_neg  out.ZD.Nyquist_Ds_neg...
               out.ZD.Nyquist_De_pos  out.ZD.Nyquist_Ds_pos...
               out.ZD.Nyquist_De_sep...
               out.ZD.Nyquist_De_cell out.ZD.Nyquist_Ds_cell] * 1e3;

end
