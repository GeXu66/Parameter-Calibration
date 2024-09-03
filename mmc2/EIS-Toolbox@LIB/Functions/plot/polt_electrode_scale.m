
% Plot DFN-like impedance models results

function Origin = polt_electrode_scale(out, Multiple, input)

%% Z_1
figure(1)
subplot(2,3,1)
dq = jet(length(Multiple));       % 5 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)        % 5 lines in plot are taken
plot(1e3 * out{d,1}.Model_DFN.Nyquist_Z_neg(:,1),...
     1e3 * out{d,1}.Model_DFN.Nyquist_Z_neg(:,2),...
     'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Set the properties of the coordinate axis.
xlabel('{\itZ}_{1}` (mΩ m^2)','fontsize',12,'fontname','Times')
ylabel('{\itZ}_{1}`` (mΩ m^2)','fontsize',12,'fontname','Times')
axis([-0.05 1 -1 0.05])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')

switch input
    case 'L_neg'
        h=legend([num2str(Multiple(1)),' {\itL}_{1}'],...
                 [num2str(Multiple(2)),' {\itL}_{1}'],...
                 [num2str(Multiple(3)),' {\itL}_{1}'],...
                 [num2str(Multiple(4)),' {\itL}_{1}'],...
                 [num2str(Multiple(5)),' {\itL}_{1}'],'Location','northeast');
    case 'epse_neg'
        h=legend([num2str(Multiple(1)),' {\itε}_{e1}'],...
                 [num2str(Multiple(2)),' {\itε}_{e1}'],...
                 [num2str(Multiple(3)),' {\itε}_{e1}'],...
                 [num2str(Multiple(4)),' {\itε}_{e1}'],...
                 [num2str(Multiple(5)),' {\itε}_{e1}'],'Location','northeast');
    case 'sigma_neg'
        h=legend([num2str(Multiple(1)),' {\itσ}_{1}'],...
                 [num2str(Multiple(2)),' {\itσ}_{1}'],...
                 [num2str(Multiple(3)),' {\itσ}_{1}'],...
                 [num2str(Multiple(4)),' {\itσ}_{1}'],...
                 [num2str(Multiple(5)),' {\itσ}_{1}'],'Location','northeast');
    case 'De'
        h=legend([num2str(Multiple(1)),' {\itD}_{e}'],...
                 [num2str(Multiple(2)),' {\itD}_{e}'],...
                 [num2str(Multiple(3)),' {\itD}_{e}'],...
                 [num2str(Multiple(4)),' {\itD}_{e}'],...
                 [num2str(Multiple(5)),' {\itD}_{e}'],'Location','northwest');
    case 'kappa'
        h=legend([num2str(Multiple(1)),' {\itκ}'],...
                 [num2str(Multiple(2)),' {\itκ}'],...
                 [num2str(Multiple(3)),' {\itκ}'],...
                 [num2str(Multiple(4)),' {\itκ}'],...
                 [num2str(Multiple(5)),' {\itκ}'],'Location','northwest');
    case 't_plus'
        h=legend([num2str(Multiple(1)),' {\itt}_{0+}'],...
                 [num2str(Multiple(2)),' {\itt}_{0+}'],...
                 [num2str(Multiple(3)),' {\itt}_{0+}'],...
                 [num2str(Multiple(4)),' {\itt}_{0+}'],...
                 [num2str(Multiple(5)),' {\itt}_{0+}'],'Location','northeast');
    case 'T'
        Multiple_T = 25 + Multiple;
        h=legend([num2str(Multiple_T(1)),' °C'],...
                 [num2str(Multiple_T(2)),' °C'],...
                 [num2str(Multiple_T(3)),' °C'],...
                 [num2str(Multiple_T(4)),' °C'],...
                 [num2str(Multiple_T(5)),' °C'],'Location','northeast');
end
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')

switch input
    case 'L_neg'
        title('Figure 15 (a) {\itL}_{1}')
    case 'epse_neg'
        title('Figure 16 (a) {\itε}_{e1}')
    case 'sigma_neg'
        title('Figure 17 (a) {\itσ}_{1}')
    case 'De'
        title('Figure 19 (a) {\itD}_e')
    case 'kappa'
        title('Figure 20 (a) {\itκ}')
    case 't_plus'
        title('Figure 21 (a) {\itt}_{0+}')
    case 'T'
        title('No Figure (a) {\itT}')
end

subplot(4,3,2)
dq = jet(length(Multiple));       % 5 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)        % 5 lines in plot are taken
semilogx(out{1,1}.f,1e3 * out{d,1}.Model_DFN.Nyquist_Z_neg(:,1),'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Represented by log axis
set(gca,'XScale','log')
% Set the properties of the coordinate axis.
% xlabel('{\itf} (Hz)','fontsize',12,'fontname','Times')
ylabel('{\itZ}_{1}` (mΩ m^2)','fontsize',12,'fontname','Times')
axis([10^-3 10^5 -0.05 1])
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Do not display x-axis scale
set(gca,'xticklabel',[]);
switch input
    case 'L_neg'
        title('Figure 15 (b) {\itL}_{1}')
    case 'epse_neg'
        title('Figure 16 (b) {\itε}_{e1}')
    case 'sigma_neg'
        title('Figure 17 (b) {\itσ}_{1}')
    case 'De'
        title('Figure 19 (b) {\itD}_e')
    case 'kappa'
        title('Figure 20 (b) {\itκ}')
    case 't_plus'
        title('Figure 21 (b) {\itt}_{0+}')
    case 'T'
        title('No Figure (b) {\itT}')
end

subplot(4,3,5)
dq = jet(length(Multiple));       % 5 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)        % 5 lines in plot are taken
semilogx(out{1,1}.f,1e3 * out{d,1}.Model_DFN.Nyquist_Z_neg(:,2),'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Represented by log axis
set(gca,'XScale','log')
% Set the properties of the coordinate axis.
xlabel('{\itf} (Hz)','fontsize',12,'fontname','Times')
ylabel('{\itZ}_{1}`` (mΩ m^2)','fontsize',12,'fontname','Times')
axis([10^-3 10^5 -1 0.05])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Adjust the position of the entire coordinate region in the figure window
set(gca,'position',[0.4108 0.5895 0.2134 0.1577])

%% Z_2
subplot(2,3,4)
dq = jet(length(Multiple));       % 5 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)        % 5 lines in plot are taken
plot(1e3 * out{d,1}.Model_DFN.Nyquist_Z_pos(:,1),...
     1e3 * out{d,1}.Model_DFN.Nyquist_Z_pos(:,2),...
     'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Set the properties of the coordinate axis.
xlabel('{\itZ}_{2}` (mΩ m^2)','fontsize',12,'fontname','Times')
ylabel('{\itZ}_{2}`` (mΩ m^2)','fontsize',12,'fontname','Times')
axis([-0.1 2 -2 0.1])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')

switch input
    case 'L_neg'
        h=legend([num2str(Multiple(1)),' {\itL}_{2}'],...
                 [num2str(Multiple(2)),' {\itL}_{2}'],...
                 [num2str(Multiple(3)),' {\itL}_{2}'],...
                 [num2str(Multiple(4)),' {\itL}_{2}'],...
                 [num2str(Multiple(5)),' {\itL}_{2}'],'Location','northeast');
    case 'epse_neg'
        h=legend([num2str(Multiple(1)),' {\itε}_{e2}'],...
                 [num2str(Multiple(2)),' {\itε}_{e2}'],...
                 [num2str(Multiple(3)),' {\itε}_{e2}'],...
                 [num2str(Multiple(4)),' {\itε}_{e2}'],...
                 [num2str(Multiple(5)),' {\itε}_{e2}'],'Location','northeast');
    case 'sigma_neg'
        h=legend([num2str(Multiple(1)),' {\itσ}_{2}'],...
                 [num2str(Multiple(2)),' {\itσ}_{2}'],...
                 [num2str(Multiple(3)),' {\itσ}_{2}'],...
                 [num2str(Multiple(4)),' {\itσ}_{2}'],...
                 [num2str(Multiple(5)),' {\itσ}_{2}'],'Location','northeast');
    case 'De'
        h=legend([num2str(Multiple(1)),' {\itD}_{e}'],...
                 [num2str(Multiple(2)),' {\itD}_{e}'],...
                 [num2str(Multiple(3)),' {\itD}_{e}'],...
                 [num2str(Multiple(4)),' {\itD}_{e}'],...
                 [num2str(Multiple(5)),' {\itD}_{e}'],'Location','northeast');
    case 'kappa'
        h=legend([num2str(Multiple(1)),' {\itκ}'],...
                 [num2str(Multiple(2)),' {\itκ}'],...
                 [num2str(Multiple(3)),' {\itκ}'],...
                 [num2str(Multiple(4)),' {\itκ}'],...
                 [num2str(Multiple(5)),' {\itκ}'],'Location','northeast');
    case 't_plus'
        h=legend([num2str(Multiple(1)),' {\itt}_{0+}'],...
                 [num2str(Multiple(2)),' {\itt}_{0+}'],...
                 [num2str(Multiple(3)),' {\itt}_{0+}'],...
                 [num2str(Multiple(4)),' {\itt}_{0+}'],...
                 [num2str(Multiple(5)),' {\itt}_{0+}'],'Location','northeast');
    case 'T'
        Multiple_T = 25 + Multiple;
        h=legend([num2str(Multiple_T(1)),' °C'],...
                 [num2str(Multiple_T(2)),' °C'],...
                 [num2str(Multiple_T(3)),' °C'],...
                 [num2str(Multiple_T(4)),' °C'],...
                 [num2str(Multiple_T(5)),' °C'],'Location','northeast');
end
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')

switch input
    case 'L_neg'
        title('Figure 15 (c) {\itL}_{2}')
    case 'epse_neg'
        title('Figure 16 (c) {\itε}_{e2}')
    case 'sigma_neg'
        title('Figure 17 (c) {\itσ}_{2}')
    case 'De'
        title('Figure 19 (c) {\itD}_e')
    case 'kappa'
        title('Figure 20 (c) {\itκ}')
    case 't_plus'
        title('Figure 21 (c) {\itt}_{0+}')
    case 'T'
        title('No Figure (c) {\itT}')
end

subplot(4,3,8)
dq = jet(length(Multiple));       % 5 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)        % 5 lines in plot are taken
semilogx(out{1,1}.f,1e3 * out{d,1}.Model_DFN.Nyquist_Z_pos(:,1),'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Represented by log axis
set(gca,'XScale','log')
% Set the properties of the coordinate axis.
% xlabel('{\itf} (Hz)','fontsize',12,'fontname','Times')
ylabel('{\itZ}_{2}` (mΩ m^2)','fontsize',12,'fontname','Times')
axis([10^-3 10^5 -0.1 2])
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Do not display x-axis scale
set(gca,'xticklabel',[]);
% Adjust the position of the entire coordinate region in the figure window
set(gca,'position',[0.4108 0.2944 0.2134 0.1577])
switch input
    case 'L_neg'
        title('Figure 15 (d) {\itL}_{2}')
    case 'epse_neg'
        title('Figure 16 (d) {\itε}_{e2}')
    case 'sigma_neg'
        title('Figure 17 (d) {\itσ}_{2}')
    case 'De'
        title('Figure 19 (d) {\itD}_e')
    case 'kappa'
        title('Figure 20 (d) {\itκ}')
    case 't_plus'
        title('Figure 21 (d) {\itt}_{0+}')
    case 'T'
        title('No Figure (d) {\itT}')
end

subplot(4,3,11)
dq = jet(length(Multiple));       % 5 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)        % 5 lines in plot are taken
semilogx(out{1,1}.f,1e3 * out{d,1}.Model_DFN.Nyquist_Z_pos(:,2),'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Represented by log axis
set(gca,'XScale','log')
% Set the properties of the coordinate axis.
xlabel('{\itf} (Hz)','fontsize',12,'fontname','Times')
ylabel('{\itZ}_{2}`` (mΩ m^2)','fontsize',12,'fontname','Times')
axis([10^-3 10^5 -2 0.1 ])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')

%% Z_3
figure(2)
subplot(2,3,1)
dq = jet(length(Multiple));       % 5 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)        % 5 lines in plot are taken
plot(1e3 * out{d,1}.Model_DFN.Nyquist_Z_sep(:,1),...
     1e3 * out{d,1}.Model_DFN.Nyquist_Z_sep(:,2),...
     'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Set the properties of the coordinate axis.
xlabel('Z_{3}` (mΩ m^2)','fontsize',12,'fontname','Times')
ylabel('Z_{3}`` (mΩ m^2)','fontsize',12,'fontname','Times')
axis([-0.01 0.2 -0.2 0.01])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')

switch input
    case 'L_neg'
        h=legend([num2str(Multiple(1)),' {\itL}_{3}'],...
                 [num2str(Multiple(2)),' {\itL}_{3}'],...
                 [num2str(Multiple(3)),' {\itL}_{3}'],...
                 [num2str(Multiple(4)),' {\itL}_{3}'],...
                 [num2str(Multiple(5)),' {\itL}_{3}'],'Location','northeast');
    case 'epse_neg'
        h=legend([num2str(Multiple(1)),' {\itε}_{e3}'],...
                 [num2str(Multiple(2)),' {\itε}_{e3}'],...
                 [num2str(Multiple(3)),' {\itε}_{e3}'],...
                 [num2str(Multiple(4)),' {\itε}_{e3}'],...
                 [num2str(Multiple(5)),' {\itε}_{e3}'],'Location','northeast');
    case 'sigma_neg'
        h=legend([num2str(Multiple(1)),' {\itσ}'],...
                 [num2str(Multiple(2)),' {\itσ}'],...
                 [num2str(Multiple(3)),' {\itσ}'],...
                 [num2str(Multiple(4)),' {\itσ}'],...
                 [num2str(Multiple(5)),' {\itσ}'],'Location','northeast');
    case 'De'
        h=legend([num2str(Multiple(1)),' {\itD}_{e}'],...
                 [num2str(Multiple(2)),' {\itD}_{e}'],...
                 [num2str(Multiple(3)),' {\itD}_{e}'],...
                 [num2str(Multiple(4)),' {\itD}_{e}'],...
                 [num2str(Multiple(5)),' {\itD}_{e}'],'Location','northwest');
    case 'kappa'
        h=legend([num2str(Multiple(1)),' {\itκ}'],...
                 [num2str(Multiple(2)),' {\itκ}'],...
                 [num2str(Multiple(3)),' {\itκ}'],...
                 [num2str(Multiple(4)),' {\itκ}'],...
                 [num2str(Multiple(5)),' {\itκ}'],'Location','northwest');
    case 't_plus'
        h=legend([num2str(Multiple(1)),' {\itt}_{0+}'],...
                 [num2str(Multiple(2)),' {\itt}_{0+}'],...
                 [num2str(Multiple(3)),' {\itt}_{0+}'],...
                 [num2str(Multiple(4)),' {\itt}_{0+}'],...
                 [num2str(Multiple(5)),' {\itt}_{0+}'],'Location','northeast');
    case 'T'
        Multiple_T = 25 + Multiple;
        h=legend([num2str(Multiple_T(1)),' °C'],...
                 [num2str(Multiple_T(2)),' °C'],...
                 [num2str(Multiple_T(3)),' °C'],...
                 [num2str(Multiple_T(4)),' °C'],...
                 [num2str(Multiple_T(5)),' °C'],'Location','northeast');
end
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')

switch input
    case 'L_neg'
        title('Figure 15 (e) {\itL}_{3}')
    case 'epse_neg'
        title('Figure 16 (e) {\itε}_{e3}')
    case 'sigma_neg'
        title('No Figure (e) {\itσ}')
    case 'De'
        title('Figure 19 (e) {\itD}_e')
    case 'kappa'
        title('Figure 20 (e) {\itκ}')
    case 't_plus'
        title('Figure 21 (e) {\itt}_{0+}')
    case 'T'
        title('Figure 24 (a) {\itT}')
end

subplot(4,3,2)
dq = jet(length(Multiple));       % 5 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)        % 5 lines in plot are taken
semilogx(out{1,1}.f,1e3 * out{d,1}.Model_DFN.Nyquist_Z_sep(:,1),'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Represented by log axis
set(gca,'XScale','log')
% Set the properties of the coordinate axis.
% xlabel('{\itf} (Hz)','fontsize',12,'fontname','Times')
ylabel('{\itZ}_{3}` (mΩ m^2)','fontsize',12,'fontname','Times')
axis([10^-3 10^5 -0.01 0.2])
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Do not display x-axis scale
set(gca,'xticklabel',[]);
switch input
    case 'L_neg'
        title('Figure 15 (f) {\itL}_{3}')
    case 'epse_neg'
        title('Figure 16 (f) {\itε}_{e3}')
    case 'sigma_neg'
        title('No Figure (f) {\itσ}')
    case 'De'
        title('Figure 19 (f) {\itD}_e')
    case 'kappa'
        title('Figure 20 (f) {\itκ}')
    case 't_plus'
        title('Figure 21 (f) {\itt}_{0+}')
    case 'T'
        title('Figure 24 (b) {\itT}')
end

subplot(4,3,5)
dq = jet(length(Multiple));       % 5 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)        % 11 lines in plot are taken
semilogx(out{1,1}.f,1e3 * out{d,1}.Model_DFN.Nyquist_Z_sep(:,2),'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Represented by log axis
set(gca,'XScale','log')
% Set the properties of the coordinate axis.
xlabel('{\itf} (Hz)','fontsize',12,'fontname','Times')
ylabel('{\itZ}_{3}`` (mΩ m^2)','fontsize',12,'fontname','Times')
axis([10^-3 10^5 -0.2 0.01])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
% Adjust the position of the entire coordinate region in the figure window
set(gca,'position',[0.4108 0.5895 0.2134 0.1577])

%% origin
Origin{1,1} = [out{1,1}.Model_DFN.Nyquist_Z_neg...      % Z_1
               out{2,1}.Model_DFN.Nyquist_Z_neg...
               out{3,1}.Model_DFN.Nyquist_Z_neg...
               out{4,1}.Model_DFN.Nyquist_Z_neg...
               out{5,1}.Model_DFN.Nyquist_Z_neg...
               out{1,1}.Model_DFN.Nyquist_Z_pos...      % Z_2
               out{2,1}.Model_DFN.Nyquist_Z_pos...
               out{3,1}.Model_DFN.Nyquist_Z_pos...
               out{4,1}.Model_DFN.Nyquist_Z_pos...
               out{5,1}.Model_DFN.Nyquist_Z_pos...
               out{1,1}.Model_DFN.Nyquist_Z_sep...      % Z_3
               out{2,1}.Model_DFN.Nyquist_Z_sep...
               out{3,1}.Model_DFN.Nyquist_Z_sep...
               out{4,1}.Model_DFN.Nyquist_Z_sep...
               out{5,1}.Model_DFN.Nyquist_Z_sep...
               out{1,1}.Model_DFN.Nyquist_Z_cell...     % Z_4
               out{2,1}.Model_DFN.Nyquist_Z_cell...
               out{3,1}.Model_DFN.Nyquist_Z_cell...
               out{4,1}.Model_DFN.Nyquist_Z_cell...
               out{5,1}.Model_DFN.Nyquist_Z_cell] * 1e3;

Origin{2,1} = [out{1,1}.f / 1e3...
               out{1,1}.Model_DFN.Nyquist_Z_neg(:,1)...      % Z_1
               out{2,1}.Model_DFN.Nyquist_Z_neg(:,1)...
               out{3,1}.Model_DFN.Nyquist_Z_neg(:,1)...
               out{4,1}.Model_DFN.Nyquist_Z_neg(:,1)...
               out{5,1}.Model_DFN.Nyquist_Z_neg(:,1)...
               out{1,1}.Model_DFN.Nyquist_Z_pos(:,1)...      % Z_2
               out{2,1}.Model_DFN.Nyquist_Z_pos(:,1)...
               out{3,1}.Model_DFN.Nyquist_Z_pos(:,1)...
               out{4,1}.Model_DFN.Nyquist_Z_pos(:,1)...
               out{5,1}.Model_DFN.Nyquist_Z_pos(:,1)...
               out{1,1}.Model_DFN.Nyquist_Z_sep(:,1)...      % Z_3
               out{2,1}.Model_DFN.Nyquist_Z_sep(:,1)...
               out{3,1}.Model_DFN.Nyquist_Z_sep(:,1)...
               out{4,1}.Model_DFN.Nyquist_Z_sep(:,1)...
               out{5,1}.Model_DFN.Nyquist_Z_sep(:,1)...
               out{1,1}.Model_DFN.Nyquist_Z_cell(:,1)...     % Z_4
               out{2,1}.Model_DFN.Nyquist_Z_cell(:,1)...
               out{3,1}.Model_DFN.Nyquist_Z_cell(:,1)...
               out{4,1}.Model_DFN.Nyquist_Z_cell(:,1)...
               out{5,1}.Model_DFN.Nyquist_Z_cell(:,1)] * 1e3;

Origin{3,1} = [out{1,1}.f / 1e3...
               out{1,1}.Model_DFN.Nyquist_Z_neg(:,2)...      % Z_1
               out{2,1}.Model_DFN.Nyquist_Z_neg(:,2)...
               out{3,1}.Model_DFN.Nyquist_Z_neg(:,2)...
               out{4,1}.Model_DFN.Nyquist_Z_neg(:,2)...
               out{5,1}.Model_DFN.Nyquist_Z_neg(:,2)...
               out{1,1}.Model_DFN.Nyquist_Z_pos(:,2)...      % Z_2
               out{2,1}.Model_DFN.Nyquist_Z_pos(:,2)...
               out{3,1}.Model_DFN.Nyquist_Z_pos(:,2)...
               out{4,1}.Model_DFN.Nyquist_Z_pos(:,2)...
               out{5,1}.Model_DFN.Nyquist_Z_pos(:,2)...
               out{1,1}.Model_DFN.Nyquist_Z_sep(:,2)...      % Z_3
               out{2,1}.Model_DFN.Nyquist_Z_sep(:,2)...
               out{3,1}.Model_DFN.Nyquist_Z_sep(:,2)...
               out{4,1}.Model_DFN.Nyquist_Z_sep(:,2)...
               out{5,1}.Model_DFN.Nyquist_Z_sep(:,2)...
               out{1,1}.Model_DFN.Nyquist_Z_cell(:,2)...     % Z_4
               out{2,1}.Model_DFN.Nyquist_Z_cell(:,2)...
               out{3,1}.Model_DFN.Nyquist_Z_cell(:,2)...
               out{4,1}.Model_DFN.Nyquist_Z_cell(:,2)...
               out{5,1}.Model_DFN.Nyquist_Z_cell(:,2)] * 1e3;

end
