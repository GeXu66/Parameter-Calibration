
% Plot DFN-like impedance models results

function Origin = polt_model_simplify(out, Multiple, input)

switch input
    case 'De_eff_sep'
        out{4,1}.Model_DFN = out{4,1}.Model_A;
        for m = 1:length(out)
            Cell{m,1} = out{m,1}.Model_DFN;
            [~, Cell{m,1}, ~]   = Extract_Ma_Ph([], Cell{m,1}, []);
        end
    case 'De_eff'
        out{4,1}.Model_A = out{4,1}.Model_B;
        for m = 1:length(out)
            Cell{m,1} = out{m,1}.Model_A;
            [~, Cell{m,1}, ~]   = Extract_Ma_Ph([], Cell{m,1}, []);
        end
    case 'sigma_eff'
        out{4,1}.Model_B = out{4,1}.Model_C;
        for m = 1:length(out)
            Cell{m,1} = out{m,1}.Model_B;
            [~, Cell{m,1}, ~]   = Extract_Ma_Ph([], Cell{m,1}, []);
        end
end

%% Difference
for n = 1:3
% absolute value
    Abso.neg(:,n)  = abs(Cell{n,1}.Bode_Z_neg(:,1)  - Cell{4,1}.Bode_Z_neg(:,1));
    Abso.pos(:,n)  = abs(Cell{n,1}.Bode_Z_pos(:,1)  - Cell{4,1}.Bode_Z_pos(:,1));
    Abso.sep(:,n)  = abs(Cell{n,1}.Bode_Z_sep(:,1)  - Cell{4,1}.Bode_Z_sep(:,1));
    Abso.cell(:,n) = abs(Cell{n,1}.Bode_Z_cell(:,1) - Cell{4,1}.Bode_Z_cell(:,1));
end

% relative value
Rela.neg  = Abso.neg  ./ Cell{4,1}.Bode_Z_neg(:,1);
Rela.pos  = Abso.pos  ./ Cell{4,1}.Bode_Z_pos(:,1);
Rela.sep  = Abso.sep  ./ Cell{4,1}.Bode_Z_sep(:,1);
Rela.cell = Abso.cell ./ Cell{4,1}.Bode_Z_cell(:,1);

%% negative electrode impedance
figure(1)
subplot(2,3,1)
dq = jet(length(Multiple));       % 4 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)        % 4 lines in plot are taken
plot(1e3 * Cell{d,1}.Nyquist_Z_neg(:,1),...
     1e3 * Cell{d,1}.Nyquist_Z_neg(:,2),...
     'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',12,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',12,'fontname','Times')
axis([-0.05 1 -1 0.05])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')

switch input
    case 'De_eff_sep'
        h=legend([num2str(Multiple(1)),' {\itD}_{e,eff3} ({\itZ}_1)'],...
                 [num2str(Multiple(2)),' {\itD}_{e,eff3} ({\itZ}_1)'],...
                 [num2str(Multiple(3)),' {\itD}_{e,eff3} ({\itZ}_1)'],...
                                       '{Z}_{A#1}','Location','northeast');
    case 'De_eff'
        h=legend([num2str(Multiple(1)),' {\itD}_{e,eff1} ({\itZ}_{A#1})'],...
                 [num2str(Multiple(2)),' {\itD}_{e,eff1} ({\itZ}_{A#1})'],...
                 [num2str(Multiple(3)),' {\itD}_{e,eff1} ({\itZ}_{A#1})'],...
                                       '{Z}_{B#1}','Location','northeast');
    case 'sigma_eff'
        h=legend([num2str(Multiple(1)),' {\itσ}_{e,eff1} ({\itZ}_{B#1})'],...
                 [num2str(Multiple(2)),' {\itσ}_{e,eff1} ({\itZ}_{B#1})'],...
                 [num2str(Multiple(3)),' {\itσ}_{e,eff1} ({\itZ}_{B#1})'],...
                                       '{Z}_{C#1}','Location','northeast');
end
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')

switch input
    case 'De_eff_sep'
        title('Figure 7 (a) {\itD}_{e,eff3}')
    case 'De_eff'
        title('Figure 8 (a) {\itD}_{e,eff1}')
    case 'sigma_eff'
        title('Figure 9 (a) {\itσ}_{eff1}')
end

subplot(2,3,2)
dq = jet(length(Multiple));       % 4 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)-1      % 4 lines in plot are taken
semilogx(out{1,1}.f,Rela.neg(:,d) * 100,'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Represented by log axis
set(gca,'XScale','log')
% Set the properties of the coordinate axis.
xlabel('{\itf} (Hz)','fontsize',12,'fontname','Times')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')

switch input
    case 'De_eff_sep'
        ylabel('Δ{\itZ}_A (%)','fontsize',12,'fontname','Times')
        h=legend([num2str(Multiple(1)),' {\itD}_{e,eff3} ({\itZ}_1)'],...
                 [num2str(Multiple(2)),' {\itD}_{e,eff3} ({\itZ}_1)'],...
                 [num2str(Multiple(3)),' {\itD}_{e,eff3} ({\itZ}_1)'],'Location','northeast');
    case 'De_eff'
        ylabel('Δ{\itZ}_B (%)','fontsize',12,'fontname','Times')
        h=legend([num2str(Multiple(1)),' {\itD}_{e,eff1} ({\itZ}_{A#1})'],...
                 [num2str(Multiple(2)),' {\itD}_{e,eff1} ({\itZ}_{A#1})'],...
                 [num2str(Multiple(3)),' {\itD}_{e,eff1} ({\itZ}_{A#1})'],'Location','northeast');
    case 'sigma_eff'
        ylabel('Δ{\itZ}_C (%)','fontsize',12,'fontname','Times')
        h=legend([num2str(Multiple(1)),' {\itσ}_{e,eff1} ({\itZ}_{B#1})'],...
                 [num2str(Multiple(2)),' {\itσ}_{e,eff1} ({\itZ}_{B#1})'],...
                 [num2str(Multiple(3)),' {\itσ}_{e,eff1} ({\itZ}_{B#1})'],'Location','northwest');
end
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')
set(gca,'FontSize',15,'ygrid','on')

switch input
    case 'De_eff_sep'
        title('Figure 7 (b) {\itD}_{e,eff3}')
        axis([10^-3 10^5 -0.5 10])
    case 'De_eff'
        title('Figure 8 (b) {\itD}_{e,eff1}')
        axis([10^-3 10^5 -2.5 50])
    case 'sigma_eff'
        title('Figure 9 (b left) {\itσ}_{eff1})')
        axis([10^-3 10^5 -5 100])
end

%% positive electrode impedance
subplot(2,3,4)
dq = jet(length(Multiple));       % 4 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)        % 4 lines in plot are taken
plot(1e3 * Cell{d,1}.Nyquist_Z_pos(:,1),...
     1e3 * Cell{d,1}.Nyquist_Z_pos(:,2),...
     'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',12,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',12,'fontname','Times')
axis([-0.1 2 -2 0.1])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')

switch input
    case 'De_eff_sep'
        h=legend([num2str(Multiple(1)),' {\itD}_{e,eff3} ({\itZ}_2)'],...
                 [num2str(Multiple(2)),' {\itD}_{e,eff3} ({\itZ}_2)'],...
                 [num2str(Multiple(3)),' {\itD}_{e,eff3} ({\itZ}_2)'],...
                                       '{Z}_{A#2}','Location','northeast');
    case 'De_eff'
        h=legend([num2str(Multiple(1)),' {\itD}_{e,eff2} ({\itZ}_{A#2})'],...
                 [num2str(Multiple(2)),' {\itD}_{e,eff2} ({\itZ}_{A#2})'],...
                 [num2str(Multiple(3)),' {\itD}_{e,eff2} ({\itZ}_{A#2})'],...
                                       '{Z}_{B#2}','Location','northeast');
    case 'sigma_eff'
        h=legend([num2str(Multiple(1)),' {\itσ}_{e,eff2} ({\itZ}_{B#2})'],...
                 [num2str(Multiple(2)),' {\itσ}_{e,eff2} ({\itZ}_{B#2})'],...
                 [num2str(Multiple(3)),' {\itσ}_{e,eff2} ({\itZ}_{B#2})'],...
                                       '{Z}_{C#2}','Location','northeast');
end
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')

switch input
    case 'De_eff_sep'
        title('Figure 7 (c) {\itD}_{e,eff3}')
    case 'De_eff'
        title('Figure 8 (c) {\itD}_{e,eff2}')
    case 'sigma_eff'
        title('Figure 9 (c) {\itσ}_{eff2}')
end

subplot(2,3,5)
dq = jet(length(Multiple));       % 4 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)-1      % 4 lines in plot are taken
semilogx(out{1,1}.f,Rela.pos(:,d) * 100,'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Represented by log axis
set(gca,'XScale','log')
% Set the properties of the coordinate axis.
xlabel('{\itf} (Hz)','fontsize',12,'fontname','Times')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')

switch input
    case 'De_eff_sep'
        ylabel('Δ{\itZ}_A (%)','fontsize',12,'fontname','Times')
        h=legend([num2str(Multiple(1)),' {\itD}_{e,eff3} ({\itZ}_2)'],...
                 [num2str(Multiple(2)),' {\itD}_{e,eff3} ({\itZ}_2)'],...
                 [num2str(Multiple(3)),' {\itD}_{e,eff3} ({\itZ}_2)'],'Location','northeast');
    case 'De_eff'
        ylabel('Δ{\itZ}_B (%)','fontsize',12,'fontname','Times')
        h=legend([num2str(Multiple(1)),' {\itD}_{e,eff2} ({\itZ}_{A#2})'],...
                 [num2str(Multiple(2)),' {\itD}_{e,eff2} ({\itZ}_{A#2})'],...
                 [num2str(Multiple(3)),' {\itD}_{e,eff2} ({\itZ}_{A#2})'],'Location','northeast');
    case 'sigma_eff'
        ylabel('Δ{\itZ}_C (%)','fontsize',12,'fontname','Times')
        h=legend([num2str(Multiple(1)),' {\itσ}_{e,eff2} ({\itZ}_{B#2})'],...
                 [num2str(Multiple(2)),' {\itσ}_{e,eff2} ({\itZ}_{B#2})'],...
                 [num2str(Multiple(3)),' {\itσ}_{e,eff2} ({\itZ}_{B#2})'],'Location','northwest');
end
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')
set(gca,'FontSize',15,'ygrid','on')

switch input
    case 'De_eff_sep'
        title('Figure 7 (d) {\itD}_{e,eff3}')
        axis([10^-3 10^5 -0.5 10])
    case 'De_eff'
        title('Figure 8 (d) {\itD}_{e,eff2}')
        axis([10^-3 10^5 -2.5 50])
    case 'sigma_eff'
        title('Figure 9 (d left) {\itσ}_{eff2})')
        axis([10^-3 10^5 -5 100])
end

%% separator impedance
figure(2)
switch input
    case 'De_eff_sep'

subplot(2,3,1)
plot(1e3 * Cell{1,1}.Nyquist_Z_sep(:,1),1e3 * Cell{1,1}.Nyquist_Z_sep(:,2),'-b','linewidth',3)
hold on
plot(1e3 * Cell{2,1}.Nyquist_Z_sep(:,1),1e3 * Cell{2,1}.Nyquist_Z_sep(:,2),'-g','linewidth',3)
plot(1e3 * Cell{3,1}.Nyquist_Z_sep(:,1),1e3 * Cell{3,1}.Nyquist_Z_sep(:,2),'-y','linewidth',3)
plot(1e3 * Cell{4,1}.Nyquist_Z_sep(:,1),1e3 * Cell{4,1}.Nyquist_Z_sep(:,2),'or','linewidth',3)
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',12,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',12,'fontname','Times')
axis([-0.005 0.1 -0.1 0.005])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
h=legend([num2str(Multiple(1)),' {\itD}_{e,eff3} ({\itZ}_3)'],...
         [num2str(Multiple(2)),' {\itD}_{e,eff3} ({\itZ}_3)'],...
         [num2str(Multiple(3)),' {\itD}_{e,eff3} ({\itZ}_3)'],...
                               '{Z}_{A#3}','Location','northeast');
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')
title('Figure 7 (e) {\itD}_{e,eff3}')

subplot(2,3,2)
dq = jet(length(Multiple));       % 4 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)-1      % 4 lines in plot are taken
semilogx(out{1,1}.f,Rela.sep(:,d) * 100,'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Represented by log axis
set(gca,'XScale','log')
% Set the properties of the coordinate axis.
xlabel('{\itf} (Hz)','fontsize',12,'fontname','Times')
ylabel('Δ{\itZ}_A (%)','fontsize',12,'fontname','Times')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
h=legend([num2str(Multiple(1)),' {\itD}_{e,eff3} ({\itZ}_3)'],...
         [num2str(Multiple(2)),' {\itD}_{e,eff3} ({\itZ}_3)'],...
         [num2str(Multiple(3)),' {\itD}_{e,eff3} ({\itZ}_3)'],'Location','northeast');
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')
title('Figure 7 (f) {\itD}_{e,eff3}')
axis([10^-3 10^5 -10 200])

end

%% cell impedance
subplot(2,3,4)
dq = jet(length(Multiple));       % 4 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)        % 4 lines in plot are taken
plot(1e3 * Cell{d,1}.Nyquist_Z_cell(:,1),...
     1e3 * Cell{d,1}.Nyquist_Z_cell(:,2),...
     'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Set the properties of the coordinate axis.
xlabel('{\itZ}` (mΩ m^2)','fontsize',12,'fontname','Times')
ylabel('{\itZ}`` (mΩ m^2)','fontsize',12,'fontname','Times')
axis([-0.1 2 -2 0.1])
% Flip the Y axis up and down.
set(gca, 'YDir','reverse')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')

switch input
    case 'De_eff_sep'
        h=legend([num2str(Multiple(1)),' {\itD}_{e,eff3} ({\itZ}_4)'],...
                 [num2str(Multiple(2)),' {\itD}_{e,eff3} ({\itZ}_4)'],...
                 [num2str(Multiple(3)),' {\itD}_{e,eff3} ({\itZ}_4)'],...
                                       '{Z}_{A#4}','Location','northwest');
    case 'De_eff'
        h=legend([num2str(Multiple(1)),' {\itD}_{e,eff} ({\itZ}_{A#4})'],...
                 [num2str(Multiple(2)),' {\itD}_{e,eff} ({\itZ}_{A#4})'],...
                 [num2str(Multiple(3)),' {\itD}_{e,eff} ({\itZ}_{A#4})'],...
                                       '{Z}_{B#4}','Location','northwest');
    case 'sigma_eff'
        h=legend([num2str(Multiple(1)),' {\itσ}_{e,eff} ({\itZ}_{B#4})'],...
                 [num2str(Multiple(2)),' {\itσ}_{e,eff} ({\itZ}_{B#4})'],...
                 [num2str(Multiple(3)),' {\itσ}_{e,eff} ({\itZ}_{B#4})'],...
                                       '{Z}_{C#4}','Location','northwest');
end
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')

switch input
    case 'De_eff_sep'
        title('Figure 7 (g) {\itD}_{e,eff3}')
    case 'De_eff'
        title('Figure 8 (e) {\itD}_{e,eff}')
    case 'sigma_eff'
        title('Figure 9 (e) {\itσ}_{eff}')
end

subplot(2,3,5)
dq = jet(length(Multiple));       % 4 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)-1      % 4 lines in plot are taken
semilogx(out{1,1}.f,Rela.cell(:,d) * 100,'color',dq(i,:),'linewidth',3)
    i = i+1;
end
% Represented by log axis
set(gca,'XScale','log')
% Set the properties of the coordinate axis.
xlabel('{\itf} (Hz)','fontsize',12,'fontname','Times')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')

switch input
    case 'De_eff_sep'
        ylabel('Δ{\itZ}_A (%)','fontsize',12,'fontname','Times')
        h=legend([num2str(Multiple(1)),' {\itD}_{e,eff3} ({\itZ}_4)'],...
                 [num2str(Multiple(2)),' {\itD}_{e,eff3} ({\itZ}_4)'],...
                 [num2str(Multiple(3)),' {\itD}_{e,eff3} ({\itZ}_4)'],'Location','northeast');
    case 'De_eff'
        ylabel('Δ{\itZ}_B (%)','fontsize',12,'fontname','Times')
        h=legend([num2str(Multiple(1)),' {\itD}_{e,eff} ({\itZ}_{A#4})'],...
                 [num2str(Multiple(2)),' {\itD}_{e,eff} ({\itZ}_{A#4})'],...
                 [num2str(Multiple(3)),' {\itD}_{e,eff} ({\itZ}_{A#4})'],'Location','northeast');
    case 'sigma_eff'
        ylabel('Δ{\itZ}_C (%)','fontsize',12,'fontname','Times')
        h=legend([num2str(Multiple(1)),' {\itσ}_{e,eff} ({\itZ}_{B#4})'],...
                 [num2str(Multiple(2)),' {\itσ}_{e,eff} ({\itZ}_{B#4})'],...
                 [num2str(Multiple(3)),' {\itσ}_{e,eff} ({\itZ}_{B#4})'],'Location','northwest');
end
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')
set(gca,'FontSize',15,'ygrid','on')

switch input
    case 'De_eff_sep'
        title('Figure 7 (h) {\itD}_{e,eff3}')
        axis([10^-3 10^5 -0.5 10])
    case 'De_eff'
        title('Figure 8 (f) {\itD}_{e,eff}')
        axis([10^-3 10^5 -2.5 50])
    case 'sigma_eff'
        title('Figure 9 (f left) {\itσ}_{eff}')
        axis([10^-3 10^5 -5 100])
end

%% absolute difference
switch input
    case 'sigma_eff'

figure(1)
subplot(2,3,3)
dq = jet(length(Multiple));       % 4 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)-1      % 4 lines in plot are taken
semilogx(out{1,1}.f,Abso.neg(:,d) * 1e3,'color',dq(i,:),'linestyle','--','linewidth',3)
    i = i+1;
end
% Represented by log axis
set(gca,'XScale','log')
% Set the properties of the coordinate axis.
xlabel('{\itf} (Hz)','fontsize',12,'fontname','Times')
ylabel('Δ{\itZ}_C (mΩ m^2)','fontsize',12,'fontname','Times')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
h=legend([num2str(Multiple(1)),' {\itσ}_{e,eff1} ({\itZ}_{B#1})'],...
         [num2str(Multiple(2)),' {\itσ}_{e,eff1} ({\itZ}_{B#1})'],...
         [num2str(Multiple(3)),' {\itσ}_{e,eff1} ({\itZ}_{B#1})'],'Location','northwest');
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')
title('Figure 9 (b right) {\itσ}_{eff1})')
axis([10^-3 10^5 -0.033 0.09])

subplot(2,3,6)
dq = jet(length(Multiple));       % 4 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)-1      % 4 lines in plot are taken
semilogx(out{1,1}.f,Abso.pos(:,d) * 1e3,'color',dq(i,:),'linestyle','--','linewidth',3)
    i = i+1;
end
% Represented by log axis
set(gca,'XScale','log')
% Set the properties of the coordinate axis.
xlabel('{\itf} (Hz)','fontsize',12,'fontname','Times')
ylabel('Δ{\itZ}_C (mΩ m^2)','fontsize',12,'fontname','Times')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
h=legend([num2str(Multiple(1)),' {\itσ}_{e,eff2} ({\itZ}_{B#2})'],...
         [num2str(Multiple(2)),' {\itσ}_{e,eff2} ({\itZ}_{B#2})'],...
         [num2str(Multiple(3)),' {\itσ}_{e,eff2} ({\itZ}_{B#2})'],'Location','northwest');
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')
title('Figure 9 (d right) {\itσ}_{eff2})')
axis([10^-3 10^5 -0.033 0.09])

figure(2)
subplot(2,3,6)
dq = jet(length(Multiple));       % 4 colours are generated
hold on
i = 1;
for d = 1:length(Multiple)-1      % 4 lines in plot are taken
semilogx(out{1,1}.f,Abso.cell(:,d) * 1e3,'color',dq(i,:),'linestyle','--','linewidth',3)
    i = i+1;
end
% Represented by log axis
set(gca,'XScale','log')
% Set the properties of the coordinate axis.
xlabel('{\itf} (Hz)','fontsize',12,'fontname','Times')
ylabel('Δ{\itZ}_C (mΩ m^2)','fontsize',12,'fontname','Times')
% Set grid.
set(gca,'FontSize',15,'xgrid','on')
set(gca,'FontSize',15,'ygrid','on')
h=legend([num2str(Multiple(1)),' {\itσ}_{e,eff} ({\itZ}_{B#4})'],...
         [num2str(Multiple(2)),' {\itσ}_{e,eff} ({\itZ}_{B#4})'],...
         [num2str(Multiple(3)),' {\itσ}_{e,eff} ({\itZ}_{B#4})'],'Location','northwest');
set(h,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')
title('Figure 9 (f right) {\itσ}_{eff})')
axis([10^-3 10^5 -0.033 0.09])

end

%% origin
Origin{1,1} = [Cell{1,1}.Nyquist_Z_neg...    % negative electrode
               Cell{2,1}.Nyquist_Z_neg...
               Cell{3,1}.Nyquist_Z_neg...
               Cell{4,1}.Nyquist_Z_neg...
               Cell{1,1}.Nyquist_Z_pos...    % positive electrode
               Cell{2,1}.Nyquist_Z_pos...
               Cell{3,1}.Nyquist_Z_pos...
               Cell{4,1}.Nyquist_Z_pos...
               Cell{1,1}.Nyquist_Z_sep...    % separator
               Cell{2,1}.Nyquist_Z_sep...
               Cell{3,1}.Nyquist_Z_sep...
               Cell{4,1}.Nyquist_Z_sep...
               Cell{1,1}.Nyquist_Z_cell...   % cell
               Cell{2,1}.Nyquist_Z_cell...
               Cell{3,1}.Nyquist_Z_cell...
               Cell{4,1}.Nyquist_Z_cell] * 1e3;

% relative difference
Origin{2,1} = [out{1,1}.f Rela.neg Rela.pos Rela.sep Rela.cell] * 100 ;

% absolute difference
Origin{3,1} = [out{1,1}.f Abso.neg Abso.pos Abso.sep Abso.cell] * 1e3 ;

% max relative difference
Origin{4,1} = [max(Rela.neg(:,1))  max(Rela.neg(:,2))  max(Rela.neg(:,3));...
               max(Rela.pos(:,1))  max(Rela.pos(:,2))  max(Rela.pos(:,3));...
               max(Rela.sep(:,1))  max(Rela.sep(:,2))  max(Rela.sep(:,3));...
               max(Rela.cell(:,1)) max(Rela.cell(:,2)) max(Rela.cell(:,3))] * 100;

end
