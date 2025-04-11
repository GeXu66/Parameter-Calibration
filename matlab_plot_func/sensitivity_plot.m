% plot_sensitivity_analysis.m
% Script to visualize weighted sensitivity analysis results with Nature-style formatting
close all;
%% Read the sensitivity analysis data
filepath = fullfile('../sensitivity_results', 'weighted_sensitivity_indices.csv');
data = readtable(filepath);
save_dir = fullfile('./sensitivity_plot');
% Check if save_dir exists, if not, create it
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
    fprintf('Created directory: %s\n', save_dir);
end
%% Define parameter categories, symbols, and colors
% Parameter categories
geometric = {'N_parallel', 'electrode_height', 'electrode_width', ...
             'Negative_electrode_thickness', 'Positive_electrode_thickness'};
         
structural = {'Negative_particle_radius', 'Positive_particle_radius', ...
              'Negative_electrode_active_material_volume_fraction', ...
              'Positive_electrode_active_material_volume_fraction', ...
              'Negative_electrode_porosity', 'Positive_electrode_porosity', ...
              'Separator_porosity', 'Maximum_concentration_in_negative_electrode', ...
              'Maximum_concentration_in_positive_electrode'};
          
transport = {'Negative_electrode_diffusivity', 'Positive_electrode_diffusivity', ...
             'Negative_electrode_Bruggeman_coefficient', 'Positive_electrode_Bruggeman_coefficient', ...
             'Negative_electrode_conductivity', 'Positive_electrode_conductivity'};
         
initial = {'Initial_concentration_in_negative_electrode', ...
           'Initial_concentration_in_positive_electrode'};

% LaTeX symbols for each parameter
param_symbols = containers.Map();
param_symbols('N_parallel') = '$N_{wind}$';
param_symbols('electrode_height') = '$H_{elec}$';
param_symbols('electrode_width') = '$W_{elec}$';
param_symbols('Negative_electrode_thickness') = '$L_n$';
param_symbols('Positive_electrode_thickness') = '$L_p$';
param_symbols('Negative_particle_radius') = '$R_{p,n}$';
param_symbols('Positive_particle_radius') = '$R_{p,p}$';
param_symbols('Negative_electrode_active_material_volume_fraction') = '$\varepsilon_{s,n}$';
param_symbols('Positive_electrode_active_material_volume_fraction') = '$\varepsilon_{s,p}$';
param_symbols('Negative_electrode_porosity') = '$\varepsilon_{e,n}$';
param_symbols('Positive_electrode_porosity') = '$\varepsilon_{e,p}$';
param_symbols('Separator_porosity') = '$\varepsilon_{e,sep}$';
param_symbols('Maximum_concentration_in_negative_electrode') = '$c_{s,n}^{max}$';
param_symbols('Maximum_concentration_in_positive_electrode') = '$c_{s,p}^{max}$';
param_symbols('Negative_electrode_diffusivity') = '$D_{s,n}$';
param_symbols('Positive_electrode_diffusivity') = '$D_{s,p}$';
param_symbols('Negative_electrode_Bruggeman_coefficient') = '$\beta_n$';
param_symbols('Positive_electrode_Bruggeman_coefficient') = '$\beta_p$';
param_symbols('Negative_electrode_conductivity') = '$\sigma_n$';
param_symbols('Positive_electrode_conductivity') = '$\sigma_p$';
param_symbols('Initial_concentration_in_negative_electrode') = '$c_{s,n}^{init}$';
param_symbols('Initial_concentration_in_positive_electrode') = '$c_{s,p}^{init}$';

% Category colors (Nature-style color palette)
colors = struct();
colors.geometric = [0, 0.447, 0.741];      % Blue
colors.structural = [0.85, 0.325, 0.098];  % Red
colors.transport = [0.466, 0.674, 0.188];  % Green
colors.initial = [0.494, 0.184, 0.556];    % Purple

% Category labels
category_labels = {'Geometric Parameters', 'Structural Parameters', ...
                  'Transport Parameters', 'Initial State Parameters'};

%% Prepare data for plotting
% Create tables for each category
geo_data = data(ismember(data.Parameter, geometric), :);
struct_data = data(ismember(data.Parameter, structural), :);
trans_data = data(ismember(data.Parameter, transport), :);
init_data = data(ismember(data.Parameter, initial), :);

% Sort each category by Total-Order sensitivity
[~, idx] = sort(geo_data.Weighted_ST, 'descend');
geo_data = geo_data(idx, :);
[~, idx] = sort(struct_data.Weighted_ST, 'descend');
struct_data = struct_data(idx, :);
[~, idx] = sort(trans_data.Weighted_ST, 'descend');
trans_data = trans_data(idx, :);
[~, idx] = sort(init_data.Weighted_ST, 'descend');
init_data = init_data(idx, :);

% Combine sorted data
sorted_data = [geo_data; struct_data; trans_data; init_data];

% Create symbols array in the same order as sorted_data
symbols = cell(height(sorted_data), 1);
for i = 1:height(sorted_data)
    param_name = sorted_data.Parameter{i};
    if isKey(param_symbols, param_name)
        symbols{i} = param_symbols(param_name);
    else
        symbols{i} = strrep(param_name, '_', '\_');
    end
end

% Calculate indices for category separators
geo_end = height(geo_data);
struct_end = geo_end + height(struct_data);
trans_end = struct_end + height(trans_data);
init_end = trans_end + height(init_data);
category_ends = [geo_end, struct_end, trans_end, init_end];

%% Set up Nature-style figure formatting
% Configure figure style
set(0, 'DefaultFigureColor', [1 1 1]);
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultTextFontName', 'Arial');
set(0, 'DefaultAxesFontSize', 13);
set(0, 'DefaultTextFontSize', 13);
set(0, 'DefaultLineLineWidth', 1.5);

%% Figure 1: Total-Order Sensitivity Indices
fig1 = figure('Position', [100, 100, 800, 600]);
ax1 = axes();

% Create horizontal bar plot with grouped colors
hold on;
b = barh(1:height(sorted_data), sorted_data.Weighted_ST);
b.FaceColor = 'flat';

% Color bars by category
for i = 1:height(geo_data)
    b.CData(i,:) = colors.geometric;
end
for i = geo_end+1:struct_end
    b.CData(i,:) = colors.structural;
end
for i = struct_end+1:trans_end
    b.CData(i,:) = colors.transport;
end
for i = trans_end+1:init_end
    b.CData(i,:) = colors.initial;
end

% Add separator lines between categories
if ~isempty(geo_data) && ~isempty(struct_data)
    line([0 max(sorted_data.Weighted_ST)*1.05], [geo_end+0.5 geo_end+0.5], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');
end
if ~isempty(struct_data) && ~isempty(trans_data)
    line([0 max(sorted_data.Weighted_ST)*1.05], [struct_end+0.5 struct_end+0.5], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');
end
if ~isempty(trans_data) && ~isempty(init_data)
    line([0 max(sorted_data.Weighted_ST)*1.05], [trans_end+0.5 trans_end+0.5], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');
end

% Add category labels
y_pos = [(1+geo_end)/2, (geo_end+1+struct_end)/2, (struct_end+1+trans_end)/2, (trans_end+1+init_end)/2];
x_pos = max(sorted_data.Weighted_ST) * 0.7;

for i = 1:length(category_labels)
    if i == 1 && ~isempty(geo_data)
        text(x_pos, y_pos(i), category_labels{i}, 'Color', colors.geometric, 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'left');
    elseif i == 2 && ~isempty(struct_data)
        text(x_pos, y_pos(i), category_labels{i}, 'Color', colors.structural, 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'left');
    elseif i == 3 && ~isempty(trans_data)
        text(x_pos, y_pos(i), category_labels{i}, 'Color', colors.transport, 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'left');
    elseif i == 4 && ~isempty(init_data)
        text(x_pos, y_pos(i), category_labels{i}, 'Color', colors.initial, 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'left');
    end
end
% Configure axes

set(ax1, 'YTick', 1:height(sorted_data));
set(ax1, 'YTickLabel', symbols);
set(ax1, 'TickLabelInterpreter', 'latex');
set(ax1, 'FontSize', 12);
xlabel('Total-Order Sensitivity Index', 'FontSize', 12);
% title('Total-Order Sensitivity Indices for Battery Parameters', 'FontSize', 14);
grid on;
box on;

% Adjust axes limits
xlim([0, max(sorted_data.Weighted_ST)*1.05]);
ylim([0.5, height(sorted_data)+0.5]);
ax3 = ax1;
% Save figure
exportgraphics(ax1, fullfile(save_dir, 'total_order_sensitivity.png'), 'Resolution', 300);
exportgraphics(ax3, fullfile(save_dir, 'total_order_sensitivity.pdf'), 'Resolution', 300);

%% Figure 2: First-Order Sensitivity Indices
% Sort data by first-order sensitivity
[~, idx] = sort(geo_data.Weighted_S1, 'descend');
geo_data_s1 = geo_data(idx, :);
[~, idx] = sort(struct_data.Weighted_S1, 'descend');
struct_data_s1 = struct_data(idx, :);
[~, idx] = sort(trans_data.Weighted_S1, 'descend');
trans_data_s1 = trans_data(idx, :);
[~, idx] = sort(init_data.Weighted_S1, 'descend');
init_data_s1 = init_data(idx, :);

% Combine sorted data
sorted_data_s1 = [geo_data_s1; struct_data_s1; trans_data_s1; init_data_s1];

% Create symbols array in the same order as sorted_data_s1
symbols_s1 = cell(height(sorted_data_s1), 1);
for i = 1:height(sorted_data_s1)
    param_name = sorted_data_s1.Parameter{i};
    if isKey(param_symbols, param_name)
        symbols_s1{i} = param_symbols(param_name);
    else
        symbols_s1{i} = strrep(param_name, '_', '\_');
    end
end

% Recalculate indices for category separators
geo_end_s1 = height(geo_data_s1);
struct_end_s1 = geo_end_s1 + height(struct_data_s1);
trans_end_s1 = struct_end_s1 + height(trans_data_s1);
init_end_s1 = trans_end_s1 + height(init_data_s1);

% Create figure
fig2 = figure('Position', [100, 100, 800, 600]);
ax2 = axes();

% Create horizontal bar plot with grouped colors
hold on;
b2 = barh(1:height(sorted_data_s1), sorted_data_s1.Weighted_S1);
b2.FaceColor = 'flat';

% Color bars by category
for i = 1:height(geo_data_s1)
    b2.CData(i,:) = colors.geometric;
end
for i = geo_end_s1+1:struct_end_s1
    b2.CData(i,:) = colors.structural;
end
for i = struct_end_s1+1:trans_end_s1
    b2.CData(i,:) = colors.transport;
end
for i = trans_end_s1+1:init_end_s1
    b2.CData(i,:) = colors.initial;
end

% Add separator lines between categories
if ~isempty(geo_data_s1) && ~isempty(struct_data_s1)
    line([0 max(sorted_data_s1.Weighted_S1)*1.05], [geo_end_s1+0.5 geo_end_s1+0.5], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');
end
if ~isempty(struct_data_s1) && ~isempty(trans_data_s1)
    line([0 max(sorted_data_s1.Weighted_S1)*1.05], [struct_end_s1+0.5 struct_end_s1+0.5], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');
end
if ~isempty(trans_data_s1) && ~isempty(init_data_s1)
    line([0 max(sorted_data_s1.Weighted_S1)*1.05], [trans_end_s1+0.5 trans_end_s1+0.5], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');
end

% Add category labels
y_pos_s1 = [(1+geo_end_s1)/2, (geo_end_s1+1+struct_end_s1)/2, (struct_end_s1+1+trans_end_s1)/2, (trans_end_s1+1+init_end_s1)/2];
x_pos_s1 = max(sorted_data_s1.Weighted_S1) * 0.7;

for i = 1:length(category_labels)
    if i == 1 && ~isempty(geo_data_s1)
        text(x_pos_s1, y_pos_s1(i), category_labels{i}, 'Color', colors.geometric, 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'left');
    elseif i == 2 && ~isempty(struct_data_s1)
        text(x_pos_s1, y_pos_s1(i), category_labels{i}, 'Color', colors.structural, 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'left');
    elseif i == 3 && ~isempty(trans_data_s1)
        text(x_pos_s1, y_pos_s1(i), category_labels{i}, 'Color', colors.transport, 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'left');
    elseif i == 4 && ~isempty(init_data_s1)
        text(x_pos_s1, y_pos_s1(i), category_labels{i}, 'Color', colors.initial, 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'left');
    end
end

% Configure axes
set(ax2, 'YTick', 1:height(sorted_data_s1));
set(ax2, 'YTickLabel', symbols_s1);
set(ax2, 'TickLabelInterpreter', 'latex');
set(ax2, 'FontSize', 12);
xlabel('First-Order Sensitivity Index', 'FontSize', 12);
% title('First-Order Sensitivity Indices for Battery Parameters', 'FontSize', 14);
grid on;
box on;

% Adjust axes limits
xlim([0, max(sorted_data_s1.Weighted_S1)*1.05]);
ylim([0.5, height(sorted_data_s1)+0.5]);

ax3 = ax2;
% Save figure
exportgraphics(ax2, fullfile(save_dir, 'first_order_sensitivity.png'), 'Resolution', 300);
exportgraphics(ax3, fullfile(save_dir, 'first_order_sensitivity.pdf'), 'Resolution', 300);

fprintf('Sensitivity plots created and saved as:\n');
fprintf('  - total_order_sensitivity.png/.pdf\n');
fprintf('  - first_order_sensitivity.png/.pdf\n');
