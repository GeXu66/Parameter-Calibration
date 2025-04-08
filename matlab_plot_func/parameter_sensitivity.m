close all; clear; clc;
plot_parameter_sensitivity('N_parallel')
plot_parameter_sensitivity('electrode_height')
plot_parameter_sensitivity('electrode_width')
plot_parameter_sensitivity('Initial_concentration_in_positive_electrode')
plot_parameter_sensitivity('Negative_electrode_active_material_volume_fraction')
plot_parameter_sensitivity('Negative_electrode_thickness')
%% function
function plot_parameter_sensitivity(param_name)
    % Creates publication-quality plots from parameter sensitivity data
    % 
    % Args:
    %   param_name: String, name of the parameter (e.g., 'N_parallel')
    
    % Directory settings
    input_dir = ['../sensitivity_results/plot_change/', param_name];
    output_dir = ['./sensitivity_change/', param_name];
    
    % Create output directory if it doesn't exist
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Define C-rates
    c_rates = [0.1, 0.2, 0.33, 1];
    
    % Read parameter values
    param_values_file = fullfile(input_dir, [param_name, '_param_values.csv']);
    param_data = readtable(param_values_file);
    param_values = param_data.param_values;
    
    % Store all data for the combined plot
    all_data = cell(length(c_rates), 1);
    max_time_across_all = 0;  % Track maximum time across all datasets
    
    % Create individual plots for each C-rate
    for i = 1:length(c_rates)
        c_rate = c_rates(i);
        
        % Format C-rate string for filenames
        if c_rate == 0.33
            c_rate_str = '0_33';
        else
            c_rate_str = strrep(num2str(c_rate), '.', '_');
        end
        
        % Read data file
        data_file = fullfile(input_dir, [param_name, '_', num2str(c_rate), 'C.csv']);
        if ~exist(data_file, 'file')
            warning('File not found: %s', data_file);
            continue;
        end
        
        % Read the CSV file
        data = readtable(data_file);
        if ~strcmp(data.Properties.VariableNames{1}, 'time')
            data.Properties.VariableNames{1} = 'time';
        end
        
        % Get time data (first column)
        time = data.time / 3600; % Convert to hours
        % Update max time if needed
        current_max_time = max(time);
        if current_max_time > max_time_across_all
            max_time_across_all = current_max_time;
        end
        
        % Store data for combined plot
        all_data{i} = struct('time', time, 'data', data, 'c_rate', c_rate);
        
        % Create figure for this C-rate
        fig = figure('Position', [100, 100, 800, 600]);
        hold on;
        
        % Create colormap - using MATLAB's built-in parula (perceptually uniform)
        cmap = parula(width(data)-1); % One color for each data column (excluding time)
        
        % Plot each parameter value - USING COLUMN INDEX INSTEAD OF NAMES
        for j = 2:width(data) % Start from column 2 (first column is time)
            voltage = data{:, j};
            if any(~isnan(voltage)) % Check if there's valid data
                plot(time, voltage, 'LineWidth', 1.5, 'Color', cmap(j-1,:));
            end
        end
        % Set xlim to max time + 60
        xlim([0, max(time) + 60/3600]);  % Add 60 seconds, converted to hours
        
        % Add labels and title
        xlabel('Time (h)', 'FontSize', 14, 'FontWeight', 'bold');
        ylabel('Terminal Voltage (V)', 'FontSize', 14, 'FontWeight', 'bold');
        % title(['Discharge Curves at ', num2str(c_rate), 'C: Effect of ', strrep(param_name, '_', '\_')], ...
        %       'FontSize', 16, 'FontWeight', 'bold');
        
        % Add colorbar
        cb = colorbar;
        colormap(cmap);
        caxis([min(param_values), max(param_values)]);
        param_display = strrep(param_name, '_', ' ');
        words = strsplit(param_display, ' ');
        for w = 1:length(words)
            if ~isempty(words{w})
                words{w}(1) = upper(words{w}(1));
            end
        end
        param_display = strjoin(words, ' ');
        ylabel(cb, param_display, 'FontSize', 14, 'FontWeight', 'bold');
        
        % Format plot
        set(gca, 'FontSize', 12, 'LineWidth', 1.5, 'Box', 'on');
        grid on;
        
        % Adjust figure
        set(gcf, 'Color', 'white');

        if strcmp(param_name, 'Initial_concentration_in_positive_electrode') && c_rate == 1
            zoom_rect = rectangle('Position', [0, 3.1, 0.1, 0.3], 'EdgeColor', [0.8, 0.2, 0.2], ...
                 'LineWidth', 1.5, 'LineStyle', '-');
            annotation('arrow', [0.2, 0.25], [0.8, 0.75], 'Color', [0.8, 0.2, 0.2], 'LineWidth', 1.5);
            % Create an inset axes
            inset_axes = axes('Position', [0.2, 0.5, 0.3, 0.25]);  % [left, bottom, width, height]
            hold on;
            
            % Plot the same data in the inset
            for j = 2:width(data)
                voltage = data{:, j};
                if any(~isnan(voltage))
                    plot(time, voltage, 'LineWidth', 1.5, 'Color', cmap(j-1,:));
                end
            end
            
            % Set the inset limits to the specified zoom region
            xlim([0, 0.1]);  % x from 0 to 0.1
            ylim([3.1, 3.4]); % y from 3.1 to 3.4
            
            % Format the inset
            set(inset_axes, 'FontSize', 10, 'LineWidth', 1.2, 'Box', 'on');
            grid on;
            
            % Go back to the main axes for the rest of the code
            axes(gca);
        end
        
        % Save figure
        output_file = fullfile(output_dir, [param_name, '_', c_rate_str, 'C.png']);
        exportgraphics(fig, output_file, 'Resolution', 300);
       
    end
    
    % Create combined plot with all C-rates
    fig_combined = figure('Position', [100, 100, 800, 600]);
    hold on;
    
    % Define colormaps for each C-rate using MATLAB's built-in scientific colormaps
    colormaps = {
        parula(width(all_data{1}.data)-1),   % 0.1C - blue to yellow
        cool(width(all_data{2}.data)-1),     % 0.2C - cyan to magenta
        hot(width(all_data{3}.data)-1),      % 0.33C - black to red to yellow
        turbo(width(all_data{4}.data)-1)     % 1C - improved rainbow
    };
    
    % Line styles for different C-rates
    line_styles = {'-', '-', '-', '-'};
    
    % Plot data for each C-rate
    legendEntries = cell(1, length(c_rates) * (width(data)-1));
    legendCount = 0;
    
    for i = 1:length(c_rates)
        if isempty(all_data{i})
            continue;
        end
        
        data = all_data{i}.data;
        time = all_data{i}.time;
        c_rate = all_data{i}.c_rate;
        
        % Adjust colormap size if needed
        num_cols = width(data)-1;
        if size(colormaps{i}, 1) ~= num_cols
            colormaps{i} = eval([func2str(colormaps{i}), '(', num2str(num_cols), ')']);
        end
        
        for j = 2:width(data) % Start from column 2 (skip time)
            voltage = data{:, j};
            
            if any(~isnan(voltage)) % Check if there's valid data
                p = plot(time, voltage, line_styles{i}, 'LineWidth', 1.0, 'Color', colormaps{i}(j-1,:));
                
                % Add to legend only for a subset of lines to keep it readable
                if mod(j, 4) == 0  % Add every 4th line to the legend
                    legendCount = legendCount + 1;
                    param_val = param_values(j-1); % j-1 because j starts from 2
                    legendEntries{legendCount} = [num2str(c_rate) 'C, ' param_name '=' num2str(param_val, '%.4g')];
                    p.DisplayName = legendEntries{legendCount};
                else
                    p.HandleVisibility = 'off';
                end
            end
        end
    end
    xlim([0, max_time_across_all + 60/3600]);
    % Add labels and title
    xlabel('Time (h)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Terminal Voltage (V)', 'FontSize', 14, 'FontWeight', 'bold');
    % title(['Effect of ', strrep(param_name, '_', '\_'), ' on Discharge Curves at Different C-rates'], ...
    %       'FontSize', 16, 'FontWeight', 'bold');
    
    % Add legend and format it
    % legendEntries = legendEntries(1:legendCount);
    % lgd = legend(legendEntries, 'Location', 'eastoutside', 'FontSize', 10);
    % lgd.Title.String = 'Parameter Values';
    
    % Format plot
    set(gca, 'FontSize', 12, 'LineWidth', 1.5, 'Box', 'on');
    grid on;
    
    % Adjust figure
    set(gcf, 'Color', 'white');
    
    % Save figure
    output_file = fullfile(output_dir, [param_name, '_all_C_rates.png']);
    exportgraphics(fig_combined, output_file, 'Resolution', 300);
    
    
    disp(['Plots saved to: ', output_dir]);
end

