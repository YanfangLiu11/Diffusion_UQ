% Specify the directory and filename
savedir = '/Users/yij/Library/CloudStorage/GoogleDrive-yanfang6525@gmail.com/My Drive/ORNL/Conditional Diffusion model/Conditional diffusion/code/python_code/1D_example/plot_result_final/matlab_plot/';

filename = 'sample_data.mat';
filepath = fullfile(savedir, filename);
% Load the data
data1 = load(filepath);
x_sample = data1.x_sample;
y_sample = data1.y_sample;

% Access the data
filename = 'labeled_data.mat';
filepath = fullfile(savedir, filename);
% Load the data
data2 = load(filepath);
y0_train= data2.y0_train;
zT = data2.zT;
xTrain = data2.xTrain;

filename = 'generated_data.mat';
filepath = fullfile(savedir, filename);
data3 = load(filepath);
xx_grid = data3.xx_grid ;
yy_grid = data3.yy_grid ;
f_hat = data3.f_hat ;

%%
% Define the figure and subplot
fig = figure('Position', [100, 100, 400, 500]); % Control the figure's size
ax2 = axes('Parent', fig, 'OuterPosition', [0, 0, 1, 1]); % Fill the figure window
scatter(x_sample, y_sample, 2.5, 'red');

% Setting axes properties
ax2.FontSize = 14; 
xticks(linspace(-2, 2, 5));
xlim([-2.1, 2.1]);
yticks(linspace(0, 4, 5));
ylim([-0.4, 4.4]);

% Thicker tick marks
ax2.TickLength = [0.005, 0.005]; % Makes tick marks longer and more visible

% Setting thicker axes lines
ax2.XAxis.LineWidth = 2.2; % Thicker X-axis line
ax2.YAxis.LineWidth = 2.2; % Thicker Y-axis line

% Text labels
text(0, min(get(gca, 'YLim')) - 0.2, 'X', 'FontSize', 18, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
text(min(get(gca, 'XLim')) - 0.2, 2, 'Y', 'FontSize', 18, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

% Adjust figure and export
fig.Units = 'inches';
ax2.Units = 'normalized';
ax2.Position = [0.08, 0.08, 0.89, 0.89];  % Adjust as needed to reduce white space
exportgraphics(ax2, 'Sample_data.png', 'ContentType', 'vector', 'Resolution', 300);

% Display the figure
figure(fig);
%%

% Define the figure and subplot
fig = figure('Position', [100, 100, 800, 600]); % Control the figure's size
ax2 = axes('Parent', fig);
hold on;
view(45, 50); % Set the view angle (azimuth, elevation)

% Create a 3D scatter plot with colormap based on xTrain_p or another color_variable
scatter3(ax2, y0_train, zT, xTrain,  3.0,  xTrain);

% Calculate midpoints for labels
midX = mean(get(gca, 'XLim'));
midY = mean(get(gca, 'YLim'));
midZ = mean(get(gca, 'ZLim'));

% Custom text labels
text(2, min(get(gca, 'YLim')), min(get(gca, 'ZLim')), 'Y', 'FontSize', 20, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
text(max(get(gca, 'XLim'))+0.7, 0, min(get(gca, 'ZLim')), 'Z', 'FontSize', 20, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
text(min(get(gca, 'XLim')), min(get(gca, 'YLim'))-1.0, 0, 'X', 'FontSize', 20, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');


xlim([-1, 5]);
xticks(linspace(0, 4, 5));
ylim([-4, 4]);
yticks(linspace(-4, 4, 5));
zlim([-2.5, 2.5]);
zticks(linspace(-2, 2, 3));


ax2.FontSize = 14; 
ax2.GridAlpha = 0.08;
ax2.LineWidth = 2; 

hold off;

grid on

colormap(ax2, 'parula'); 

saveas(fig, fullfile(savedir, 'labeled_data.png'));

% Display the figure
figure(fig);


%%
% Define the figure and subplot
fig = figure('Position', [100, 100, 800, 600]); % Control the figure's size
ax2 = axes('Parent', fig);
hold on;
view(45, 50); % Set the view angle (azimuth, elevation)

% Create a 3D scatter plot with colormap based on xTrain_p or another color_variable
mesh(xx_grid,yy_grid,  f_hat);
% s.FaceColor = 'flat';

% Set axis limits
xlim([-1, 5]);
ylim([-4, 4]);
zlim([-2.5, 2.5]);

% Set axis ticks
xticks(linspace(0, 4, 5));
yticks(linspace(-4, 4, 5));
zticks(linspace(-2, 2, 3));

% Custom text labels
text(2, min(get(gca, 'YLim'))-0.8, min(get(gca, 'ZLim')), 'Y', 'FontSize', 20, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
text(max(get(gca, 'XLim'))+0.7, 0, min(get(gca, 'ZLim')), 'Z', 'FontSize', 20, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
text(min(get(gca, 'XLim')), min(get(gca, 'YLim'))-1.0, 0, 'X', 'FontSize', 20, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');


ax2.FontSize = 14; 
ax2.GridAlpha = 0.08;
ax2.LineWidth = 2; 

hold off;

grid on

saveas(fig, fullfile(savedir, 'Generated_data.png'));

% Display the figure
figure(fig);


% 
% %%
% % Define the figure
% fig = figure('Position', [100, 100, 3000, 300]); % Control the figure's size
% 
% % First subplot - 2D scatter plot
% ax1 = subplot(1,3, 1);  % Create a subplot in the first slot
% scatter(ax1, x_sample, y_sample, 2.5, 'red');
% ax1.FontSize = 14; 
% xticks(ax1, linspace(-2, 2, 5));
% xlim(ax1, [-2.5, 2.5]);
% yticks(ax1, linspace(0, 4, 5));
% ylim(ax1, [-1, 5]);
% xlabel('X');
% ylabel('Y');
% 
% % Second subplot - 3D scatter plot
% ax2 = subplot(1,3, 2, 'Parent', fig);  % Create a subplot in the second slot
% view(ax2, 45, 50); % Set the view angle (azimuth, elevation)
% scatter3(ax2, y0_train, zT, xTrain, 3.0, xTrain);
% ax2.FontSize = 14; 
% xlim(ax2, [-1, 5]);
% ylim(ax2, [-4, 4]);
% zlim(ax2, [-2.5, 2.5]);
% xticks(ax2, linspace(0, 4, 5));
% yticks(ax2, linspace(-4, 4, 5));
% zticks(ax2, linspace(-2, 2, 3));
% xlabel(ax2, 'Y');
% ylabel(ax2, 'Z');
% zlabel(ax2, 'X');
% grid(ax2, 'on');
% colormap(ax2, 'parula');
% 
% % Third subplot - 3D mesh plot
% ax3 = subplot(1,3, 3, 'Parent', fig);  % Create a subplot in the third slot
% view(ax3, 45, 50); % Set the view angle (azimuth, elevation)
% mesh(ax3, xx_grid, yy_grid, f_hat);
% ax3.FontSize = 14; 
% xlim(ax3, [-1, 5]);
% ylim(ax3, [-4, 4]);
% zlim(ax3, [-2.5, 2.5]);
% xticks(ax3, linspace(0, 4, 5));
% yticks(ax3, linspace(-4, 4, 5));
% zticks(ax3, linspace(-2, 2, 3));
% xlabel(ax3, 'Y');
% ylabel(ax3, 'Z');
% zlabel(ax3, 'X');
% grid(ax3, 'on');
% 
% % Optionally, you can save the entire figure with all three subplots
% % saveas(fig, fullfile(savedir, 'Combined_Plots.png'));
% 
% % Display the figure
% figure(fig);
% 
% 
