clear; clc; close all;
path = fileparts(which('femplot.m'));
load('diamond.mat');


tris = [];
for i = 1:length(triangles)
    if length(triangles{i}) == 3
        tris = [tris; triangles{i}];
    end
end
tris = tris + 1;

tetras = [];
for i = 1:length(triangles)
    if length(triangles{i}) == 4
        tetras = [tetras; triangles{i}];
    end
end
tetras = tetras + 1;

myblue = [0, 94, 255]/255;
mylightblue = [0, 168, 255]/255;

c = ones(size(points,1),1);
cm = ones(size(tetras,1),1);

% fig = figure(); hold on;
% set(fig,'Visible', 'on', 'color', [1,1,1], 'Position', [1, 1, 900,900]);
% tetramesh(tetras, points, cm, 'Edgecolor','none','Facecolor',myblue)
% view(0, 45)
% axis equal
% axis off
% filename = strcat(path, '/perspective_transparent');
% export_fig(filename, '-png', '-m4','-transparent')

points = [points(:,3), points(:,1), points(:,2)];
fig = figure(); hold on;
set(fig,'Visible', 'off', 'color', [1,1,1], 'Position', [1, 1, 900,900]);
tetramesh(tetras, points, cm, 'Edgecolor',myblue,'Facecolor',mylightblue,'Facealpha',0.5)
view(10, 10)
axis equal
axis off
filename = strcat(path, '/diamond_transparent_mesh_rot');
export_fig(filename, '-png', '-m4','-transparent')



% fig = figure(); hold on;
% set(fig,'Visible', 'on', 'color', [1,1,1], 'Position', [1, 1, 900,900]);
% trisurf(tris, points(:,3), points(:,2), points(:,1), c, 'Edgecolor','none','Facecolor',myblue)
% view(0, 45)
% axis equal
% axis off
% filename = strcat(path, '/perspective_transparent');
% export_fig(filename, '-png', '-m4','-transparent')


% fig = figure(); hold on;
% set(fig,'Visible', 'off', 'color', [1,1,1], 'Position', [1, 1, 900,300]);
% trisurf(tris, points(:,3), points(:,1), points(:,2), c, 'Edgecolor',myblue,'Facecolor',mylightblue,'Facealpha',0.5)
% view(0, 0)
% axis equal
% axis off
% filename = strcat(path, '/trunk_transparent_mesh');
% export_fig(filename, '-png', '-m4','-transparent')
