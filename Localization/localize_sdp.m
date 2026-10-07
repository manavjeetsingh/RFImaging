clear; clc; close all;

% Paths to SDPwrapper, SFSDP and its subprograms
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
addpath(fullfile(thisDir, 'SFSDP', 'SFSDP'));
addpath(fullfile(thisDir, 'SFSDP', 'SFSDP', 'subprograms'));

%% Settings
sDim        = 3;      % 2 or 3
noOfSensors = 500;
noOfAnchors = 12;
areaSize    = 4;      % sensors/anchors lie in [0, areaSize]^sDim (m)
radioRange  = 2;      % only pairs closer than this are measured (m)
noisyFac    = 0.00;   % distance noise: d * (1 + noisyFac*randn)
rng(1);               % reproducible

%% Generate dummy locations
% sDim x (noOfSensors + noOfAnchors); anchors are the last noOfAnchors columns
sensors = areaSize * rand(sDim, noOfSensors);
anchors = areaSize * rand(sDim, noOfAnchors);
trueLocationMatrix = [sensors, anchors];

%% Generate noisy distance measurements
% noOfSensors x (noOfSensors + noOfAnchors). Entry (i,j) is the noisy distance
% between sensor i and node j if they are within radio range, else 0.
% Sensor-sensor block is upper triangular (i < j), as SFSDP expects.
n = noOfSensors + noOfAnchors;
distanceMeasurements = zeros(noOfSensors, n);
for i = 1:noOfSensors
    for j = i+1:n
        d = norm(trueLocationMatrix(:, i) - trueLocationMatrix(:, j));
        if d < radioRange
            distanceMeasurements(i, j) = d * max(1 + noisyFac*randn, 0.1);
        end
    end
end
distanceMeasurements = sparse(distanceMeasurements);
fprintf('%d distance measurements\n', nnz(distanceMeasurements));

%% Parameters
pars = struct();
pars.sDim        = sDim;
pars.noOfSensors = noOfSensors;
pars.noOfAnchors = noOfAnchors;
pars.edgeAlgo    = "all";     % use every measured edge
pars.SDPsolver   = 'sedumi';
pars.localMethodSW = 1;       % refine SDP solution with gradient method
pars.show_plots  = 0;
pars.verbose     = 0;

%% Localize
[cost, estimatedlocationMatrix, before_degreeList, degreeList] = ...
    SDPwrapper(pars, distanceMeasurements, trueLocationMatrix);

%% Results
trueS = trueLocationMatrix(:, 1:noOfSensors);
estS  = estimatedlocationMatrix(:, 1:noOfSensors);
locErr = vecnorm(estS - trueS);
fprintf('Distance RMSE (cost): %.4f m\n', cost);
fprintf('Location error: mean %.4f m, median %.4f m, max %.4f m\n', ...
    mean(locErr), median(locErr), max(locErr));

figure; hold on; grid on; axis equal;
plotPoints(anchors, 'ks', 'MarkerFaceColor', 'k');
plotPoints(trueS, 'bo');
plotPoints(estS, 'r*');
for s = 1:noOfSensors
    plotPoints([trueS(:, s), estS(:, s)], 'r-');
end
if sDim == 3, view(3); end
legend('Anchors', 'True', 'Estimated', 'Location', 'bestoutside');
title(sprintf('SDP localization (%dD), mean error %.3f m', sDim, mean(locErr)));

%% Helpers
function plotPoints(X, varargin)
% Plot columns of X (2 x N or 3 x N) with plot or plot3.
if size(X, 1) == 2
    plot(X(1, :), X(2, :), varargin{:});
else
    plot3(X(1, :), X(2, :), X(3, :), varargin{:});
end
end
