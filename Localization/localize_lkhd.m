clear; clc; close all;

% Path to LikelyHoodLocalization
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

%% Settings (2D only: the method draws circles on an image)
noOfSensors = 50;
noOfAnchors = 24;
areaSize    = 4;      % sensors/anchors lie in [0, areaSize]^2 (m)
radioRange  = 2;      % only pairs closer than this are measured (m)
noisyFac    = 0.05;   % coarse distance noise: d * (1 + noisyFac*randn)
phaseNoise  = 0.05;   % phase noise std (rad)
freq        = 915e6;  % main frequency (Hz)
rng(1);               % reproducible

halfLambda = 3e8 / freq / 2;   % phase is wrapped mod pi -> ambiguity of lambda/2 (m)

%% Generate dummy locations
% 2 x (noOfSensors + noOfAnchors); anchors are the last noOfAnchors columns
sensors = areaSize * rand(2, noOfSensors);
anchors = areaSize * rand(2, noOfAnchors);
trueLocationMatrix = [sensors, anchors];

%% Generate measurements
% All matrices are n x n, upper triangular (i < j), distances in metres.
n = noOfSensors + noOfAnchors;
trueDist  = zeros(n);   % true distance for every pair
linkDist  = zeros(n);   % true distance, only for measured links
distEst   = zeros(n);   % coarse (noisy) distance estimate
phase     = zeros(n);   % phase at main frequency, wrapped to [0, pi)
kEst      = zeros(n);   % estimated number of lambda/2 cycles
for i = 1:n
    for j = i+1:n
        d = norm(trueLocationMatrix(:, i) - trueLocationMatrix(:, j));
        trueDist(i, j) = d;
        if d < radioRange
            linkDist(i, j) = d;
            distEst(i, j)  = d * max(1 + noisyFac*randn, 0.1);
            phase(i, j)    = mod(2*pi*d/(2*halfLambda) + phaseNoise*randn, pi);
            wrapped        = phase(i, j) / pi * halfLambda;
            kEst(i, j)     = max(round((distEst(i, j) - wrapped) / halfLambda), 0);
        end
    end
end
fprintf('%d distance measurements\n', nnz(linkDist));

%% Parameters (positions, lim, radio_range, lambda in cm)
pars = struct();
pars.num_tags         = noOfSensors;
pars.num_anchors      = noOfAnchors;
pars.groundDistances0 = trueDist;
pars.groundDistances  = linkDist;
pars.dist_ests        = distEst;
pars.dist_ests_k      = distEst;   % unused by the method, but must exist
pars.mainfreqPhase    = phase;
pars.seedK            = kEst;
pars.scale            = 3;                  % pixels per cm
pars.lambda           = halfLambda * 100;   % cm
pars.radio_range      = radioRange * 100;   % cm
pars.lim              = areaSize * 100;     % cm
pars.circle_ind       = 3 * ones(n);        % circles drawn per link (3, 5 or 7)
pars.all_circles      = 0;
pars.main_runs        = 1;
pars.savefile         = false;

numFreqs = 1;   % only used for the output folder name
anchType = 2;   % 2 = random anchors (only used for the output folder name)

%% Localize
[X_hat_tags, orig_tag_idx, orig_anchor_idx] = LikelyHoodLocalization( ...
    pars, sensors*100, anchors*100, numFreqs, anchType);

% The function moves confidently-located tags to the end of X_hat_tags;
% put the estimates back in the original sensor order and convert to m.
order = [orig_tag_idx, orig_anchor_idx(noOfAnchors+1:end)];
estS = zeros(2, noOfSensors);
estS(:, order) = X_hat_tags / 100;

%% Results
trueS = sensors;
locErr = vecnorm(estS - trueS);
fprintf('Location error: mean %.4f m, median %.4f m, max %.4f m\n', ...
    mean(locErr), median(locErr), max(locErr));

figure; hold on; grid on; axis equal;
plot(anchors(1, :), anchors(2, :), 'ks', 'MarkerFaceColor', 'k');
plot(trueS(1, :), trueS(2, :), 'bo');
plot(estS(1, :), estS(2, :), 'r*');
for s = 1:noOfSensors
    plot([trueS(1, s), estS(1, s)], [trueS(2, s), estS(2, s)], 'r-');
end
legend('Anchors', 'True', 'Estimated', 'Location', 'bestoutside');
title(sprintf('Likelihood localization, mean error %.3f m', mean(locErr)));
