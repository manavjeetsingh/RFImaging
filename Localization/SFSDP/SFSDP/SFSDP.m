function [xMatrix,info,pars,distanceMatrix,totalMeasurements,before_degList,degreeList] ...
    = SFSDP(sDim,noOfSensors,noOfAnchors,xMatrix0,distanceMatrix0,pars)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% SFSDP --- A sparse version of FSDP
% Sunyoung Kim, Masakazu Kojima^* and Hayato Waki
% July , 2008
%
% Revised July 2009
% Sunyoung Kim, Masakazu Kojima^*, Hayato Waki and Makoto Yamashita
%
% Revised January 2010
% Sunyoung Kim, Masakazu Kojima^*, Hayato Waki and Makoto Yamashita
%
% * Department of Computing and Mathematical Sciences
%   Tokyo Institute of Technology
%   Oh-Okayama, Meguro, Tokyo 152-8552
%   e-mail: kojima@is.titech.ac.jp
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% SFSDP is proposed in the paper
% S. Kim, M. Kojima and H. Waki, "Exploiting Sparsity in SDP Relaxation 
% for Sensor Network Localization", SIAM Journal of Optimization
% Vol.20 (1) 192-215 (2209). 
%
% SFSDP is a sparse version of FSDP which was proposed in 
% P. Biswas and Y. Ye (2004) Semidefinite programming for ad hoc wireless 
% sensor network localization,in Proceedings of the third international 
% symposium on information processing in sensor networks, ACM press, 46-54. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Acknowledgment
% 
% The authors of this software package are grateful to Professor Yinyu Ye
% who provided us with the original version of FSDP, and Professor Kim Chuan Toh 
% for lots of helpful comments and providing MATLAB programs refineposition.m 
% and procrustes.m. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is a component of the software package SFSDP 
% Copyright (C) 2008 Sunyoung Kim, Masakazu Kojima and Hayato Waki
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sDim :    the dimension of the space in which sensors and anchors are
%           located; sDim is either 2 or 3. 
% noOfSensors   : the number of sensors.
% noOfAnchors   :   the number m_a of anchors; the last m_a columns of xMatrix0, 
% xMatrix0  :   sDim x n matrix of sensors' and anchors' locations 
%               in the sDim dimensional space, where n is the total number of 
%               sensors and anchors, and anchors are placed in the last m_a columns,                
%           or
%               sDim x m_a matrix of anchors in the sDim dimensional space,
%               where m_a denotes the number of anchors. 
%               If noOfAnchors == 0 then xMatrix0 can be [];
% distanceMatrix0 : the sparse (and noisy) distance matrix between xMatrix0(:,i) 
%               and xMatrix0(:,j);
%               distanceMatrix0(i,j) = the (noisy) distance between xMatrix0(:,i) 
%               and xMatrix0(:,j) if i < j; 
%               distanceMatrix0(i,j) = 0 if i >= j.
% pars --- parameters
%   parameters used for SeDuMi, default values: 
%       pars.eps = 1.0e-7 for sedumi;
%       pars.free = 0 for sedumi;
%       pars.fid = 0 for sedumi and sdpa;
%       pars.alg, pars.theta, pars.beta, pars.vplot, pars.maxiter for sedumi; 
%       See the manual of SeDuMi. 
%   parameters used for SDPA, default values;
%       OPTION.epsilonStar  = min([pars.eps,1.0e-7]);
%       OPTION.epsilonDash  = min([pars.eps,1.0e-7]);
%       OPTION.print = 'display' if pars.fid = 1;
%                    = 'no' if pars.fid = 0; 
%       OPTION.print = pars.fid if isstr(pars.fid); 
%       See the manual of SDPA. 
%   parameters used for SFSDP
%       pars.SDPsolver  = 'sedumi' to solver the SDP problem by SeDuMi; default
%                       = 'sdpa' to solver the SDP problem by SDPA
%       pars.sparseSW   = 0 ---> a modified version of FSDP
%                       = 1 ---> SFSDP
%       pars.minDegree = sDim + 2;
%           This parameter control the minimum degree of each sensors. If
%           we increase pars.minDegree, we expect a stronger relaxation but
%           more larger cputime. 
%       pars.edgeSelectionSW = 1; 
%           = 0 ---> using all edges to construct an SDP relaxation problem
%           = 1 ---> selection of edges suitable for FSDP with pars.sparseSW = 0 
%                    and SFSDP with pars.sparseSW = 1. 
%       pars.objSW 
%           = 0 if #anchors >= sDim+1 and no noise 
%                   ---> solving quadratic equalities with objective function = 0
%           = 1 if #anchors >= sDim+1 and noise 
%                   ---> minimizing the 1-norm error
%           = 2 if #anchors = 0  and no noise
%                   ---> minimizing a regularization term 
%           = 3 if #anchors = 0 and noise
%                   ---> minimizing the 1-norm error + a regularization term 
%       pars.noisyFac 
%           = [] if noisyFac is not specified or unknown 
%           = noisyFac if noisFac is known. This parameter will be use to
%             bound the error epsilon_{ij}^+ and epsilon_{ij}^-. 
%       pars.edgeSelectionSW
%           = 0 --- using all edges to construct an SDP relaxation problem
%           = 1 --- selection of edges suitable for the SFSDP
%           = 2 --- uniform selection of edges
%       pars.regTermFactor 
%           = factor attached to refularization term;
%             if (pars.objSW == 2) || (pars.objSW == 3)
%                 pars.regTermFactor = 1;
%             else
%                 pars.regTermFactor = 0;
%             end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% xMatrix   :   sDim x n matrix of sensors' and anchors' locations computed 
%               in the sDim dimensional space, 
%               where n is the total number of sensors and anchors, and 
%               anchors are placed in the last m_a columns 
% info      :   info from SeDuMi or SDPA output
%   SeDuMi case: 
%       cpusec,feasratio,iter,pinf,dinf,numerr,timing,cpusec: see the manual of SeDuMi.
%       eTimeBuildSDP : elapsed time for building the SDP problem
%       eTimeSolveSDP : elapsed time spent in the SDP solver
%       eTimeConvSDP : elapsed time for conversion
%       eTimeAddBounds : elapsed time for adding bounds
%       eTimeRetSolution : elapsed time for retrieving sensors' locations
%   SDPA case: 
%       phasevalue,iteration,cpusec,primalObj,dualObj,
%       primalError,dualError,digits,dualityGap,mu,dimacs,
%       solveTime,convertingTime,retrievingTime,sdpaTime : see the manual of SDPA
%       eTimeBuildSDP : elapsed time for building the SDP problem
%       eTimeSolveSDP : elapsed time spent in the SDP solver
%       eTimeConvSDP : elapsed time for conversion
%       eTimeAddBounds : elapsed time for adding bounds
%       eTimeRetSolution : elapsed time for retrieving sensors' locations
% distanceMatrix    :   the distance submatrix used to construct an SDP
%                       relaxation problem in SFSDP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% startingTime = cputime; 
startingTime = tic; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To save memory, the global variables below are used
global A b c K distanceMatrix
distanceMatrix = distanceMatrix0;
originaldistanceMatrix0=distanceMatrix0;
originalXMatrix0=xMatrix0;
clear distanceMatrix0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% pars.edgeSelectionSW = 1;
% = 0 --- using all edges to construct an SDP relaxation problem
% = 1 --- selection of edges suitable for the FSDP
% = 2 --- uniform selection of edges

% Default parameters ---> 
SDPsolverDefault = 'sdpa';
% SDPsolverDefault = 'sedumi';

sparseSWdefault = 1;
%sparseSWdefault = 0;

regTermFactorDefault = 1.0; % if the regularization term is added 
                            % this factor is used. 
% <--- Default parameters

if nargin < 6
    pars.eps = 1.0e-7;
    pars.free = 0;
    pars.fid = 0; 
    pars.sparseSW = sparseSWdefault;
    pars.SDPsolver = SDPsolverDefault; 
    if pars.sparseSW <= 1
        pars.minDegree = sDim + 2;
        pars.edgeSelectionSW = 1; 
        % = 0 --- using all edges to construct an SDP relaxation problem
        % = 1 --- selection of edges suitable for the FSDP
        % = 2 --- uniform selection of edges
    else % pars.sparseSW == 2
        pars.minDegree = sDim + 4;
        pars.edgeSelectionSW = 2;
    end
    pars.noisyFac = 0.1; 
    if noOfAnchors >= sDim+1
        pars.objSW = 1;
        pars.regTermFactor = 0.0; 
    else
        pars.objSW = 3;
        pars.regTermFactor = regTermFactorDefault;
    end
else
    if ~isfield(pars,'eps')
        pars.eps = 1.0e-7;
    end
    if ~isfield(pars,'free')
        pars.free = 0;
    end
    if ~isfield(pars,'fid')
    	pars.fid = 0;
    end
    if ~isfield(pars,'sparseSW')
        pars.sparseSW = sparseSWdefault; 
        % pars.sparseSW = 0; % ---> a modified version of FSDP by Biswas and Ye
        % pars.sparseSW = 1; % ---> SFSDP
        % pars.sparseSW = 2; 
    end
    if ~isfield(pars,'SDPsolver')
        pars.SDPsolver = SDPsolverDefault;
    end
    if ~isfield(pars,'minDegree')
        if pars.sparseSW <= 1
            pars.minDegree = sDim + 2;
        else % pars.sparseSW == 2
            pars.minDegree = sDim + 4;
        end
    end
    if ~isfield(pars,'noisyFac')
        pars.noisyFac = 0.1;
    end
    if ~isfield(pars,'objSW')
        if pars.noisyFac < 1.0e-12
            if noOfAnchors >= sDim+1
                pars.objSW = 0;
                if ~isfield(pars,'regTermFactor')
                    pars.regTermFactor = 0.0; 
                end
            else
                pars.objSW = 2;
                pars.regTermFactor = regTermFactorDefault;
            end                
        else
            if noOfAnchors >= sDim+1
                pars.objSW = 1;
                if ~isfield(pars,'regTermFactor')
                    pars.regTermFactor = 0.0; 
                end
            else
                pars.objSW = 3;
                pars.regTermFactor = regTermFactorDefault;
            end                
        end
    end
    if ~isfield(pars,'regTermFactor')
        if (pars.objSW == 0) || (pars.objSW == 1)
            pars.regTermFactor = 0.0; 
        else
            pars.regTermFactor = regTermFactorDefault;
        end 
    end
    if ~isfield(pars,'edgeSelectionSW')
        if pars.sparseSW <= 1
            pars.edgeSelectionSW = 1;
            % = 0 --- using all edges to construct an SDP relaxation problem
            % = 1 --- selection of edges suitable for the FSDP
            % = 2 --- uniform selection of edges
        else % pars.sparseSW == 2
            pars.edgeSelectionSW = 2;            
        end
    end
end

if (pars.sparseSW == 2) && (pars.minDegree < sDim+3)
    pars.minDegree = sDim+3;
    if pars.verbose
        fprintf('## reset pars.minDegree = max([pars.minDegree,sDim+3]) = %d\n',pars.minDegree); 
    end
end

% Checking the sizes of xMatrix0 --->
if noOfAnchors > 0
    if isempty(xMatrix0) 
        error('## xMatrix0 needs to be sDim x (noOfSensors + noAnchors) or sDim x noAnchors.');
    elseif size(xMatrix0,1) ~= sDim
        error('## xMatrix0 needs to be sDim x (noOfSensors + noAnchors) or sDim x noAnchors.');
    elseif (size(xMatrix0,2) ~= (noOfSensors + noOfAnchors)) && (size(xMatrix0,2) ~= noOfAnchors) 
        error('## xMatrix0 needs to be sDim x (noOfSensors + noAnchors) or sDim x noAnchors.');
    end
else % noOfAnchors == 0
    if ~isempty(xMatrix0) 
        if (size(xMatrix0,1) ~= sDim) || (size(xMatrix0,2) ~= noOfSensors) 
            error('## xMatrix0 needs to be sDim x (noOfSensors + noAnchors) or sDim x noAnchors.');
        end
    end
end
rmsdSW = 1; 
if noOfAnchors == 0
    if isempty(xMatrix0) 
        if pars.verbose
            fprintf('## an anchor free problem and sensor locations are not given\n'); 
        end
        xMatrix0 = sparse(sDim,noOfSensors); 
        rmsdSW = 0; 
    end
elseif noOfAnchors < sDim+1
    if pars.verbose
        fprintf('## case 1 <= noOfAnchors < sDim+1 can not be handled effectively\n'); 
    end
    error('   take noOfAnchors = 0 and modify distanceMatrix'); 
elseif (noOfAnchors > 0) && (size(xMatrix0,2) == noOfAnchors)
    % Only anchors' location are given ---> expand the distanceMatrix 
	% to the size sDim x (noOfSensors + noOfAnchors)
    rmsdSW = 0; 
    if pars.verbose
        fprintf('## only anchor locations are given\n'); 
    end
    xMatrix0 = [sparse(sDim,noOfSensors), xMatrix0(:,1:noOfAnchors)]; 
end 
% <--- Checking the sizes of xMatrix0
% Checking the size of distanceMatrix ---> 
if (size(distanceMatrix,2) ~= noOfSensors + noOfAnchors) || (size(distanceMatrix,1) ~= noOfSensors)
	error('## distanceMatrix needs to be noAnchors x (noOfSensors + noAnchors)');
end
if norm(tril(distanceMatrix),inf) > 1.0e-8
    if pars.verbose
        fprintf('## distanceMatrix should be upper triangular\n');
        fprintf('   distanceMatrix = triu(distanceMatrix,1)\n');
    end
    distanceMatrix = triu(distanceMatrix,1);
end
% <--- Checking the size of distanceMatrix

if pars.sparseSW == 0 
    if pars.verbose
        fprintf('\nA modified version of FSDP (Biswas and Ye)\n');
        fprintf('Sunyoung Kim, Masakazu Kojima, Hayato Waki and Makoto Yamashita\n');
        fprintf('Version 1.22, January 2010\n\n');
    end
elseif pars.sparseSW == 1
    if pars.verbose
        fprintf('\nSFSDP --- A Sparse version of FSDP (Biswas and Ye)\n');
        fprintf('Sunyoung Kim, Masakazu Kojima, Hayato Waki and Makoto Yamashita\n');
        fprintf('Version 1.22, January 2010\n\n');
    end
else
    error('## pars.sparseSW needs to be either 0 or 1');    
end

noOfAnchors0 = noOfAnchors; 

if noOfAnchors0 == 0
    % Anchor free case --->
    % Fixing 3 or 4 points both in the 2 and 3 dimensional cases, respectively, and
    % adding regularization terms to the objective function later.
    [clique,anchorLocation] = findClique(distanceMatrix,sDim+1);
    controlSW = 0; 
    if ~isempty(clique)
        if isempty(anchorLocation)
            rand('state',2414);
            i=0;
            controlSW = 0; 
            while controlSW == 0
                i = i+1;
                [temp,p1] = sort(rand(1,noOfSensors));
                distanceMatrix2 = distanceMatrix + distanceMatrix';
                distanceMatrix2 = distanceMatrix2(p1,p1);
                distanceMatrix2 = triu(distanceMatrix2,1); 
                [clique,anchorLocation] = findClique(distanceMatrix2,sDim+1);
                if (~isempty(clique)) && (~isempty(anchorLocation))
                    controlSW = 1; 
                elseif i == 10
                    controlSW = -1;
                end
            end
            clear distanceMatrix2
            clique = p1(clique);  
        else
            controlSW = 1;
        end
        if controlSW == 1
            if sDim == 2
                if pars.verbose
                    fprintf('## an anchor free problem. Fix 3 sensors location temporarily ---> 3 anchors \n');
                end
            elseif sDim == 3
                if pars.verbose
                    fprintf('## an anchor free problem. Fix 4 sensors location temporarily ---> 4 anchors \n');
                end
            end
            distanceMatrix = distanceMatrix + distanceMatrix';
            sensorsIdx = setdiff(1:noOfSensors,clique);
            perm1 = [sensorsIdx,clique];
            [temp,perm0] = sort(perm1);
            distanceMatrix = distanceMatrix(perm1,perm1);
            debugSW = 0;
            if debugSW == 1
                full(xMatrix0)
                full(distanceMatrix)
                xMatrix2 = xMatrix0(:,perm1);
                dMat = sparse(noOfSensors,noOfSensors);
                for i=1:noOfSensors
                    for j=i+1:noOfSensors
                        if distanceMatrix(i,j) > 0
                            dMat(i,j) = norm(xMatrix2(:,i)' -xMatrix2(:,j)');
                        end
                    end
                end
                norm(full(dMat+dMat'-distanceMatrix))
                full(anchorLocation + 0.5)
             end
            distanceMatrix = triu(distanceMatrix,1);
            xMatrix0(:,1:noOfSensors-sDim-1) = sparse(sDim,noOfSensors-sDim-1);
            xMatrix0(:,noOfSensors-sDim:noOfSensors) = anchorLocation + 0.5;
            noOfSensors = noOfSensors-sDim-1;
            noOfAnchors = sDim+1;
            distanceMatDeleted = distanceMatrix(noOfSensors+1:noOfSensors+noOfAnchors,:); 
            distanceMatrix = distanceMatrix(1:noOfSensors,:);
         end
    else
        controlSW = -1;
    end
    if controlSW == -1
        if pars.verbose
            fprintf('## an anchor free problem. Fix 2 sensors location temporarily. \n');
        end
        distanceMatrix = distanceMatrix + distanceMatrix';
        % Fixing two points as anchors --->
        IMat = spones(distanceMatrix + distanceMatrix') + (noOfSensors+1)*speye(noOfSensors,noOfSensors);
        perm = symrcm(IMat);
        ix = perm(fix(noOfSensors/2));
        [dist,iy] = max(distanceMatrix(ix,:));
        % Moving the two sensors to the (noOfSensors-1)th and (noOfSensors)th ancors, and fix their
        % coodinates.
        % the permutation being used to retrieve the original arrangement --->
        perm0 = (1:noOfSensors);
        perm0(ix) = noOfSensors-1;
        perm0(iy) = noOfSensors;
        perm0(noOfSensors-1) = ix;
        perm0(noOfSensors) = iy;
        % <--- the permutation being used to retrieve the original arrangement
        xMatrix0 = xMatrix0(:,perm0);
        distanceMatrix = distanceMatrix(perm0,perm0);
        distanceMatrix = triu(distanceMatrix);
        xMatrix0(:,noOfSensors-1) = 0.5;
        xMatrix0(:,noOfSensors) = 0.5;
        xMatrix0(1,noOfSensors) = xMatrix0(1,noOfSensors) + dist;
        xMatrix0(:,1:noOfSensors-2) = zeros(sDim,noOfSensors-2);
        noOfSensors = noOfSensors-2;
        noOfAnchors = 2;
        distanceMatDeleted = distanceMatrix(noOfSensors+1:noOfSensors+noOfAnchors,:);
        distanceMatrix = distanceMatrix(1:noOfSensors,:);
        % <--- Fixing two points as anchors
    end
end

% Selecting edges from a given sensor network localization problem 
% --->
%
% !!! distanceMatrix is updated !!!
%

if (pars.noisyFac == 0) && (pars.sparseSW <= 1)
    ubdForSenToAnchorEdge = sDim+1;
else
    ubdForSenToAnchorEdge = (sDim+1)*2;
    % ubdForSenToAnchorEdge = ;
end
if pars.edgeSelectionSW == 1 % for FSDP and SFSDP
    if pars.edgeAlgo=="distance"
        [distanceMatrix,totalMeasurements, before_degList, degreeList] = selectEdgesForSFSDP_distanceLimitMaxDist(sDim,noOfSensors,distanceMatrix,xMatrix0,pars.minDegree,ubdForSenToAnchorEdge,pars.edges_rand_state,pars.show_plots);
    elseif pars.edgeAlgo=="oneStage"
        [distanceMatrix,totalMeasurements, before_degList, degreeList] = selectEdgesForSFSDP_onePhase(sDim,noOfSensors,distanceMatrix,xMatrix0,pars.minDegree,ubdForSenToAnchorEdge,pars.edges_rand_state,pars.show_plots);
    elseif pars.edgeAlgo=="twoStage"
        [distanceMatrix,totalMeasurements, before_degList, degreeList] = selectEdgesForSFSDP_twoPhase(sDim,noOfSensors,distanceMatrix,xMatrix0,pars.minDegree,ubdForSenToAnchorEdge,pars.edges_rand_state,pars.show_plots);
    elseif pars.edgeAlgo=="arbitrary"
        [distanceMatrix,totalMeasurements, before_degList, degreeList] = selectEdgesForSFSDP_equalInterval(sDim,noOfSensors,distanceMatrix,xMatrix0,pars.minDegree,ubdForSenToAnchorEdge,pars.edges_rand_state,pars.show_plots);
    elseif pars.edgeAlgo=="original"
        [distanceMatrix] = selectEdgesForSFSDP_orig(sDim,noOfSensors,distanceMatrix,xMatrix0,pars.minDegree,ubdForSenToAnchorEdge,pars.edges_rand_state,pars.show_plots);
        totalMeasurements=-1;
        degreeList=-1;
        before_degList=-1;
    elseif pars.edgeAlgo=="all"
        totalMeasurements=-1;
        degreeList=-1;

        [~,colSize] = size(distanceMatrix);
        mAj_mat=distanceMatrix;
        mAj_mat(colSize,colSize)=0;
        mAj_mat=mAj_mat+mAj_mat';
        g=graph(mAj_mat);
        before_degList=degree(g);
    else
        error("edgeAlgo not set in pars")
    end
elseif pars.edgeSelectionSW == 2 % for a trial modified version of ESDP
    [distanceMatrix] = selectEdgesUniformly(sDim,noOfSensors,distanceMatrix,pars.minDegree,ubdForSenToAnchorEdge);
end

% <---
% Selecting edges from a given sensor network localization problem 

noOfEdges = nnz(distanceMatrix); 
noOfEdges2 = nnz(distanceMatrix(:,1:noOfSensors)); 
degreeVector = sum(spones([distanceMatrix(:,1:noOfSensors)+distanceMatrix(:,1:noOfSensors)',...
    distanceMatrix(:,noOfSensors+1:noOfSensors+noOfAnchors)]),2); 
minDeg = full(min(degreeVector')); 
maxDeg = full(max(degreeVector')); 
averageDeg = full(sum(degreeVector')/noOfSensors); 

if pars.verbose
    fprintf('## sDim = %d, noOfSensors = %d, noOfAnchors = %d\n',...
        sDim,noOfSensors,noOfAnchors);
end
if strcmp(pars.SDPsolver,'sedumi')
    if pars.verbose
        fprintf('## pars: SDPsolver = %s, eps = %6.2e, free = %d\n',... 
            pars.SDPsolver,pars.eps,pars.free);
    end
else % sdpa
    if pars.verbose
        fprintf('## pars: SDPsolver = %s, eps = %6.2e\n',... 
            pars.SDPsolver,pars.eps);
    end
end
if pars.verbose
    fprintf('## pars: sparseSW = %d, minDegree = %d, edgeSelectionSW = %d\n', ...
        pars.sparseSW,pars.minDegree,pars.edgeSelectionSW); 
    fprintf('## pars: objSW = %d, noisyFac = %5.1e, regTermFactor = %4.2f\n',... 
        pars.objSW,pars.noisyFac,pars.regTermFactor);
end

if pars.sparseSW == 0
    if pars.verbose
        fprintf('## the number of dist. eq. used in FSDP between two sensors  = %d\n',noOfEdges2);
        fprintf('## the number of dist. eq. used in FSDP between a sensor & an anchor = %d\n',noOfEdges-noOfEdges2);
        fprintf('## the min., max. and ave. degrees over sensor nodes = %d, %d, %6.2f\n',minDeg,maxDeg,averageDeg);
    end
elseif pars.sparseSW == 1
    if pars.verbose
        fprintf('## the number of dist. eq. used in SFSDP between two sensors  = %d\n',noOfEdges2);
        fprintf('## the number of dist. eq. used in SFSDP between a sensor & an anchor = %d\n',noOfEdges-noOfEdges2);
        fprintf('## the min., max. and ave. degrees over sensor nodes = %d, %d, %6.2f\n',minDeg,maxDeg,averageDeg);
    end
elseif pars.sparseSW == 2
    if pars.verbose
        fprintf('## the number of dist. eq. used in modified ESDP between two sensors  = %d\n',noOfEdges2);
        fprintf('## the number of dist. eq. used in modified ESDP between a sensor & an anchor = %d\n',noOfEdges-noOfEdges2);
        fprintf('## the min., max. and ave. degrees over sensor nodes = %d, %d, %6.2f\n',minDeg,maxDeg,averageDeg);
    end
end
clear degreeVector
clear noOfEdges noOfEdges2 minDeg maxDeg averageDeg
% <--- 
% Selecting edges from a given sensor network localization problem 
% for FSDP, SFSDP and modified ESDP

[A,b,c,K] = generatePrimalFSDP(xMatrix0,noOfAnchors,distanceMatrix,pars);
anchorMatrix = xMatrix0(:,noOfSensors+[1:noOfAnchors]);
clear xMatrix0

% the dimensions and sizes of vectors and matrices used in the FSDP
% relaxations ---> 
mDim1 = nnz(distanceMatrix); % the number of distant equations
% the total number of equalities =
% the row size of the constraint matrix A = the number of equality constraitns
if (pars.objSW == 0) || (pars.objSW == 2)
    % the number of LP variables;
    nDimLP = 0;
else
    % the number of LP variables;
    % each nonzero distance yields 2 LP variables
    nDimLP = 2*mDim1;
end

nDimSDP = (noOfSensors+sDim)*(noOfSensors+sDim);
% the size of the vectorized SDP block in the sedumi format
nDim = nDimLP + nDimSDP;
% <--- the dimensions and sizez of vectors and matrices used in the FSDP
% relaxations 
        
% Adding regularization terms to the objective function for anchor free
% cases --->
if (pars.regTermFactor > 1.0e-10) && (pars.sparseSW <= 1) % (pars.objSW == 2) || (pars.objSW == 3)
    if (pars.sparseSW == 1)
        perturbeEpsion2 = 1;
        pertbationMat = sparse(noOfSensors+sDim,noOfSensors+sDim);
        rand('state',3201);
        pertbationMat0 = sparse(noOfSensors+sDim,noOfSensors+sDim);
        [sparsityPatternMat] = genSparsityPatternMat(A,c,K);
        sparsityPatternMat = sparsityPatternMat(1:noOfSensors,1:noOfSensors) + (1+noOfSensors)*10*speye(noOfSensors,noOfSensors);
        permutation = symamd(sparsityPatternMat);
        [RMat,p] = chol(sparsityPatternMat(permutation,permutation));
        for i=1:noOfSensors
            nzIndices = find(RMat(i,1:noOfSensors));
            for j=nzIndices
                if j > i
                    pertbationMat0(i,i) = pertbationMat0(i,i) -perturbeEpsion2;
                    pertbationMat0(j,j) = pertbationMat0(j,j) -perturbeEpsion2;
                    pertbationMat0(i,j) = pertbationMat0(i,j) +perturbeEpsion2;
                    pertbationMat0(j,i) = pertbationMat0(j,i) +perturbeEpsion2;
                end
            end
        end
        pertbationMat(permutation,permutation) = pertbationMat0(1:noOfSensors,1:noOfSensors);
        clear pertbationMat0 permutation sparsityPatternMat
	else
        perturbeEpsion2 = 2.0;
        pertbationMat = sparse(noOfSensors+sDim,noOfSensors+sDim);
        pertbationMat0 = perturbeEpsion2*(ones(noOfSensors,noOfSensors) - noOfSensors*speye(noOfSensors,noOfSensors)); 
        pertbationMat(1:noOfSensors,1:noOfSensors) = pertbationMat0; 
        clear pertbationMat0
    end
    if (pars.objSW == 0) || (pars.objSW == 2)
        c = reshape(pertbationMat,nDim,1);
    else % (pars.objSW == 1) ||(pars.objSW == 3)
        c = [c(1:nDimLP,:); pars.regTermFactor*reshape(pertbationMat,nDim-nDimLP,1)];  
    end
    clear pertbationMat
end
% <--- Adding regularization terms to the objective function for anchor free
% cases

% Adding a simplex constraint for the psd variable matrix ---> 
% pars.xAbsBound = 1.0;
% if isfield(pars,'xAbsBound')
%     simplexBound = pars.xAbsBound*(K.s+1);
%     simplexCoefMat = reshape(speye(K.s,K.s),1,K.s*K.s);
%     c = [c(1:K.l,:);sparse(1,1);c(K.l+1:size(A,2),:)];
%     A = [A(:,1:K.l),sparse(size(A,1),1),A(:,K.l+1:size(A,2))];
%     A = [A; [sparse(1,K.l),1,simplexCoefMat]];
%     b = [b; simplexBound];
%     K.l = K.l+1;
% end
% <--- Adding a simplex constraint for the psd variable matrix

% Strengthen the sparsity pattern ---> 
% sPatternMat = sparse(K.s,K.s);
% sPatternMat(:,K.s-sDim+1:K.s) = 1;%ones(K.s,sDim);
% sPatternMat(K.s-sDim+1:K.s,:) = 1;%ones(sDim,K.s);
% if isfield(K,'l') && ~isempty(K.l) 
%     sPatternVect = [sparse(1,K.l), reshape(sPatternMat,1,K.s*K.s)];
% else
%     sPatternVect = reshape(sPatternMat,1,K.s*K.s);
% end
% clear sPatternMat
% end
% <--- Strengthen the sparsity pattern

eTimeBuildSDP = toc(startingTime); 
if pars.verbose
    fprintf('## elapsed time for generating an SDP relaxation problem = %8.2f\n',eTimeBuildSDP);
end

% pars.sparseSW = 1;
if  pars.sparseSW >= 1
    pars.sDim = sDim;
    startingTime = tic; 
    [SDP,clique,convMat] = convDual(sDim,pars);
    % <----------- MS: Plotting Cliques
    if pars.verbose
        fprintf("---------------------")
    end
    aj_mat=distanceMatrix;
    [~,colSize] = size(distanceMatrix);
    aj_mat(colSize,colSize)=0;
    aj_mat=aj_mat+aj_mat';
    if pars.verbose
        fprintf("---------------------")
    end
    figure()
    G=graph(aj_mat);
    h=plot(G);
    length(clique)
    % clique{1};

    cmap = hsv(clique{1}.NoC);
    for component = 1:clique{1}.NoC
        idx = sum(clique{1}.NoElem(1:component-1));
        this_clique=clique{1}.Elem(idx+(1:clique{1}.NoElem(component)));
        highlight(h,this_clique,'NodeColor', cmap(component,:));
    end
        

    highlight(h,G,'EdgeColor',	"white")
    title("Cliques")
    % -------Plotting Cliques-------------->
    eTimeConvSDP = toc(startingTime);     
    % <--- Conversion to a sparse LMI SDP
    clear A b c
    % Adding upper bounds to \epsilon_{pq}^{+} and \epsilon_{pq}^{-} --->
    startingTime = tic; 
    if (~isempty(pars.noisyFac)) && ((pars.objSW == 1) || (pars.objSW == 3))
        phi = 3; 
        ratio1 = 1-1/(1+phi*pars.noisyFac)^2;
        ratio2 = 1/(1-phi*pars.noisyFac)^2-1;
        ratio = max([ratio1,ratio2]); 
        distanceVect = reshape(distanceMatrix',1,size(distanceMatrix,1)*size(distanceMatrix,2));
        nzIdxSet = find(distanceVect);
        dimAdd = length(nzIdxSet);
        rowSize = size(SDP.A,1);
        colSize = size(SDP.A,2);
        AMatAdd = sparse(rowSize,dimAdd);
        cAdd = (distanceVect(nzIdxSet) .* distanceVect(nzIdxSet))' * ratio;
        for i=1:dimAdd
            AMatAdd((i-1)*2+1,i) = 1;
            AMatAdd(i*2,i) = 1;
        end
        SDP.A = [SDP.A(:,1:SDP.K.f+SDP.K.l),AMatAdd,SDP.A(:,SDP.K.f+SDP.K.l+1:colSize)];
        SDP.c = [SDP.c(1:SDP.K.f+SDP.K.l,:); cAdd; SDP.c(SDP.K.f+SDP.K.l+1:colSize,:)];
        SDP.K.l = SDP.K.l + dimAdd;
        clear distanceVect AMatAdd cAdd
    end
    % <--- Adding upper bounds to \epsilon_{pq}^{+} and \epsilon_{pq}^{-}
    eTimeAddBounds = toc(startingTime);     

    startingTime = tic; 
    if ~isfield(pars,'SDPsolver') || isempty(pars.SDPsolver) || strcmp(pars.SDPsolver,'sedumi')
        if isnumeric(pars.fid) && (pars.fid == 0)
            if pars.verbose
                fprintf('Starting SeDuMi\n');
            end
        end
        [xbar,ybar,info] = sedumi(SDP.A,SDP.b,SDP.c,SDP.K,pars);
        if isnumeric(pars.fid) && (pars.fid == 0)
            if pars.verbose
                fprintf('Finished SeDuMi\n');
            end
        end
    elseif strcmp(pars.SDPsolver,'sdpa') % pars.SDPsolver='sdpa;
        OPTION.epsilonStar  = max([pars.eps,1.0e-7]);
        OPTION.epsilonDash  = max([pars.eps,1.0e-7]);
        if (pars.fid == 1)
            OPTION.print = 'display';
        elseif (pars.fid == 0)
            OPTION.print = 'no';
        elseif isstr(pars.fid)  
            OPTION.print = pars.fid; 
        end                
        [xbar,ybar,info] = sedumiwrap(SDP.A,SDP.b,SDP.c,SDP.K,[],OPTION);
    end
    info.eTimeSolveSDP = toc(startingTime); 
    
    % Retrieving an optimal solution of the original FSDP
    % ---> 
    startingTime = tic; 
    [x] = retrieveFromConvDual(K,ybar,convMat,clique);
    clear convMat clique ybar SDP
    % Extarcting locations of sensors --->
    sdpMatrix = reshape(x(K.l+1:K.l+K.s*K.s,1),K.s,K.s);
    xMatrix = sdpMatrix(noOfSensors+1:K.s,1:noOfSensors);
    xMatrix = full([xMatrix,anchorMatrix]); 
    eTimeRetSolution = toc(startingTime); 
    clear K sdpMatrix
    % <--- Extarcting locations of sensors
    if pars.verbose
        fprintf('## elapsed time for retrieving an optimal solution = %8.2f\n',eTimeRetSolution);
    end
    % <--- 
    % Retrieving the original ordering for the anchor free case
else % pars.sparseSW == 0
    eTimeConvSDP = 0.0; 
    startingTime = tic; 
    if (~isempty(pars.noisyFac)) && ((pars.objSW == 1) || (pars.objSW == 3))
        phi = 3; 
        ratio1 = 1-1/(1+phi*pars.noisyFac)^2;
        ratio2 = 1/(1-phi*pars.noisyFac)^2-1;
        [rowSize,colSize] = size(A); 
        A = [[sparse(rowSize,K.l), A];speye(K.l,K.l),speye(K.l,K.l),sparse(K.l,colSize-K.l)];
        c = [sparse(K.l,1);c];        
        distanceVect = reshape(distanceMatrix',1,size(distanceMatrix,1)*size(distanceMatrix,2));
        nzIdxSet = find(distanceVect);
        bAdd = zeros(K.l,1);
        idxSet = 2*[1:length(nzIdxSet)]; 
        bAdd(idxSet,:) = (distanceVect(nzIdxSet) .* distanceVect(nzIdxSet))' * ratio2;         
        idxSet = idxSet - 1;
        bAdd(idxSet,:) = (distanceVect(nzIdxSet) .* distanceVect(nzIdxSet))' * ratio1;        
        b = [b; bAdd]; 
        K.l = 2*K.l; 
        clear distanceVect bAdd
    end
    eTimeAddBounds = toc(startingTime);     
    startingTime = tic;
    if ~isfield(pars,'SDPsolver') || isempty(pars.SDPsolver) || strcmp(pars.SDPsolver,'sedumi')
        if isnumeric(pars.fid) && (pars.fid == 0)
            if pars.verbose
                fprintf('Starting SeDuMi\n');
            end
        end
        [x,y,info] = sedumi(A,b,c,K,pars);
 %       info.eTimeSolveSDP = info.cpusec; 
        if isnumeric(pars.fid) && (pars.fid == 0)
            if pars.verbose
                fprintf('Finished SeDuMi\n');
            end
        end
    elseif strcmp(pars.SDPsolver,'sdpa') % pars.SDPsolver='sdpa;
        OPTION.epsilonStar  = min([pars.eps,1.0e-7]);
        OPTION.epsilonDash  = min([pars.eps,1.0e-7]);
        if (pars.fid == 1)
            OPTION.print = 'display';
        elseif (pars.fid == 0)
            OPTION.print = '';
        elseif isstr(pars.fid)  
            OPTION.print = pars.fid; 
        end
        [x,y,info] = sedumiwrap(A,b,c,K,[],OPTION);
    end
    info.eTimeSolveSDP = toc(startingTime); 

    startingTime = tic;    
    nDim = length(x); 
    x = x(K.l+1:nDim);
    xMatrix = reshape(x,K.s,K.s);
    xMatrix = full([xMatrix(K.s-sDim+1:K.s,1:K.s-sDim),anchorMatrix]);
    eTimeRetSolution = toc(startingTime); 
    clear A b c K
    if pars.verbose
        fprintf('## elapsed time for retrieving an optimal solution = %8.2f\n',eTimeRetSolution); 
    end
end
info.eTimeBuildSDP = eTimeBuildSDP; 
info.eTimeConvSDP = eTimeConvSDP; 
info.eTimeAddBounds = eTimeAddBounds;
info.eTimeRetSolution = eTimeRetSolution; 

if (noOfAnchors0 == 0) % || (noOfAnchors0 == 1)
    distanceMatrix = [distanceMatrix;distanceMatDeleted];
    distanceMatrix = distanceMatrix + distanceMatrix';
    distanceMatrix = triu(distanceMatrix(perm0,perm0),1);
    xMatrix = xMatrix(:,perm0); 
end



% ----------- Added later by MS ------------
NoOfEdgesForSDP = nnz(distanceMatrix);
% Refining locations of sensors by a gradient method  --->
if (pars.localMethodSW == 1 && pars.show_plots~=1)
    % A gradient method developed by Kim Toh 
    startingTime = tic; 
    NoOfEdgesForGradMethod = nnz(distanceMatrix);
    if (NoOfEdgesForGradMethod >= 3*NoOfEdgesForSDP) && (NoOfEdgesForGradMethod >= 1.0e5)
        [distanceMatrix] = selectEdgesUniformly(sDim,noOfSensors,distanceMatrix,3*pars.minDegree,3*(sDim+1));
        NoOfEdgesForGradMethod = nnz(distanceMatrix);
    end
    DD = [distanceMatrix; originaldistanceMatrix0(:,noOfSensors+1:noOfSensors+noOfAnchors)',sparse(noOfAnchors,noOfAnchors)];    
    [xMatrix,Info] = ... 
        refinepositions(xMatrix(:,1:noOfSensors),originalXMatrix0(:,noOfSensors+1:noOfSensors+noOfAnchors),DD,3000,1.0e-8); 
    eTimeGradMethod = toc(startingTime); 
    if pars.verbose
        fprintf('## elapsed time for a gradient method = %8.2f\n',eTimeGradMethod);
    end
    xMatrix = [xMatrix,originalXMatrix0(:,noOfSensors+1:noOfSensors+noOfAnchors)];
    % Computing error in distance equations ---> 
    [meanError,maxError] = checkDistance(xMatrix,distanceMatrix); 
    % <--- Computing error in distance equations
    if pars.verbose
        fprintf('## mean error in dist. eq. = %6.2e, max. error in dist. eq. = %6.2e\n',meanError,maxError);    
    end
    % eTimeTotal = eTimeTotal + toc(startingTime);    
    % Adujust the coordinates for anchor free cases ---> 
    if (pars.objSW == 2) || (pars.objSW == 3) && (rmsdSW == 1) 
        % cpuTimeStart = cputime; 
        startingTime = tic;
        xMatrix = full(xMatrix);
        [dist,zMatrix] = procrustes(originalXMatrix0', xMatrix');
        xMatrix = zMatrix';
        if pars.verbose
    %        fprintf('## cpu time for adjusting cordinates = %8.2f\n',cputime-cpuTimeStart);
            % eTimeTotal = eTimeTotal + toc(startingTime);            
            fprintf('## elapsed time for adjusting cordinates = %8.2f\n',toc(startingTime));
        end
    % <--- Adujust the coordinates for anchor free cases 
    end
	startingTime = tic;
    if (rmsdSW == 1)
        residualVector= reshape(xMatrix(:,1:noOfSensors) - originalXMatrix0(:,1:noOfSensors),1,sDim*noOfSensors);
        rmsdGR = norm(residualVector)/sqrt(noOfSensors);
        % <--- Computing rmsd
        if pars.verbose
            fprintf('## rmsd = %6.2e\n',rmsdGR);
        end
    else
        rmsdGR = [];
    end
    % <--- Computing rmsd
    % eTimeTotal = eTimeTotal + toc(startingTime);
    % % Drawing a picture of computed and true locations of sensors ---> 
    % if (sDim == 2) || (sDim == 3)
    %     figNo=problemId*10+3; % 0+pictureNo;
    %     startingTime = tic; 
    %     drawPicture(figNo,sDim,noisyFac,xMatrix0',xMatrix',rmsdGR,noOfSensors);
    %     eTimeDrawPict = eTimeDrawPict+toc(startingTime); 
    %     fprintf('## see Figure %d\n',figNo);
    %     eTimeTotal = eTimeTotal + toc(startingTime);
    % end
    % % <--- Drawing a picture of computed and true locations of sensors    
else
    % rmsdGR = [];
    % eTimeGradMethod = [];
end
% <--- Refining locations of sensors by a gradient method

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [meanError,maxError] = checkDistance(xMatrix,distanceMatrix)
% noOfSEnsors = size(xMatrix,1);
noOfSEnsors = size(distanceMatrix,1); 
totalError = 0.0; 
maxError = 0.0;
for p=1:noOfSEnsors
    nzIdx = find(distanceMatrix(p,:)); 
    for q = nzIdx
        error = abs(norm(xMatrix(:,p)-xMatrix(:,q)) - distanceMatrix(p,q)); 
        totalError = totalError + error;
        maxError = max(maxError,error);
    end
end
meanError = totalError/nnz(distanceMatrix);
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Thanks to Kim Chuan Toh
% 
%%*************************************************************************
%% refinepositions: steepest descent with back-tracking line search to 
%%                  minimize 
%%
%% f(X) = sum_{j<=k: djk is given}  (norm(xj-xk)-djk)^2 
%%        + sum_{j,k: djk is given} (norm(xj-ak)-djk)^2
%%
%% input: X0     = [sensor position in column format]
%%        anchor = [anchor position in column format] 
%%        DD     = [sensor-sensor distance, senor-anchor distance
%%                  anchor-sensor distance,  0]
%% (optional) maxit = maximum number of gradient iterations allowed 
%% (optional) tol   = stopping tolerance  
%%
%% output: Xiter = [refined sensor position]
%%         Info.objective  = history of objective values
%%         Info.gradnorm   = history of norm of the gradients
% %%         Info.steplength = history of step-lengths
%%         Info.cputime    = cputime taken
%% 
%% child functions: gradfun2, objfun2
%%*************************************************************************

  function [Xiter,Info] = refinepositions(X0,anchor,DD,maxit,tol)

  if (nargin < 4); maxit = 3000; end
  if (nargin < 5); tol = 1e-9; end

  ttime = cputime; 
 
  [dummy,n] = size(X0);
  [dummy,m] = size(anchor); 
  if (size(DD,2) ~= n+m) 
     error('gradescent: dimension of X0 or DD not correct')
  end
  D1 = DD(1:n,1:n); 
  [II,JJ,dd] = find(triu(D1)); 
  if (size(dd,1) < size(dd,2)); dd = dd'; end
  ne = length(II); 
  S  = sparse(II,1:ne,1,n+m,ne)-sparse(JJ,1:ne,1,n+m,ne);   
  if (m > 0)
    %
    % 2008-06-13 Waki
    % change [1:m] into (1:m)
    %
     D2 = DD(1:n,n+(1:m)); 
     [I2,J2,d2] = find(D2); 
     if (size(d2,1) < size(d2,2)); d2 = d2'; end
     ne2 = length(I2);
     S2 = sparse(I2,1:ne2,1,n+m,ne2)-sparse(n+J2,1:ne2,1,n+m,ne2);
     S  = [S,S2]; 
     dd = [dd; d2]; 
  end
  Sg = S'; Sg = [Sg(1:length(dd),1:n), sparse(length(dd),m)]; 
  X0 = [X0, anchor]; 
  obj = objfun2(X0,S,dd); 
  gradx = gradfun2(X0,S,dd,Sg);
%%
  Info.objective  = zeros(1,maxit); 
  Info.gradnorm   = zeros(1,maxit); 
  Info.steplength = zeros(1,maxit); 
  Info.objective(1)  = obj; 
  Info.gradnorm(1)   = sqrt(max(sum(gradx.*gradx)));
%%
  Xiter = X0; objold = obj; 
  for iter = 1:maxit
     objnew = inf;
     gradx = gradfun2(Xiter,S,dd,Sg);
     alpha = 0.2; count = 0;
    %
    % 2008-06-13 Waki
    % change & into &&
    %
     while (objnew > objold) && (count < 20)
        alpha = 0.5*alpha; count = count + 1; 
        Xnew  = Xiter - alpha*gradx;
        objnew = objfun2(Xnew,S,dd);
     end 
     Xiter = Xnew;
     Info.objective(iter+1)  = objnew;
     Info.gradnorm(iter+1)   = sqrt(max(sum(gradx.*gradx)));
     Info.steplength(iter+1) = alpha; 
     if (abs(objnew-objold)/(1+abs(objold)) < tol); break; end 
     objold = objnew; 
  end
  if (m > 0); Xiter = Xiter(:,1:n); end
  ttime = cputime-ttime; 
  Info.cputime = ttime; 

%%*************************************************************************
%% Find the function value
%% f(X) = sum_{j<=k} (norm(xj-xk)-djk)^2 + sum_{j,k} (norm(xj-ak)-djk)^2
%%
%% input: X = [sensor position, anchor]
%%*************************************************************************
    %
    % 2008-06-13 Waki
    % Remove colon
    %
  function  objval = objfun2(X,S,dd)

  Xij = X*S; 
  normXij = sqrt(sum(Xij.*Xij))';  
  objval = norm(normXij-dd)^2;

%%*************************************************************************
%% Find the gradient of the function 
%% f(X) = sum_{j<=k} (norm(xj-xk)-djk)^2 + sum_{j,k} (norm(xj-ak)-djk)^2
%%
%% input: X = [sensor position, anchor]
%%*************************************************************************
    %
    % 2008-06-13 Waki
    % Remove colon
    %
  function  G = gradfun2(X,S,dd,Sg)

  ne = length(dd);
  Xij = X*S;
  normXij = sqrt(sum(Xij.*Xij))'+eps;
  tmp = 1-dd./normXij;
  G = Xij*spdiags(2*tmp,0,ne,ne);
  G = G*Sg;
%%*************************************************************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    
    
