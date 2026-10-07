%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [clique,anchorLocation] = findClique(distanceMatrix,cliqueSize);

clique = [];
anchorLocation = [];
[noOfSensors,colSize] = size(distanceMatrix); 
if cliqueSize < 3
    fprintf('# cliqueSize = %d < 3 = %d\n',cliqueSize);
    return
elseif 4 < cliqueSize
    fprintf('# cliqueSize = %d > 4 = %d\n',cliqueSize);
    return
elseif noOfSensors < colSize
    fprintf('# noOfSensors = %d < colSize = %d\n',noOfSensors,colSize);
    return
elseif noOfSensors > colSize
    fprintf('# noOfSensors = %d > colSize = %d\n',noOfSensors,colSize);
    return
elseif noOfSensors < 3
    fprintf('# noOfSensors = %d < cliqueSize = %d\n',noOfSensors,cliqueSize);
    return    
end

incidenceMatrix = spones(distanceMatrix+distanceMatrix')+speye(noOfSensors,noOfSensors); 

if cliqueSize == 3 
    [clique] = find3Clique(incidenceMatrix);
%    clique
    i1 = clique(1); x1 = zeros(2,1); 
    i2 = clique(2); x2 = zeros(2,1); 
    i3 = clique(3); x3 = zeros(2,1); 
    x2(1,1) = distanceMatrix(i1,i2);
    x3(1,1) = (distanceMatrix(i1,i3)^2-distanceMatrix(i2,i3)^2+x2(1,1)^2)/(2*x2(1,1)); 
    a = distanceMatrix(i1,i3)^2 - x3(1,1)^2; 
    if a < 0
%        fprintf('## Some error computing 3 points\n');
        anchorLocation = [];
        return
    end
    x3(2,1) = sqrt(a);
    anchorLocation = [x1,x2,x3];
elseif cliqueSize == 4 
    [clique] = find4Clique(incidenceMatrix); 
    i1 = clique(1); x1 = zeros(3,1);
    i2 = clique(2); x2 = zeros(3,1);
    i3 = clique(3); x3 = zeros(3,1);
    i4 = clique(4); x4 = zeros(3,1);
    x2(1,1) = distanceMatrix(i1,i2);
    x3(1,1) = (distanceMatrix(i1,i3)^2-distanceMatrix(i2,i3)^2+x2(1,1)^2)/(2*x2(1,1)); 
    a = distanceMatrix(i1,i3)^2 - x3(1,1)^2; 
    if a < 0
        anchorLocation = [];
        return
    else
        x3(2,1) = sqrt(a);        
    end
    bVect = [distanceMatrix(i2,i4)^2-distanceMatrix(i1,i4)^2-x2'*x2;distanceMatrix(i3,i4)^2-distanceMatrix(i1,i4)^2-x3'*x3];
    AMat = -2*[x2(1,1), 0; x3(1,1), x3(2,1)];
    y = AMat \ bVect;
    x4(1,1) = y(1,1);
    x4(2,1) = y(2,1); 
    b = distanceMatrix(i1,i4)^2 -x4(1,1)^2 -x4(2,1)^2; 
    if b < 0
        x4(3,1) = 0.0;
        anchorLocation = [];
        return
    end
    x4(3,1) = sqrt(b); 
	anchorLocation = [x1,x2,x3,x4];
else
    return
end

debugSW = 0;
if debugSW == 1
    format long
    mDim = length(clique); 
    clique
    full(distanceMatrix(clique,clique))
    dMat = zeros(mDim,mDim);
    for i=1:mDim
        p=clique(i);
        for j=i+1:mDim
            q=clique(j);
            dMat(i,j) = norm(anchorLocation(:,i)'-anchorLocation(:,j)');
        end
    end
    full(dMat)
    format short
    XXXXX
end

return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cliqueOut] = checkclique(incidenceMatrix,cliqueIn) 

% incidenceMatrix --- nodes to nodes incident matrix
% 	0, 1 matrix
%   = 1 if i = j
%   = 1 if the node i is incident to tne node j
%   = 0 otherwise

cliqueOut = [];
if isempty(cliqueIn) 
    return
elseif length(unique(cliqueIn)) < length(cliqueIn) 
    return
end
kDim = length(cliqueIn); 
checkVector = reshape(incidenceMatrix(cliqueIn,cliqueIn),1,kDim*kDim); 
if isempty(find(checkVector == 0)) 
    cliqueOut = cliqueIn;
end
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [clique] = find3Clique(incidenceMatrix)

clique = [];
[noOfSensors,colSize] = size(incidenceMatrix); 
if noOfSensors < colSize
    fprintf('noOfSensors = %d < colSize = %d\n',noOfSensors,colSize);
    return
elseif noOfSensors > colSize
    fprintf('noOfSensors = %d > colSize = %d\n',noOfSensors,colSize);
    return
elseif noOfSensors < 3
    fprintf('noOfSensors = %d < 3\n',noOfSensors);
    return    
end

p=0;
controlSW = 0;
while controlSW == 0
    p = p+1;
    pthRowNzIdx = find(incidenceMatrix(p,p+1:noOfSensors));
    if ~isempty(pthRowNzIdx)
        pthRowNzIdx = pthRowNzIdx + p;
        pDim = length(pthRowNzIdx); 
        i = 0;
        while (i < pDim) && (controlSW == 0)
            i = i+1;
            q = pthRowNzIdx(i); 
            if q < noOfSensors
                qthRowNzIdx = find(incidenceMatrix(q,q+1:noOfSensors));
                if ~isempty(qthRowNzIdx)
                    qthRowNzIdx = qthRowNzIdx + q;
                    qDim = length(qthRowNzIdx); 
                    j = 0;
                    while (j < qDim) && (controlSW == 0)
                        j = j+1;
                        r = qthRowNzIdx(j);
                        [clique] = checkclique(incidenceMatrix,[p,q,r]);
                        if ~isempty(clique)
                            controlSW = 1;
                        end
                    end
                end
            end
        end
    end
    if (p == noOfSensors-2) && (controlSW == 0)
        controlSW = -1;
    end
end

% clique

return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [clique] = find4Clique(incidenceMatrix)

clique = [];
[noOfSensors,colSize] = size(incidenceMatrix); 
if noOfSensors < colSize
    fprintf('noOfSensors = %d < colSize = %d\n',noOfSensors,colSize);
    return
elseif noOfSensors > colSize
    fprintf('noOfSensors = %d > colSize = %d\n',noOfSensors,colSize);
    return
elseif noOfSensors < 4
    fprintf('noOfSensors = %d < 4\n',noOfSensors);
    return    
end

p=0;
controlSW = 0;
while controlSW == 0
    p = p+1;
    pthRowNzIdx = find(incidenceMatrix(p,p+1:noOfSensors));
    if ~isempty(pthRowNzIdx)
        pthRowNzIdx = pthRowNzIdx + p;
        pDim = length(pthRowNzIdx); 
        i = 0;
        while (i < pDim) && (controlSW == 0)
            i = i+1;
            q = pthRowNzIdx(i); 
            if q < noOfSensors-1
                qthRowNzIdx = find(incidenceMatrix(q,q+1:noOfSensors));
                if ~isempty(qthRowNzIdx)
                    qthRowNzIdx = qthRowNzIdx + q;
                    qDim = length(qthRowNzIdx); 
                    j = 0;
                    while (j < qDim) && (controlSW == 0)
                        j = j+1;
                        r = qthRowNzIdx(j);
                        if r < noOfSensors 
                            rthRowNzIdx = find(incidenceMatrix(r,r+1:noOfSensors));
                            if ~isempty(rthRowNzIdx)
                                rthRowNzIdx = rthRowNzIdx + r;
                                rDim = length(rthRowNzIdx); 
                                k = 0; 
                                while (k < rDim)  && (controlSW == 0) 
                                    k = k+1;
                                    s = rthRowNzIdx(k);
                                    [clique] = checkclique(incidenceMatrix,[p,q,r,s]);
                                    if ~isempty(clique)
                                        controlSW = 1;
                                    end
                                end
                            end
                         end
                    end
                end
            end
        end
    end
    if (p == noOfSensors-3) && (controlSW == 0)
        controlSW = -1;
    end
end

return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
