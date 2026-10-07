%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [distanceMatrix] = selectEdgesUniformly(sDim,noOfSensors,distanceMatrix,minDegree,ubdForSenToAnchorEdge)

[rowSize,colSize] = size(distanceMatrix);
noOfAnchors = colSize - noOfSensors; 
senToAnchorDistMat = sparse(rowSize,noOfAnchors); 

% startingTime = tic; 

% Selecting edges between sensors and anchors 
% ---> 
countVector = sparse(1,rowSize); 
for r=1:noOfAnchors
    nzIdx = find(distanceMatrix(:,noOfSensors+r)' > 0); 
    if ~isempty(nzIdx)
        for p=nzIdx
            if (countVector(p) < ubdForSenToAnchorEdge)
                senToAnchorDistMat(p,r) = distanceMatrix(p,noOfSensors+r); 
                countVector(p) = countVector(p) + 1;
            end
        end
    end
end
% degreeVector = sum(spones(distanceMatrix(:,noOfSensors+1:noOfSensors+noOfAnchors)),2)'; 
% idx1 = find(degreeVector > ubdForSenToAnchorEdge); 
% idx2 = find(degreeVector <= ubdForSenToAnchorEdge); 
% if ~isempty(idx1)
%     for p=idx1
%         nzIdx = find(distanceMatrix(p,noOfSensors+1:noOfSensors+noOfAnchors));
%         nzIdxUsed = nzIdx(1:ubdForSenToAnchorEdge); 
%         senToAnchorDistMat(p,nzIdxUsed) = distanceMatrix(p,noOfSensors+nzIdxUsed);
%     end
%     countVector(idx2) = ubdForSenToAnchorEdge;        
% end
% if ~isempty(idx2)
%     senToAnchorDistMat(idx1,:) = distanceMatrix(idx1,noOfSensors+1:noOfSensors+noOfAnchors);
%     countVector(idx2) = degreeVector(idx2); 
% end
% < ---
% Selecting edges between sensors and anchors 

% fprintf('##0 %6.1f\n',toc(startingTime));
% startingTime = tic;

countVector = min([countVector;repmat(sDim,1,noOfSensors)],[],1);

% fprintf('##1 %6.1f\n',toc(startingTime));
% startingTime = tic;

% Selecting edges between sensors and anchors 
% ---> 
noOfEffectiveDegrees = sum(countVector); 
totalDegrees = noOfSensors*minDegree; 
rand('state',3201);
[nzzRowIdx,nzzColIdx,nzzValue] = find(distanceMatrix(:,1:noOfSensors)); 
% clear distanceMatrix
noOfNzz = length(nzzRowIdx); 
[temp,permVect] = sort(rand(noOfNzz,1)); 
nzzRowIdx = nzzRowIdx(permVect);
nzzColIdx = nzzColIdx(permVect);
nzzValue = nzzValue(permVect);
selRowIdx = [];
selColIdx = [];
selValue  = [];
k = 0; 

% fprintf('##2 %6.1f\n',toc(startingTime));
% startingTime = tic;

while (noOfEffectiveDegrees < totalDegrees) && (k < noOfNzz) 
    k = k+1;
%    k = permVect(j);
    p = nzzRowIdx(k); 
    q = nzzColIdx(k); 
    if (countVector(1,p) < minDegree) 
        selRowIdx = [selRowIdx;p];
        selColIdx = [selColIdx;q];
        selValue  = [selValue;nzzValue(k)];
        countVector(p) = countVector(p)+1;
        noOfEffectiveDegrees = noOfEffectiveDegrees+1;
        if (countVector(1,q) < minDegree)
            countVector(q) = countVector(q)+1;
            noOfEffectiveDegrees = noOfEffectiveDegrees+1;
        end
    elseif (countVector(1,q) < minDegree) 
        selRowIdx = [selRowIdx;p];
        selColIdx = [selColIdx;q];
        selValue  = [selValue;nzzValue(k)];
        countVector(q) = countVector(q)+1;
        noOfEffectiveDegrees = noOfEffectiveDegrees+1;
    end
end

% fprintf('##3 %6.1f\n',toc(startingTime));
% startingTime = tic;

[temp,permVect] = sort(selColIdx);
distanceMatrix = ...
     [sparse(selRowIdx(permVect),selColIdx(permVect),selValue(permVect),noOfSensors,noOfSensors),...
     senToAnchorDistMat]; 

% fprintf('##4 %6.1f\n',toc(startingTime));

% distanceMatrix = ...
%     [sparse(selRowIdx,selColIdx,selValue,noOfSensors,noOfSensors),...
%     senToAnchorDistMat]; 

return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
