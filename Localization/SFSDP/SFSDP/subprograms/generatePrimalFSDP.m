%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  [A,b,c,K] = generatePrimalFSDP(xMatrix0,noOfAnchors,distanceMatrix,pars)

% 
% This program is based on the Biswas-Ye FSDP, but modified for computational efficiency. 
%
% Input 
% xMatrix0  :   sDim x n matrix of sensors and anchors in the sDim dimensional space, 
%               where n is the number of total sensors and anchors and 
%               anchors are placed in the last m_a columns, 
%               where m_a denotes the number of anchors. 
% noOfAnchors   : the number m_a of anchors; the last m columns of xMatrix0.
% distanceMatrix : the sparse (and noisy) distance matrix between xMatrix0(:,i) and xMatrix0(:,j)

% global ORIGINAL MEMORY
% if MEMORY == 1
%     d = whos('*');
%     mem = 0;
%     for i=1:length(d)
%         mem = mem + d(i).bytes;
%     end
% end

sDim = size(xMatrix0,1); % the dimension of the space
n = size(distanceMatrix,2); % the number of all sensors and anchors

noOfSensors = n-noOfAnchors; % the number of sensors
mDim1 = nnz(distanceMatrix); % the number of distant equations
mDim2 = sDim*(sDim+1)/2; % the number of equalities to specify W = I
mDim = mDim1 + mDim2; 
    % the total number of equalities = 
    % the row size of the constraint matrix A = the number of equality constraitns
if (pars.objSW == 0) || (pars.objSW == 2)
    nDimLP = 0;
else
    nDimLP = 2*mDim1; 
end
    % the number of LP variables; 
    % each nonzero distance yields 2 LP variables
nDimSDP = (noOfSensors+sDim)*(noOfSensors+sDim);
    % the size of the vectorized SDP block in the sedumi format
nDim = nDimLP + nDimSDP;
    % the column size of the constraint matrix A = the number of real
    % variables
K.l = nDimLP;
K.s = noOfSensors+sDim;
%A = sparse(mDim,nDim);
b = sparse(mDim,1);
% cost vector
if (pars.objSW == 1) || (pars.objSW == 3)
    c = sparse((1:nDimLP),1,1,nDim,1,nDimLP);
    %c(1:nDimLP,1) = 1;
else
    c = sparse(nDim,1);
end
    if (pars.objSW == 1) || (pars.objSW == 3)
        s = nnz(distanceMatrix(1:noOfSensors,:));
        RowIdx = [(1:s),(1:s)];
        ColIdx = [((1:s)-1)*2+1,((1:s)-1)*2+2];
        Val = [ones(1,s),-ones(1,s)];
    else
        RowIdx = [];
        ColIdx = [];
        Val = [];
    end
    rowPointer = 0;
    for p = 1:noOfSensors
        nzIdx = find(distanceMatrix(p,:));
        if ~isempty(nzIdx)
            sIdx = nzIdx(nzIdx <= noOfSensors);
            aIdx = nzIdx(nzIdx >  noOfSensors);
            if ~isempty(sIdx)
                s = length(sIdx);
                Ypp = K.l + (p-1)*K.s +p;
                Ypp = repmat(Ypp,1,s);
                Yqq = K.l + (sIdx-1)*K.s +sIdx;
                Ypq = K.l + (p-1)*K.s +sIdx;
                Yqp = K.l + (sIdx-1)*K.s +p;

                idx = rowPointer + (1:s);
                RowIdx = [RowIdx,repmat(idx,1,4)];
                ColIdx = [ColIdx,Ypp,Yqq,Ypq,Yqp];
                Val = [Val,ones(1,2*s),-ones(1,2*s)];
                b(idx,1) = distanceMatrix(p,sIdx).*distanceMatrix(p,sIdx);
                rowPointer = rowPointer + s;
            end
            if ~isempty(aIdx)
                Ypp = K.l + (p-1)*K.s +p;
                idx = repmat(rowPointer+(1:length(aIdx)),2*sDim+1,1);
                idx = idx(:);
                RowIdx = [RowIdx,idx'];
                
                idx = [Ypp,K.l + (p-1)*K.s + noOfSensors + (1:sDim),K.l + (noOfSensors-1+(1:sDim))*K.s + p];
                idx = repmat(idx',1,length(aIdx));
                idx = idx(:);
                ColIdx = [ColIdx,idx'];
                
                tmp = repmat(-xMatrix0(1:sDim,aIdx),2,1);
                tmp = [ones(1,length(aIdx));tmp];
                tmp = tmp(:);
                Val = [Val,tmp'];
                tmp = zeros(1,length(aIdx));
                for i=1:length(aIdx)
                   tmp(i) =  xMatrix0(:,aIdx(i))'*xMatrix0(:,aIdx(i));
                end
                b(rowPointer+(1:length(aIdx)),1) = distanceMatrix(p,aIdx).*distanceMatrix(p,aIdx) - tmp;
                rowPointer = rowPointer + length(aIdx);
            end
        end
    end
    for i=1:sDim
        for j=i:sDim
            rowPointer = rowPointer + 1;
            if i == j;
                Iii = K.l + (noOfSensors+i-1)*K.s + noOfSensors + i;
                RowIdx = [RowIdx,rowPointer];
                ColIdx = [ColIdx,Iii];
                Val = [Val,1];
                b(rowPointer,1) = 1;
            else
                Iij = K.l + (noOfSensors+i-1)*K.s + noOfSensors + j;
                Iji = K.l + (noOfSensors+j-1)*K.s + noOfSensors + i;
                RowIdx = [RowIdx,rowPointer,rowPointer];
                ColIdx = [ColIdx,Iij,Iji];
                Val = [Val,1,1];
                b(rowPointer,1) = 0;
            end
        end
    end
%    [temp,permVect] = sort(ColIdx);
    A = sparse(RowIdx,ColIdx,Val,mDim,nDim,length(Val));
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
