function [SDP,clique,convMat] = convDual(sDim,pars);
global A b c K distanceMatrix
fprintf("#------------convDual--------------#")
if pars.sparseSW == 1 %SFSDP
	[SDP,clique,convMat] = convDualSFSDP(sDim,pars);
else % pars.sparseSW == 2; a trial modified version of ESDP 
    noOfSensors = size(distanceMatrix,1); 
    [SDP,clique,convMat] = convDualESDP(sDim,spones(distanceMatrix(:,1:noOfSensors)),pars);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [SDP,clique,convMat] = convDualSFSDP(sDim,pars)
global A b c K distanceMatrix clique

% fprintf('## Conversion of a primal SDP into a sparse dual SDP based on the psd completion\n');
%
% Primal SDP ---> Sparse Dual SDP
% 2008/01/30
% Masakazu Kojima

% 'ConvDual3'

% K

% % Strengthen the sparsity pattern ---> 
% sPatternMat = sparse(K.s,K.s);
% sPatternMat(:,K.s-sDim+1:K.s) = 1;%ones(K.s,sDim);
% sPatternMat(K.s-sDim+1:K.s,:) = 1;%ones(sDim,K.s);
% if isfield(K,'l') && ~isempty(K.l) 
%     sPatternVect = [sparse(1,K.l), reshape(sPatternMat,1,K.s*K.s)];
% else
%     sPatternVect = reshape(sPatternMat,1,K.s*K.s);
% end
% clear sPatternMat
% % end
% % <--- Strengthen the sparsity pattern


SDP = [];
if isfield(K,'q') && ~isempty(K.q)
    fprintf('## Qcone can not be processed\n');
    return 
end
if isfield(K,'r') && ~isempty(K.r)
    fprintf('## Rcone can not be processed\n');
    return
end

if ~isfield(K,'s') || isempty(K.s) 
    fprintf('## no conversion because K.s = []\n'); 
    return
end

% Transpose c if c is a row vector 
if size(c,1) < size(c,2)
    c = c';
end
% Transpose b if b is a row vector 
if size(b,1) < size(b,2)
    b = b';
end

% Check the dimensions of free variables 
if isfield(K,'f') && ~isempty(K.f) && K.f > 0 
    fDim = K.f;
    SDP.K.f = K.f;
else
    fDim = 0; 
    SDP.K.f = 0;
end
% Check the dimensions of LP variables
if isfield(K,'l') && ~isempty(K.l) && K.l > 0 
    ellDim = K.l;
    SDP.K.l = K.l; 
else
    ellDim = 0;
end

%[mDimA,nDimA] = size(A);
mDimA = size(A,1);

% Initialization --->
%    AbarT = []; % To be updated
SDP.A = [];
SDP.b = []; % To be apdated
SDP.c = b; %To be updated, adding 0s
SDP.K.f = mDimA;
SDP.K.s = [];
mDim = 0; % To be updated
nDim = mDimA; % To be updated
colPointer = 0;
% <--- Initialization

% noOfSDPcones = length(K.s);
% clique = cell(1,noOfSDPcones);
% convMat = cell(1,noOfSDPcones);
noOfSDPcones = 1; 
clique = cell(1,1);
convMat = cell(1,1);

if fDim > 0
    colPointer = colPointer + fDim; 
end
if ellDim > 0
    colPointer = colPointer + ellDim; 
end 

Kadd.s = K.s;

[clique{1}] = cliquesFromSpMatD(pars);

colPointer = 0;
% Constraints for free variables
if fDim > 0
    SDP.b = -c(1:fDim,:);
%    AbarT = A(:,1:fDim);
    SDP.A = A(:,1:fDim)'; 
    mDim = fDim;
    colPointer = colPointer + fDim; 
end

% Constraints for LP variables
if ellDim > 0
    SDP.b = [SDP.b;-c(colPointer+(1:ellDim),:)]; 
%    AbarT = [AbarT,A(:,colPointer+1:colPointer+ellDim)];
    SDP.A = [SDP.A;A(:,colPointer+(1:ellDim))'];
    mDim = mDim+ellDim;
    SDP.c = [SDP.c;sparse(ellDim,1)]; 
%    AbarT = [AbarT;[sparse(ellDim,colPointer),-speye(ellDim,ellDim)]]; 
    SDP.A = [SDP.A,[sparse(ellDim,colPointer),-speye(ellDim)]'];
    SDP.K.l = ellDim;
    colPointer = colPointer + ellDim; 
    nDim = nDim + ellDim; 
end 

% Constraints for SDP variables --->
[AbarAdd,SDP.bAdd,KbarAdd,convMatAdd] = primalToSparseDual(colPointer);
convMat{1} = convMatAdd;
nDimAdd = size(AbarAdd,2) - mDimA;
mDimAdd = size(AbarAdd,1);
if isempty(SDP.A)
    %        AbarT = AbarTAdd;
    SDP.A = AbarAdd;
else
    SDP.A = [ [SDP.A(:,1:mDimA); AbarAdd(:,1:mDimA)], ...
        [SDP.A(:,mDimA+1:nDim); sparse(mDimAdd,nDim-mDimA)], ...
        [sparse(mDim,nDimAdd); AbarAdd(:,mDimA+(1:nDimAdd))] ];
end
SDP.b = [SDP.b; SDP.bAdd];
SDP.c = [SDP.c; sparse(nDimAdd,1)];
SDP.K.s = [SDP.K.s, KbarAdd.s];
mDim = mDim + mDimAdd;
nDim = nDim + nDimAdd;
colPointer = colPointer + K.s*K.s;
% <--- Constraints for SDP variables

debugSW = 0;
if debugSW == 1
    %    Abar = AbarT';
    for kk=1:noOfSDPcones
        fprintf('clique{%d}.Set\n',kk);
        full(clique{kk}.Set)
    end
    fprintf('SDP.K.s = \n');
    SDP.K
    SDP.K.s
    noOfSDPcones = length(SDP.K.s);
    rowSize = size(SDP.A,1);
    fprintf('Cbar = \n');
    pointer = SDP.K.f;
    for k=1:noOfSDPcones
        full(reshape(SDP.c(pointer+1:pointer+SDP.K.s(k)*SDP.K.s(k),1),SDP.K.s(k),SDP.K.s(k)))
        pointer = pointer + SDP.K.s(k)*SDP.K.s(k);
    end
    for i=1:rowSize
        fprintf('SDP.A{%d} = \n',i);
        pointer = SDP.K.f;
        for k=1:noOfSDPcones
            full(reshape(SDP.A(i,pointer+1:pointer+SDP.K.s(k)*SDP.K.s(k)),SDP.K.s(k),SDP.K.s(k)))
            pointer = pointer + SDP.K.s(k)*SDP.K.s(k);
        end
    end
    XXXXX
end
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [clique] = cliquesFromSpMatD(pars)
global distanceMatrix

%
% 2008-06-13 Waki
% Caution!
% I have changed the structures of clique.
% 
% Before:
% the structures of clique are 
%     NoC, maxC, minC, Set and idxMatrix.
%
% After:
% the structures of clique are 
%     NoC, maxC, minC, Elem, NoElem and idxMatrix.
%
% Elem is a row vector, which contains all cliques.
% NoElem is NOC-dimensional row vector, which has each size of cliques
%
% For example, we consider three cliques
% c1 ={1,2}, c2 = {1,3,4}, c3 ={5}.
%
% Then 
% clique.Elem =[1,2,1,3,4,5]
% clique.NoElem = [2,3,1]
%
% If you want to access the i-th clqiue, execute the following command:
%
% idx = sum(clique.NoElem(1:i-1));
% clique.Elem(idx+(1:clique.NoElem(i)));
%
% In this example, if you want to access the second clique, you should execute
%
% idx = clique.NoElem(1);
% clique.Elem(idx+(1:clique.NoElem(2)));
%

noOfSEnsors = size(distanceMatrix,1); 
nDim = noOfSEnsors+pars.sDim;

sparsityPatternMat = ... 
    spones(distanceMatrix(:,1:noOfSEnsors)+distanceMatrix(:,1:noOfSEnsors)');
sparsityPatternMat = [[sparsityPatternMat,ones(noOfSEnsors,pars.sDim)];...
    ones(pars.sDim,noOfSEnsors+pars.sDim)]; 
sparsityPatternMat = sparsityPatternMat + (2*nDim+1)*speye(nDim,nDim);
%%%%%
orderingSW = 0;
if orderingSW == 0%
% pars.edgeRemoveSW == 0;
    %% minimum degree ordering
%    orderingSW = 0;
    I = symamd(sparsityPatternMat);
elseif orderingSW == 1
    %% sparse reverse Cuthill-McKee ordering
%    orderingSW = 1;
    I = symrcm(sparsityPatternMat);   
elseif orderingSW == 3
% pars.edgeRemoveSW = 3;
    %% sparse reverse Cuthill-McKee ordering only for the sensor network
    %% problem
%    orderingSW = 1;
    I = symrcm(sparsityPatternMat(1:nDim-pars.sDim,1:nDim-pars.sDim));
    I = [I,(nDim-pars.sDim+1):nDim];     
%    I = symamd(sparsityPatternMat);
% if orderingSW == 0
%     
%     %% minimum degree ordering
%     I = symamd(sparsityPatternMat);
% elseif orderingSW == 1
%     %% sparse reverse Cuthill-McKee ordering
%     I = symrcm(sparsityPatternMat);
end
%% cholesky decomposition
[R,p] = chol(sparsityPatternMat(I,I));
if (p > 0)
    error('Correlative sparsity matrix is not positive definite.');
end

clear sparsityPatternMat

debug = 0;
if debug == 1
    RR = R+R';
    spy(RR);
    XXXXX
end
%%
%% Step3
%% Finding the maxmal clieques
%%
%% put 1 for nonzero element of R
    Cliques = spones(R);
    [value,orig_idx] = sort(I);
    remainIdx = 1;
    for i=2:nDim
        idx = i:nDim;
        one = find(Cliques(i,idx));
        noOfone = length(one);
        %
        % 2008-06-08 Waki
        % replace multiplication by sum.
        %
        cliqueResult = sum(Cliques(remainIdx,idx(one)),2);
        if isempty(find(full(cliqueResult) == noOfone,1))
            remainIdx = [remainIdx;i];
        end
    end
    cSet = Cliques(remainIdx,orig_idx);
    %%
    %% Clique Information
    %%
    clique.NoC  = length(remainIdx);
    [I,J] = find(cSet');
    clique.Elem = I;
    clique.NoElem = full(sum(cSet,2));
    clique.NoElem = clique.NoElem';
    clique.maxC = full(max(clique.NoElem));
    clique.minC = full(min(clique.NoElem));
    Cliques = sparse(nDim,nDim);
    idx = 0;
    for i=1:clique.NoC
        s = clique.NoElem(i);
        tmp = clique.Elem(idx+(1:s));
        idx = idx + s;
        Cliques(tmp,tmp) = 1;
    end
    Cliques = tril(Cliques);
    
    %
    % 2008-06-10 Waki
    % Replace the part of renumbering Cliques by a faster version.
    %
    [I,J,V] = find(Cliques);
    s = length(V);
    clique.idxMatrix = sparse(J,I,(1:s),nDim,nDim,s);
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Abar,bbar,Kbar,convMat] = primalToSparseDual(colPointer)
global A c K clique
    if size(c,1) > 1 && size(c,2) > 1
       error('c must be a vector, not a matrix.'); 
    end
    sDim = K.s;
    [J,I] = find(clique{1}.idxMatrix');
    posInA = (I-1)*sDim + J;
    bbar = -2*c(colPointer+posInA);
    Abar = 2*A(:,colPointer+posInA);
    eqIdx = find(J == I);
    bbar(eqIdx) = -c(colPointer+posInA(eqIdx));
    Abar(:,eqIdx) = A(:,colPointer+posInA(eqIdx));
    Abar = Abar';
    [rowSizeAbarT,sdpDim] = size(Abar'); 

    % Sparse SDP constraint matrix in dual format --->
    psdConstMat = cell(1,clique{1}.maxC);
    for i=1:clique{1}.maxC
        psdConstMat{i} = [];
    end
    convMat = cell(1,clique{1}.NoC);
    Kbar.s = clique{1}.NoElem;
    idx = 0;
    for i=1:clique{1}.NoC
        sDimE = clique{1}.NoElem(i);
        tmpIdx = clique{1}.Elem(idx+(1:sDimE));
        idx = idx + sDimE;
        tempVector = clique{1}.idxMatrix(tmpIdx,tmpIdx)';
        tempVector = tempVector(:);
        tempVector = tempVector';
        idxInAbarSDPT = tempVector(tempVector ~= 0);
        if isempty(psdConstMat{sDimE})
            [RowIdx, ColIdx] = genPsdConstMat2(sDimE);
            convMat{i} = sparse(idxInAbarSDPT(ColIdx),RowIdx,1,sdpDim,sDimE*sDimE,sDimE*sDimE);
        else
            psdMat = psdConstMat{sDimE};
            convMat{i}(idxInAbarSDPT,:) = psdMat';
        end
    end
    AbarSDPT = [convMat{1:clique{1}.NoC}];
    % <--- Sparse SDP constraint matrix in dual format

    Abar = [Abar, -AbarSDPT];
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [SDP,clique,convMat] = convDualESDP(sDim,distanceMatrix,pars)
global A b c K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a trial modified version of ESDP  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 'convDualESDP'

% fprintf('## Conversion of a primal SDP into a sparse dual SDP based on the psd completion\n');

startingTime = cputime;

SDP = [];

% Transpose c if c is a row vector 
if size(c,1) < size(c,2)
    c = c';
end
% Transpose b if b is a row vector 
if size(b,1) < size(b,2)
    b = b';
end

% Check the dimensions of free variables 
if isfield(K,'f') && ~isempty(K.f) && K.f > 0 
    fDim = K.f;
    SDP.K.f = K.f;
else
    fDim = 0; 
    SDP.K.f = 0;
end
% Check the dimensions of LP variables
if isfield(K,'l') && ~isempty(K.l) && K.l > 0 
    ellDim = K.l;
    SDP.K.l = K.l; 
else
    ellDim = 0;
end

mDimA = size(A,1);

% Initialization --->
SDP.A = [];
SDP.b = []; % To be apdated
SDP.c = b; %To be updated, adding 0s
SDP.K.f = mDimA;
SDP.K.s = [];
mDim = 0; % To be updated
nDim = mDimA; % To be updated
colPointer = 0;
% <--- Initialization

% Constraints for free variables
if fDim > 0
    SDP.b = -c(1:fDim,:);
%    AbarT = A(:,1:fDim);
    SDP.A = A(:,1:fDim)'; 
    mDim = fDim;
    colPointer = colPointer + fDim; 
end

% Constraints for LP variables
if ellDim > 0
    SDP.b = [SDP.b;-c(colPointer+(1:ellDim),:)]; 
    SDP.A = [SDP.A;A(:,colPointer+(1:ellDim))'];
    mDim = mDim+ellDim;
    SDP.c = [SDP.c;sparse(ellDim,1)]; 
    SDP.A = [SDP.A,[sparse(ellDim,colPointer),-speye(ellDim)]'];
    SDP.K.l = ellDim;
    colPointer = colPointer + ellDim; 
    nDim = nDim + ellDim; 
end 

noOfSDPcones = length(K.s);
clique = cell(1,noOfSDPcones);
convMat = cell(1,noOfSDPcones);

noOfSensors = size(distanceMatrix,1);
commonElements = [noOfSensors+1:noOfSensors+sDim]; 

for kk=1:noOfSDPcones
    %    startingTime = cputime;
    sdpDim = K.s(kk);
    Kadd.s = sdpDim;
    oneClique.NoC = nnz(distanceMatrix); 
    oneClique.NoElem = (sDim+2)*ones(1,oneClique.NoC); 
    oneClique.maxC = sDim+2;
    oneClique.minC = sDim+2;
    [rowIdx,colIdx] = find(distanceMatrix);
    oneClique.Elem = reshape([rowIdx'; colIdx'; repmat(commonElements',1,oneClique.NoC)],1,(sDim+2)*oneClique.NoC);
    distanceMatrix2 = distanceMatrix+speye(noOfSensors,noOfSensors); 
    pointer = 0;
    oneClique.idxMatrix = [];
    for i=1:noOfSensors
        idx = find(distanceMatrix2(i,:));
        idx = [idx,[noOfSensors+1:noOfSensors+sDim]]; 
        oneColumn = sparse(noOfSensors+sDim,1); 
        nonzeros = nnz(idx); 
        oneColumn(idx,1) = pointer + [1:nonzeros]';
        oneClique.idxMatrix = [oneClique.idxMatrix,oneColumn]; 
        pointer = pointer + nonzeros; 
    end
    for i=1:sDim
        oneColumn = sparse(noOfSensors+sDim,1); 
        oneColumn(noOfSensors+i:noOfSensors+sDim,1) = [pointer+1:pointer+1+(sDim-i)]'; 
        oneClique.idxMatrix = [oneClique.idxMatrix,oneColumn]; 
        pointer = pointer+(sDim-i)+1; 
    end
    
    oneClique.idxMatrix = oneClique.idxMatrix'; 
    clique{kk} = oneClique;
    [AbarAdd,SDP.bAdd,KbarAdd,convMatAdd] = ...
        primalToSparseDualESDP(A(:,colPointer+(1:sdpDim*sdpDim)),c(colPointer+(1:sdpDim*sdpDim),1),Kadd,clique{kk});
%    [AbarAdd,SDP.bAdd,KbarAdd,convMatAdd] = ...
%        primalToSparseDual(colPointer);
    convMat{kk} = convMatAdd;
    nDimAdd = size(AbarAdd,2) - mDimA;
    mDimAdd = size(AbarAdd,1);
    % update --->
    if isempty(SDP.A)
        SDP.A = AbarAdd;
    else
        SDP.A = [ [SDP.A(:,1:mDimA); AbarAdd(:,1:mDimA)], ...
            [SDP.A(:,mDimA+1:nDim); sparse(mDimAdd,nDim-mDimA)], ...
            [sparse(mDim,nDimAdd); AbarAdd(:,mDimA+(1:nDimAdd))] ];
    end
    SDP.b = [SDP.b; SDP.bAdd];
    SDP.c = [SDP.c; sparse(nDimAdd,1)];
    SDP.K.s = [SDP.K.s, KbarAdd.s];
    mDim = mDim + mDimAdd;
    nDim = nDim + nDimAdd;
    % <--- update
    colPointer = colPointer + sdpDim*sdpDim;
end
% <--- Constraints for SDP variables

debugSW = 0;
if debugSW == 1
    for kk=1:noOfSDPcones
        fprintf('clique{%d}.Set\n',kk);
        full(clique{kk}.Set)
    end
    fprintf('SDP.K.s = \n');
    SDP.K
    SDP.K.s
    noOfSDPcones = length(SDP.K.s);
    rowSize = size(SDP.A,1);
    fprintf('Cbar = \n');
    pointer = SDP.K.f;
    for k=1:noOfSDPcones
        full(reshape(SDP.c(pointer+1:pointer+SDP.K.s(k)*SDP.K.s(k),1),SDP.K.s(k),SDP.K.s(k)))
        pointer = pointer + SDP.K.s(k)*SDP.K.s(k);
    end
    for i=1:rowSize
        fprintf('SDP.A{%d} = \n',i);
        pointer = SDP.K.f;
        for k=1:noOfSDPcones
            full(reshape(SDP.A(i,pointer+1:pointer+SDP.K.s(k)*SDP.K.s(k)),SDP.K.s(k),SDP.K.s(k)))
            pointer = pointer + SDP.K.s(k)*SDP.K.s(k);
        end
    end
    XXXXX
end
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Abar,bbar,Kbar,convMat] = primalToSparseDualESDP(A,c,K,clique)
    if size(c,1) > 1 && size(c,2) > 1
       error('c must be a vector, not a matrix.'); 
    end
    sDim = K.s;
    [J,I] = find(clique.idxMatrix');
    posInA = (I-1)*sDim + J;
    bbar = -2*c(posInA);
    Abar = 2*A(:,posInA);
    eqIdx = find(J == I);
    bbar(eqIdx) = -c(posInA(eqIdx));
    Abar(:,eqIdx) = A(:,posInA(eqIdx));
    Abar = Abar';
    [rowSizeAbarT,sdpDim] = size(Abar'); 

    % Sparse SDP constraint matrix in dual format --->
    psdConstMat = cell(1,clique.maxC);
    for i=1:clique.maxC
        psdConstMat{i} = [];
    end
    convMat = cell(1,clique.NoC);
    Kbar.s = clique.NoElem;
    idx = 0;
    for i=1:clique.NoC
        sDimE = clique.NoElem(i);
        tmpIdx = clique.Elem(idx+(1:sDimE));
        idx = idx + sDimE;
        tempVector = clique.idxMatrix(tmpIdx,tmpIdx)';
        tempVector = tempVector(:);
        tempVector = tempVector';
        idxInAbarSDPT = tempVector(tempVector ~= 0);
        if isempty(psdConstMat{sDimE})
            [RowIdx, ColIdx] = genPsdConstMat2(sDimE);
            convMat{i} = sparse(idxInAbarSDPT(ColIdx),RowIdx,1,sdpDim,sDimE*sDimE,sDimE*sDimE);
        else
            psdMat = psdConstMat{sDimE};
            convMat{i}(idxInAbarSDPT,:) = psdMat';
        end
    end
    AbarSDPT = [convMat{1:clique.NoC}];
    % <--- Sparse SDP constraint matrix in dual format

    Abar = [Abar, -AbarSDPT];
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [psdMat] = genPsdConstMat(sDimE)
%     RowIdx = zeros(sDimE*sDimE,1);
%     ColIdx = zeros(sDimE*sDimE,1);
%     idx = 1;
%     k = 1;
%     for row = 1:sDimE
%         RowIdx(idx) = (row-1)*sDimE + row;
%         ColIdx(idx) = k;
%         idx = idx + 1;
%         k = k + 1;
%         for col = (row+1):sDimE
%             RowIdx(idx) = (row-1)*sDimE + col;
%             ColIdx(idx) = k;
%             idx = idx + 1;
%             RowIdx(idx) = (col-1)*sDimE + row;
%             ColIdx(idx) = k;
%             idx = idx + 1;
%             k = k + 1;
%         end
%     end
%     psdMat = sparse(RowIdx,ColIdx,1,sDimE*sDimE,sDimE*(sDimE+1)/2,sDimE*sDimE);
%     clear RowIdx ColIdx
% 
% return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 2008-06-08 Waki 
% Revise genPsdConstMat. Change outputed variable.
%
%
function [RowIdx, ColIdx] = genPsdConstMat2(sDimE)

[I,J] = find(ones(sDimE));
RowIdx = (J-1)*sDimE + I;
tmpMat = zeros(sDimE);
idx = 0;
for i=1:sDimE
    tmpMat(i,i:sDimE) = idx + (i:sDimE);
    idx = idx + sDimE -i;
end
ColIdx = triu(tmpMat)+triu(tmpMat,1)';
ColIdx = ColIdx(:);
% clear tmpMat
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%









