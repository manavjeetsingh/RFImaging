%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sparsityPatternMat] = genSparsityPatternMat(A,c,K)
sparsityPatternMat = [];
if ~isfield(K,'s') || isempty(K.s)  
    fprintf('## K.s is not assigned\n');
    return
elseif length(K.s) > 1
    fprintf('## length(K.s) = 1 is assumed in this implementation\n');
    return
end

fDim = 0;
if isfield(K,'f') && ~isempty(K.f) && K.f > 0 
    fDim = K.f;  
end

ellDim = 0;
if isfield(K,'l') && ~isempty(K.l) && K.l > 0 
    ellDim = K.l;
end

qDim = 0;
if isfield(K,'q') && ~isempty(K.q)
    qDim = sum(K.q); 
end

rDim = 0;
if isfield(K,'r') && ~isempty(K.r)
    rDim = sum(K.r); 
end

nonSDPdim = fDim+ellDim+qDim+rDim; 

sDim = 0; 
if isfield(K,'s') && ~isempty(K.s)
    sDim = sum(K.s);
end

if size(c,2) > size(c,1)
    c = c';
end
% Construction of sparsityPatternMat
sparsityPatternMat = speye(sDim);

if isfield(K,'s') && ~isempty(K.s)
    pointer = 0;
    for p=1:length(K.s)
        kDim = K.s(p); 
        tempVec = abs(c(nonSDPdim+pointer+(1:kDim*kDim),1))'; 
        tempVec = tempVec + sum(abs(A(:,nonSDPdim+pointer+(1:kDim*kDim))),1);
        tempMat = reshape(tempVec,kDim,kDim); 
        sparsityPatternMat(pointer+(1:kDim),pointer+(1:kDim)) = sparsityPatternMat(pointer+(1:kDim),pointer+(1:kDim)) + tempMat; 
        pointer = pointer + kDim; 
    end
end
sparsityPatternMat = spones(sparsityPatternMat);

return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%