%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xPrimal] = retrieveFromConvDual(K0,ybar,convMat,clique)

if isfield(K0,'f') && ~isempty(K0.f) && K0.f > 0
    fDim = K0.f;
else
    fDim = 0;
end
if isfield(K0,'l') && ~isempty(K0.l) && K0.l > 0
    ellDim = K0.l;
else
    ellDim = 0;
end

if fDim > 0
    xPrimalFree = ybar(1:fDim,1);
else
    xPrimalFree =[];
end

if ellDim > 0
    xPrimalLP = ybar(fDim+1:fDim+ellDim,1);
else
    xPrimalLP =[];
end

noOfSDPcones = length(K0.s);
ybarPointer = fDim+ellDim;
xPrimalSDP = [];
for kk=1:noOfSDPcones
    sDim = K0.s(kk);
    xAddMat = sparse(sDim,sDim);
    kDim = size(convMat{kk}{1},1);
    idx = 0;
    for p=1:clique{kk}.NoC
        pVect = convMat{kk}{p}' * ybar(ybarPointer+(1:kDim));
        sDimE = clique{kk}.NoElem(p);
        pMat = reshape(pVect,sDimE,sDimE);
        tmpIdx = clique{kk}.Elem(idx+(1:sDimE));
        xAddMat(tmpIdx,tmpIdx) = pMat;
        idx = idx +sDimE;
    end
    ybarPointer = ybarPointer+kDim;
    xPrimalSDP = [xPrimalSDP; reshape(xAddMat,sDim*sDim,1)];
end

xPrimal = [xPrimalFree; xPrimalLP; xPrimalSDP];
return
