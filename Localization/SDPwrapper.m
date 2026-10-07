function [cost, estimatedlocationMatrix, before_degreeList, degreeList] = SDPwrapper(pars, distanceMatrix0, ...
    trueLocationMatrix)
    % 1. Call SDP as a function to get the localization
    pars.free= 0; pars.eps= 1.0000e-05; pars.objSW = 1;
    pars.edges_rand_state=0;
    pars.plot=0;
    pars.sparseSW = 0;
    pars.analyzeData = 0;
    
    [estimatedlocationMatrix,info,pars,distanceMatrix,totalMeasurements,before_degreeList,degreeList]=SFSDP(pars.sDim,pars.noOfSensors,...
     pars.noOfAnchors,trueLocationMatrix,distanceMatrix0,pars);
    
    % 2. Calculate the cost using estimatedLocationMatrix and
    % trueLocationMatrix
    trueDistances=zeros([pars.noOfSensors,pars.noOfSensors]);
    estimatedDistances=zeros([pars.noOfSensors,pars.noOfSensors]);
    for sensorI = 1:pars.noOfSensors
        for sensorJ = 1:pars.noOfSensors
            if sensorI>=sensorJ
                continue
            else
                if distanceMatrix0(sensorI, sensorJ)~=0
                    trueLoc1=trueLocationMatrix(:,sensorI);
                    trueLoc2=trueLocationMatrix(:,sensorJ);
                    truedist=norm(trueLoc1-trueLoc2);
    
                    estLoc1=estimatedlocationMatrix(:,sensorI);
                    estLoc2=estimatedlocationMatrix(:,sensorJ);
                    estDist=norm(estLoc1-estLoc2);
    
                    trueDistances(sensorI,sensorJ)=truedist;
                    estimatedDistances(sensorI, sensorJ)=estDist;
                end
            end
        end
    end

    % cost=rmse(trueDistances,estimatedDistances,'all');
    cost=rmse(distanceMatrix0(1:pars.noOfSensors,1:pars.noOfSensors),estimatedDistances,'all');
end