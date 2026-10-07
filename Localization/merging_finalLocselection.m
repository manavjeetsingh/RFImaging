clear;

internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/MultipleFreqSim''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/subprograms''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/MultipleFreqSim/SmartSearchLocalization''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/sdpam7-windows''')

pyenv(Version="../../../.venv/Scripts/python.exe")
% pyenv(Version="../.venv/bin/python")
py.sys.path().append('../../../')

anchorType=2;
freqsToUse=3;

if anchorType==2
    lklhd_dat=load("Data_svr_se/dat_randomF"+int2str(freqsToUse)+"KEst.mat");
else
    lklhd_dat=load("Data_svr_se/dat_sidesF"+int2str(freqsToUse)+"KEst.mat");
end
wavelengthMultiple=1.5;


% for dat_num_anchors=lklhd_dat.all_numAnchors(4:4)
for dat_num_anchors=lklhd_dat.all_numAnchors
    
    before_sdp_loc_error=[];
    mid_sdp_loc_error=[];
    after_sdp_loc_error=[];

    before_sdp_rmsd_error=[];
    mid_sdp_rmsd_error=[];
    after_sdp_rmsd_error=[];

    all_xMatrix0={};
    all_xMatrixEst={};
    
    for exp_num=1:length(lklhd_dat.all_lkhd_X_anchors_orig{dat_num_anchors})
    % for exp_num=1:2
        disp(exp_num)
       

        X_tags_original=lklhd_dat.all_lkhd_X_tags_orig{dat_num_anchors}{exp_num}/100;
        X_tag_lkhd=lklhd_dat.all_lkhd_X_hat_tags{dat_num_anchors}{exp_num}/100;
        X_anchors=lklhd_dat.all_lkhd_X_anchors_orig{dat_num_anchors}{exp_num}/100;
        
        b_locerrs=calc_locError(X_tags_original, X_tag_lkhd);
        before_sdp_loc_error=[before_sdp_loc_error b_locerrs];

        b_rmsderrs=calc_rmsd(X_tags_original, X_tag_lkhd);
        before_sdp_rmsd_error=[before_sdp_rmsd_error b_rmsderrs];
        
        origTagIdx=lklhd_dat.all_lkhd_tag_Oidx{dat_num_anchors}{exp_num};
        origUnlocalizedTagIdx=lklhd_dat.all_lkhd_unlocalized_tag_Oidx{dat_num_anchors}{exp_num};
        origAnchorIdx=lklhd_dat.all_lkhd_anchor_tag_Oidx{dat_num_anchors}{exp_num};

        xMatrix0Lkhd=[X_tags_original X_anchors];
        xMatrixEstLkhd=[X_tag_lkhd X_anchors];
        lkhdgroundDistances0=zeros(size(lklhd_dat.all_lkhd_pars{dat_num_anchors}{exp_num}.seedK));
        lkhdgroundDistances=zeros(size(lklhd_dat.all_lkhd_pars{dat_num_anchors}{exp_num}.seedK));
        num_sensors=lklhd_dat.all_lkhd_pars{dat_num_anchors}{exp_num}.num_tags;
        for i=1:num_sensors
            for j =1:num_sensors+dat_num_anchors
                if i>=j
                    continue;
                else
                    lkhdgroundDistances0(i,j)=norm(xMatrix0Lkhd(:,i)-xMatrix0Lkhd(:,j));
                    sensorDist_=lkhdgroundDistances0(i,j);
                    if sensorDist_<=2 && sensorDist_>=0.2
                        lkhdgroundDistances(i,j)=sensorDist_;
                    else
                        lkhdgroundDistances(i,j)=0;
                    end
                end
            end
        end

        % %Visuaize
        % figure;
        % hold on
        % 
        % plot(xMatrix0Lkhd(1,1:num_sensors),xMatrix0Lkhd(2,1:num_sensors), ...
        %     'o',"DisplayName","Ground truth")
        % plot(xMatrix0Lkhd(1,num_sensors+1:num_sensors+dat_num_anchors), ...
        %     xMatrix0Lkhd(2,num_sensors+1:num_sensors+dat_num_anchors), ...
        %     'g^',"DisplayName","Anchors")
        % plot(xMatrixEstLkhd(1,1:num_sensors), ...
        %     xMatrixEstLkhd(2,1:num_sensors),'*',"DisplayName", ...
        %     "Localization")
        % 
        % plot([xMatrix0Lkhd(1,1:num_sensors); ...
        %     xMatrixEstLkhd(1,1:num_sensors)], ...
        %     [xMatrix0Lkhd(2,1:num_sensors); ...
        %     xMatrixEstLkhd(2,1:num_sensors)],'b-','HandleVisibility', ...
        %     'off')
        % title("Before"+exp_num)
        % hold off

        % SDP
        generatedProblem=load("Datasets_svr_se/S"+lklhd_dat.all_lkhd_pars{dat_num_anchors}{exp_num}.num_tags+"A"+...
        lklhd_dat.all_lkhd_pars{dat_num_anchors}{exp_num}.num_anchors+"anchType"+lklhd_dat.anchor_type+...
        "Freqs"+"-1"+"/"+exp_num+".mat");
        sdpResultDat=load("Data_svr_se/directDistLocalization_S"+lklhd_dat.all_lkhd_pars{dat_num_anchors}{exp_num}.num_tags+ ...
            "A"+lklhd_dat.all_lkhd_pars{dat_num_anchors}{exp_num}.num_anchors+ ...
            "anchType"+lklhd_dat.anchor_type+"EdgeAlgoallmaxseed200allPhasestrueFreqs"+freqsToUse+".mat");

        newXMatrix=xMatrixEstLkhd;
        newXMatrix0=xMatrix0Lkhd;
        newNumSensors=length(origUnlocalizedTagIdx);
        newNumAnchors=length(origAnchorIdx);
   
        pars={};
        pars.useK=false; % use distances directly instead of gettingK and then using it.
        pars.sDim=2; % 2D Space
        pars.noisyFac=0; % No link distance noise, phase measurement will have noise
        pars.radiorange=2; % Max radio range (grid size 4x4)
        pars.minradiorange=0.2; % Smaller links will have coupling
        pars.anchorType=lklhd_dat.anchor_type; % See generateProblem.m file for more info
        pars.noOfAnchors=lklhd_dat.all_lkhd_pars{dat_num_anchors}{exp_num}.num_anchors; 
        pars.noOfSensors=lklhd_dat.all_lkhd_pars{dat_num_anchors}{exp_num}.num_tags;
        pars.edgeAlgo="all";
        pars.maxIterations=1; % no searching for better k
        pars.add_errors=1; % phase errors
        pars.allPhases=true;
        pars.freqsToUse=-1;
        pars.verbose=false;
        seedK=zeros(pars.noOfSensors, pars.noOfSensors+pars.noOfAnchors);
        pars.localMethodSW=1;
        pars.show_plots=0;


        % [cost, sdpXMatrix ,before_degreeList,degreeList] = SDPwrapper(pars, seedK, generatedProblem.dist_ests_k, ...
        % generatedProblem.xMatrix0);
        sdpXMatrix=sdpResultDat.all_estimatedlocationMatrix{exp_num};
      
        
        m_locerrs=calc_locError(generatedProblem.xMatrix0(:,1:pars.noOfSensors), sdpXMatrix(:,1:pars.noOfSensors));
        mid_sdp_loc_error=[mid_sdp_loc_error m_locerrs];

        m_rmsderrs=calc_rmsd(generatedProblem.xMatrix0(:,1:pars.noOfSensors), sdpXMatrix(:,1:pars.noOfSensors));
        mid_sdp_rmsd_error=[mid_sdp_rmsd_error m_rmsderrs];

        % Final locmat selection
        
        for tno=1:pars.noOfSensors
            tno_converted=find(origTagIdx==tno);
            dist=norm(xMatrixEstLkhd(:,tno_converted)-sdpXMatrix(:,tno));
            if dist<wavelengthMultiple*0.16 || dist>5*0.16
                sdpXMatrix(:,tno)=xMatrixEstLkhd(:,tno_converted);
            end

        end

        a_locerrs=calc_locError(generatedProblem.xMatrix0(:,1:pars.noOfSensors), sdpXMatrix(:,1:pars.noOfSensors));
        after_sdp_loc_error=[after_sdp_loc_error a_locerrs];

        a_rmsderrs=calc_rmsd(generatedProblem.xMatrix0(:,1:pars.noOfSensors), sdpXMatrix(:,1:pars.noOfSensors));
        after_sdp_rmsd_error=[after_sdp_rmsd_error a_rmsderrs];

        all_xMatrix0{exp_num}=generatedProblem.xMatrix0;
        all_xMatrixEst{exp_num}=sdpXMatrix;
    end

    figure;
    hold on;
    cdfplot(before_sdp_loc_error)
    cdfplot(mid_sdp_loc_error)
    cdfplot(after_sdp_loc_error)
    legend("Before",'Mid',"After")
    title("Anchors: "+dat_num_anchors)
    xlabel("Loc error (m)")
    xscale log
    hold off;

    figure;
    hold on;
    cdfplot(before_sdp_rmsd_error)
    cdfplot(mid_sdp_rmsd_error)
    cdfplot(after_sdp_rmsd_error)
    legend("Before",'Mid',"After")
    title("Anchors: "+dat_num_anchors)
    xlabel("RMSD error (m)")
    hold off;
    
    save("Data_svr_se/MergedLoc"+wavelengthMultiple+"S"+lklhd_dat.all_lkhd_pars{dat_num_anchors}{exp_num}.num_tags+"A"+...
        lklhd_dat.all_lkhd_pars{dat_num_anchors}{exp_num}.num_anchors+"anchType"+lklhd_dat.anchor_type+...
        "Freqs"+freqsToUse+".mat", ...
        "before_sdp_loc_error","mid_sdp_loc_error","after_sdp_loc_error", ...
        "before_sdp_rmsd_error","mid_sdp_rmsd_error","after_sdp_rmsd_error", ...
        "all_xMatrix0", "all_xMatrixEst")
end


function [distanceMatrix] = generateDistMat(xMatrixLkhd, distanceMatrixOriginal, ...
    origTagIdx, numSensors, numAnchors)
    distanceMatrix=zeros(numSensors,numSensors+numAnchors);
    
    distanceMatrixOriginalAjmat=distanceMatrixOriginal;
    distanceMatrixOriginalAjmat(size(distanceMatrixOriginal,1)+1:size(distanceMatrixOriginal,2),:)=0;
    distanceMatrixOriginalAjmat=distanceMatrixOriginalAjmat+distanceMatrixOriginalAjmat';
    
    for i=1:size(distanceMatrix,1)
        for j=1:size(distanceMatrix,2)
            if i>=j 
                continue;
            else
                %% Idea 2: Fix the tags that are localized well by lkhd
                % method and treat them as anchors. Create dist matrix
                % accordingly
                
                % disp("==> "+i + " " + j + ";")
                % i_converted=origTagIdx(i);
                % if j<=size(distanceMatrix,1) % j is till numSensors
                %     j_converted=origTagIdx(j);
                % else
                %     j_converted=j;
                % end

                % distanceMatrix(i,j)=distanceMatrixOriginalAjmat(i_converted, j_converted);

                i_converted=find(origTagIdx==i);
                if j<=size(distanceMatrix,1) % j is till numSensors
                    j_converted=find(origTagIdx==j);
                else
                    j_converted=j;
                end
                
                
                distanceMatrix(i_converted,j_converted)=distanceMatrixOriginalAjmat(i, j);
                distanceMatrix(j_converted,i_converted)=distanceMatrixOriginalAjmat(i, j);
        

                
            
            end
        end
    end
   

end

function [loc_errors] = calc_locError(xMatrix, aMatrix)
    c=size(xMatrix,2);
    loc_errors=zeros(1,c);
    
    for p=1:c
        loc_errors(p)=norm(xMatrix(:,p)-aMatrix(:,p));
    end
    
end

function [rmsd] = calc_rmsd(xMatrix, aMatrix)
    c=size(xMatrix,2);
    sd=0;
    for p=1:c
        sd=sd+(norm(xMatrix(:,p)-aMatrix(:,p)))^2;
    end
    msd=sd/c;
    rmsd=sqrt(msd);
end