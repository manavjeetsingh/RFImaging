internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/subprograms''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/sdpam7-windows''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/examples''')

MAX_MIN_DEGREE=14;

repetitions=20;
% measurement_error_var=0.001;
dims=2;
% radio_range=0.15;
num_sensors=500;
num_anchors=4;
anchor_layout_type=0;
randomization_seed=144;
minimum_degree=dims+3;
show_plots=0;
radio_ranges=[0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0,1.5,2.0];
errorMMDeg4_001=zeros(1,length(radio_ranges));
errorMMDeg4_01=zeros(1,length(radio_ranges));
errorMMDeg4_05=zeros(1,length(radio_ranges));
errorMMDeg4_08=zeros(1,length(radio_ranges));

figure;
for radio_range_idx = 1:length(radio_ranges)
    radio_range=radio_ranges(radio_range_idx);
    fprintf("===============================\n")
    fprintf("Starting with radio range: "+radio_range+"\n")
    fprintf("===============================\n")
    
    % rmsds_001=zeros(repetitions,MAX_MIN_DEGREE);
    % for i=MAX_MIN_DEGREE:-1:3
    %     % figure();
    %     for selection_seed = 1:repetitions
    %         rmsd=test_FSDP(dims,0.001,radio_range,num_sensors,anchor_layout_type,num_anchors,randomization_seed,i,show_plots,selection_seed);
    %         rmsds_001(selection_seed,i)=rmsd;
    %     end
    % end
    % 
    % 
    % rmsds_01=zeros(repetitions,MAX_MIN_DEGREE);
    % for i=MAX_MIN_DEGREE:-1:3
    %     % figure();
    %     for selection_seed = 1:repetitions
    %         rmsd=test_FSDP(dims,0.01,radio_range,num_sensors,anchor_layout_type,num_anchors,randomization_seed,i,show_plots,selection_seed);
    %         rmsds_01(selection_seed,i)=rmsd;
    %     end
    % end
    % 
    % rmsds_05=zeros(repetitions,MAX_MIN_DEGREE);
    % for i=MAX_MIN_DEGREE:-1:3
    %      % figure();
    %     for selection_seed = 1:repetitions
    %         rmsd=test_FSDP(dims,0.05,radio_range,num_sensors,anchor_layout_type,num_anchors,randomization_seed,i,show_plots,selection_seed);
    %         rmsds_05(selection_seed,i)=rmsd;
    %     end
    % end
    % 
    % rmsds_08=zeros(repetitions,MAX_MIN_DEGREE);
    % for i=MAX_MIN_DEGREE:-1:3
    %      % figure();
    %     for selection_seed = 1:repetitions
    %         rmsd=test_FSDP(dims,0.08,radio_range,num_sensors,anchor_layout_type,num_anchors,randomization_seed,i,show_plots,selection_seed);
    %         rmsds_08(selection_seed,i)=rmsd;
    %     end
    % end
    % 
    % all_rmsds={};
    % all_rmsds.rmsds_001=rmsds_001;
    % all_rmsds.rmsds_01=rmsds_01;
    % all_rmsds.rmsds_05=rmsds_05;
    % all_rmsds.rmsds_08=rmsds_08;
    
    load("ErrorRRDegree/S"+num_sensors+"A"+num_anchors+"RR"+radio_range+"Reps"+repetitions+"MMDeg"+MAX_MIN_DEGREE+"ConstantLayout.mat")
    errorMMDeg4_001(radio_range_idx)=mean(rmsds_001(:,4));
    errorMMDeg4_01(radio_range_idx)=mean(rmsds_01(:,4));
    errorMMDeg4_05(radio_range_idx)=mean(rmsds_05(:,4));
    errorMMDeg4_08(radio_range_idx)=mean(rmsds_08(:,4));
    % break
    % plot_rmsds(MAX_MIN_DEGREE,rmsds_001,rmsds_01,rmsds_05,rmsds_08,"Error not normalized w.r.t range","Radio range: "+radio_range,length(radio_ranges),radio_range_idx,[0,.15])
    
    % rmsds_001=rmsds_001/radio_range*100;
    % rmsds_01=rmsds_01/radio_range*100;
    % rmsds_05=rmsds_05/radio_range*100;
    % rmsds_08=rmsds_08/radio_range*100;
    
    % plot_rmsds(MAX_MIN_DEGREE,rmsds_001,rmsds_01,rmsds_05,rmsds_08,"Error normalized in terms of percentage of radio range","Radio range: "+radio_range,length(radio_ranges),radio_range_idx,[])
end
hold on
plot(radio_ranges,errorMMDeg4_001,"bO--",'DisplayName',"0.001")
plot(radio_ranges,errorMMDeg4_01,"rO--",'DisplayName',"0.01")
plot(radio_ranges,errorMMDeg4_05,"gO--",'DisplayName',"0.05")
plot(radio_ranges,errorMMDeg4_08,"cO--",'DisplayName',"0.08")
xlabel("Radio Range");
ylabel("RMSD error");
title("Radio Range v/s RMSD at minDegree="+minimum_degree)
legend
hold off
