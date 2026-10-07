internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/subprograms''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/sdpam7-windows''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/examples''')

runs=2000;
measurement_error_var=0.05;
dims=2;
radio_range=0.15;
num_sensors=500;
num_anchors=4;
anchor_layout_type=0;
randomization_seed=144;
minimum_degree=5;
show_plots=0;

all_rmsds=zeros(1,runs);
for selection_seed = 1:runs
    rmsd=test_FSDP(dims,measurement_error_var,radio_range,num_sensors,anchor_layout_type,num_anchors,randomization_seed,minimum_degree,show_plots,selection_seed);
    all_rmsds(selection_seed)=rmsd;
end

% selection_seed=runs+1;
% while true
%     rmsd=test_FSDP(dims,measurement_error_var,radio_range,num_sensors,anchor_layout_type,num_anchors,randomization_seed,minimum_degree,show_plots,selection_seed);
%     all_rmsds(selection_seed)=rmsd;
%     selection_seed=selection_seed+1;
%     if abs(rmsd-max(all_rmsds)) < 0.01*max(all_rmsds) || selection_seed>=10000
%         break 
%     end
% end

minimum=min(all_rmsds)
maximum=max(all_rmsds)
diff=maximum-minimum
avg=mean(all_rmsds)

save("RandLinkSelectionResults/S"+num_sensors+"A"+num_anchors+"RR"+radio_range+"N"+measurement_error_var+"MMDeg"+minimum_degree+"Runs"+runs+".mat")