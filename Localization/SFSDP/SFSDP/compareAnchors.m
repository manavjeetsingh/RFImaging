internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/subprograms''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/sdpam7-windows''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/examples''')

% MAX_MIN_DEGREE=7;

repetitions=50;
dims=2;
% all_num_sensors=[500,750,1000];
all_num_sensors=[500];


num_anchors=[4,8,12,16];
% num_anchors=[12];
anchor_layout_type=6   ;
minimum_degree=dims+2;
show_plots=0;
radio_ranges=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4];
edgeAlgos=["equalInterval","distance","onePhase","twoPhase"];
selection_seed=0;
% measurementErrorSDs=[0.05,0.1];
measurementErrorSDs=[0.05];

for num_anchor=num_anchors
    for measurementErrorSD = measurementErrorSDs
        for num_sensors = all_num_sensors
            for radio_range = radio_ranges
                fprintf("===============================\n")
                fprintf("Starting with radio range: "+radio_range+"; sensros: "+num_sensors+"\n")
                fprintf("===============================\n")
                
                for edgeAlgo=edgeAlgos
                    rmsds_1=zeros(repetitions,minimum_degree);
                    totalMeasurements_1=zeros(repetitions,minimum_degree);
                    % figure();
                    for randomization_seed = 1:repetitions
                        [rmsd,totalMeasurements]=test_FSDP(dims,measurementErrorSD,radio_range,num_sensors,anchor_layout_type,num_anchor,randomization_seed,minimum_degree,show_plots,selection_seed,edgeAlgo);
                        rmsd
                        rmsds_1(randomization_seed,minimum_degree)=rmsd;
                        totalMeasurements_1(randomization_seed,minimum_degree)=totalMeasurements;
                    end
                    save("NumAnchorComparison/S"+num_sensors+"A"+num_anchor+"RR"+radio_range+"Reps"+repetitions+"MDeg"+minimum_degree+"MErrorSD"+measurementErrorSD...
                        +"EdgeAlgo"+edgeAlgo+"AddedErrorMean0.02_grad_method3"+".mat")
                end
        
                
            end
            
        end
    end
end
