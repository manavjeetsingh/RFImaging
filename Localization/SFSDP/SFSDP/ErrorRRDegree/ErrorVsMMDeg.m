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
% minimum_degree=dims+3;
show_plots=0;
radio_ranges=[0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0,1.5,2.0];

figure;
for radio_range_idx = 1:length(radio_ranges)

    fprintf("===============================\n")
    fprintf("Starting with radio range: "+radio_range+"\n")
    fprintf("===============================\n")
    radio_range=radio_ranges(radio_range_idx);
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
    
    plot_rmsds(MAX_MIN_DEGREE,rmsds_001,rmsds_01,rmsds_05,rmsds_08,"Error not normalized w.r.t range","Radio range: "+radio_range,length(radio_ranges),radio_range_idx,[0,.15])
    
    rmsds_001=rmsds_001/radio_range*100;
    rmsds_01=rmsds_01/radio_range*100;
    rmsds_05=rmsds_05/radio_range*100;
    rmsds_08=rmsds_08/radio_range*100;
    
    % plot_rmsds(MAX_MIN_DEGREE,rmsds_001,rmsds_01,rmsds_05,rmsds_08,"Error normalized in terms of percentage of radio range","Radio range: "+radio_range,length(radio_ranges),radio_range_idx,[])
end



function plot_rmsds(MAX_MIN_DEGREE,rmsds_001,rmsds_01,rmsds_05,rmsds_08,plt_title,plt_subtitle,total_plots,plot_pos,y_range)
    % figure(1)
    subplot(total_plots/2,2,plot_pos)
    % Confidence intervels of measurement error 0.001
    yconf_001 = [max(rmsds_001(:,3:MAX_MIN_DEGREE)) min(rmsds_001(:,MAX_MIN_DEGREE:-1:3))];
    % Confidence intervels of measurement error 0.01
    yconf_01 = [max(rmsds_01(:,3:MAX_MIN_DEGREE)) min(rmsds_01(:,MAX_MIN_DEGREE:-1:3))];
    % Confidence intervels of measurement error 0.001
    yconf_05 = [max(rmsds_05(:,3:MAX_MIN_DEGREE)) min(rmsds_05(:,MAX_MIN_DEGREE:-1:3))];
    % Confidence intervels of measurement error 0.001
    yconf_08 = [max(rmsds_08(:,3:MAX_MIN_DEGREE)) min(rmsds_08(:,MAX_MIN_DEGREE:-1:3))];
    
    x=3:MAX_MIN_DEGREE;
    xconf = [x x(end:-1:1)] ;
    
    hold on
    
    fill001 = fill(xconf,yconf_001,'blue');
    fill001.FaceAlpha=0.1;
    fill001.EdgeColor = 'none';  
    fill001.HandleVisibility="off";
    
    fill01 = fill(xconf,yconf_01,'red');
    fill01.FaceAlpha=0.1;
    fill01.EdgeColor = 'none'; 
    fill01.HandleVisibility="off";
    
    fill05 = fill(xconf,yconf_05,'green');
    fill05.FaceAlpha=0.1;
    fill05.EdgeColor = 'none'; 
    fill05.HandleVisibility="off";
    
    fill08 = fill(xconf,yconf_08,'cyan');
    fill08.FaceAlpha=0.1;
    fill08.EdgeColor = 'none'; 
    fill08.HandleVisibility="off";
    
    
    plot(x,mean(rmsds_001(:,3:MAX_MIN_DEGREE)),'bO--','DisplayName',"0.001");
    plot(x,mean(rmsds_01(:,3:MAX_MIN_DEGREE)),'rO--','DisplayName',"0.01");
    plot(x,mean(rmsds_05(:,3:MAX_MIN_DEGREE)),'gO--','DisplayName',"0.05");
    plot(x,mean(rmsds_08(:,3:MAX_MIN_DEGREE)),'cO--','DisplayName',"0.08"); 
    title(plt_title);
    subtitle(plt_subtitle);
    if length(y_range)~=0
        ylim(y_range);
        yticks(y_range(1):0.02:y_range(2))
        % fprintf("ylim")
    end
    grid on
    legend
    hold off
    % print('plots.pdf', '-dpdf');
end