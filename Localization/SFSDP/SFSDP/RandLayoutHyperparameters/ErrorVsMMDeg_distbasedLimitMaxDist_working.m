internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/subprograms''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/sdpam7-windows''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/examples''')

MAX_MIN_DEGREE=7;
repetitions=100;
dims=2;
all_num_sensors=[100,300,500];
num_anchors=4;
anchor_layout_type=0;
show_plots=0;
radio_ranges=[0.15,0.2,0.3,0.4];
selection_seed=0;

plot_nxo=1;
for num_sensor_idx = 1:length(all_num_sensors)
    % Resetting the variables
    MAX_MIN_DEGREE=7;
    all_num_sensors=[100,300,500];
    radio_ranges=[0.15,0.2,0.3,0.4];
    
    num_sensors=all_num_sensors(num_sensor_idx);
    for radio_range_idx = 1:length(radio_ranges)
        radio_range=radio_ranges(radio_range_idx);
        
        load("RandLayoutHyperparameters/S"+num_sensors+"A"+num_anchors+"RR"+radio_range+"Reps"+repetitions+"MDeg"+MAX_MIN_DEGREE+"DynLayoutLimitMaxDist_working.mat")
        all_num_sensors=[100,300,500];
        radio_ranges=[0.15,0.2,0.3,0.4];
        % subplot(length(all_num_sensors),length(radio_ranges),plot_no)
        plot_nxo
        subplot(3,4,plot_nxo)
        
        plot_rmsds(MAX_MIN_DEGREE,rmsds_1,rmsds_01,rmsds_05,{"Error not normalized w.r.t range","Radio range: "+radio_range+"Num Sensors: "+num_sensors},num_sensors)
        
        plot_nxo=plot_nxo+1;
    end
    
end



function plot_rmsds(MAX_MIN_DEGREE,rmsds_1,rmsds_01,rmsds_05,plt_title,num_sensors)


    % figure
    % Confidence intervels of measurement error 0.1
    quartiles_1=quantile(rmsds_1(:,3:MAX_MIN_DEGREE),3);
    % quartiles_1=quantile([rmsds_1(:,3:MAX_MIN_DEGREE),rmsds_1(:,num_sensors)],3);
    % yconf_1 = [max([rmsds_1(:,3:MAX_MIN_DEGREE),rmsds_1(:,num_sensors)]) min([rmsds_1(:,MAX_MIN_DEGREE:-1:3),rmsds_1(:,num_sensors)])];
    yconf_1 = [quartiles_1(3,:) quartiles_1(1,end:-1:1)];
    % Confidence intervels of measurement error 0.01
    quartiles_01=quantile(rmsds_01(:,3:MAX_MIN_DEGREE),3);
    % quartiles_01=quantile([rmsds_01(:,3:MAX_MIN_DEGREE),rmsds_01(:,num_sensors)],3);
    % yconf_01 = [max([rmsds_01(:,3:MAX_MIN_DEGREE),rmsds_01(:,num_sensors)]) min([rmsds_01(:,MAX_MIN_DEGREE:-1:3),rmsds_01(:,num_sensors)])];
    yconf_01 = [quartiles_01(3,:) quartiles_01(1,end:-1:1)];
    % Confidence intervels of measurement error 0.05
    quartiles_05=quantile(rmsds_05(:,3:MAX_MIN_DEGREE),3);
    % quartiles_05=quantile([rmsds_05(:,3:MAX_MIN_DEGREE),rmsds_05(:,num_sensors)],3);
    % yconf_05 = [max([rmsds_05(:,3:MAX_MIN_DEGREE),rmsds_05(:,num_sensors)]) min([rmsds_05(:,MAX_MIN_DEGREE:-1:3),rmsds_05(:,num_sensors)])];
    yconf_05 = [quartiles_05(3,:) quartiles_05(1,end:-1:1)];

    
    x=[3:MAX_MIN_DEGREE,MAX_MIN_DEGREE+1];
    x_small=3:MAX_MIN_DEGREE;
    xconf_small=[x_small x_small(end:-1:1)] ;
    xconf = [x x(end:-1:1)] ;
    
    
    hold on
    
    fill1 = fill(xconf_small,yconf_1,'blue');
    fill1.FaceAlpha=0.1;
    fill1.EdgeColor = 'none';  
    fill1.HandleVisibility="off";

    fill01 = fill(xconf_small,yconf_01,'red');
    fill01.FaceAlpha=0.1;
    fill01.EdgeColor = 'none'; 
    fill01.HandleVisibility="off";

    fill05 = fill(xconf_small,yconf_05,'green');
    fill05.FaceAlpha=0.1;
    fill05.EdgeColor = 'none'; 
    fill05.HandleVisibility="off";
    

    
    
    % plot(x,mean([rmsds_1(:,3:MAX_MIN_DEGREE),rmsds_1(:,num_sensors)]),'bO--','DisplayName',"0.1");
    % plot(x,[mean(rmsds_1(:,3:MAX_MIN_DEGREE)),rmsds_1(1,num_sensors)],'bO--','DisplayName',"0.1");
    plot(x_small,mean([rmsds_1(:,3:MAX_MIN_DEGREE)]),'bO--','DisplayName',"0.1");
    % plot(x,mean([rmsds_01(:,3:MAX_MIN_DEGREE),rmsds_01(:,num_sensors)]),'rO--','DisplayName',"0.01");
    % plot(x,[mean(rmsds_01(:,3:MAX_MIN_DEGREE)),rmsds_01(1,num_sensors)],'rO--','DisplayName',"0.01");
    sorted_rmsds_01=sort(rmsds_01(:,3:MAX_MIN_DEGREE));
    plot(x_small,mean([sorted_rmsds_01(1:end-1)]),'rO--','DisplayName',"0.01");
    % plot(x,mean([rmsds_05(:,3:MAX_MIN_DEGREE),rmsds_05(:,num_sensors)]),'gO--','DisplayName',"0.05");
    % plot(x,[mean(rmsds_05(:,3:MAX_MIN_DEGREE)),rmsds_05(1,num_sensors)],'gO--','DisplayName',"0.05");
    plot(x_small,mean([rmsds_05(:,3:MAX_MIN_DEGREE)]),'gO--','DisplayName',"0.05");
    
    xlabel("minDegree (last value represents minDegree=numSensors)")
    ylabel("RMSD")
    title(plt_title);
    legend
    grid on
    hold off
end