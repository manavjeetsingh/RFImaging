internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/subprograms''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/sdpam7-windows''')
internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''C:/Users/Manavjeet Singh/Git/Localization-simulation/SFSDP/SFSDP/examples''')

% MAX_MIN_DEGREE=7;

repetitions=30;
dims=2;
all_num_sensors=[500];


num_anchors=16;
anchor_layout_type=0;
minimum_degree=dims+2;
show_plots=0;
radio_ranges=[0.1,0.15,0.2,0.25,0.3,0.35,0.4];
edgeAlgos=["equalInterval","distance","onePhase","twoPhase"];
% measurementErrorSDs=[0.05,0.1];
measurementErrorSDs=[0.05];
choice_grad_only=[0,1];
edgeAlgosNames=["arbitrary","angle","oneStage","twoStage"];
% colors=['r','m','b'];
colors=["#D95319","#77AC30","#0072BD","#aa721D"];
selection_seed=0;
rmsdVsRR_all=zeros(repetitions,length(radio_ranges),length(edgeAlgos));
rmsdVsRR_avg=zeros(length(radio_ranges),length(edgeAlgos));
rmsdVsRR_q1=zeros(length(radio_ranges),length(edgeAlgos));
rmsdVsRR_q3=zeros(length(radio_ranges),length(edgeAlgos));
% measurementErrorSDs=[0.02,0.05,0.1];
linksVsRR_all=zeros(repetitions,length(radio_ranges),length(edgeAlgos));
averageDeg_all=zeros(repetitions,length(radio_ranges),length(edgeAlgos));

plot_num=0;
for density = all_num_sensors
    
    for measurementErrorSD = measurementErrorSDs
        
        for grad_only=choice_grad_only
            
            
                for radio_rangeIdx = 1:length(radio_ranges)
                    fprintf("===============================\n")
                    fprintf("Starting with radio range: "+radio_ranges(radio_rangeIdx)+"; sensros: "+density+"\n")
                    fprintf("===============================\n")
                    
                    for edgeAlgoIdx=1:length(edgeAlgos)
                        
                        load("GradSDPComparison/S"+density+"A"+num_anchors+"RR"+radio_ranges(radio_rangeIdx)+"Reps"+repetitions+"MDeg"+minimum_degree+"MErrorSD"+measurementErrorSD...
                            +"EdgeAlgo"+edgeAlgos(edgeAlgoIdx)+"GradOnly"+grad_only+"AddedErrorMean0.02"+".mat", ...
                            'rmsds_1','totalMeasurements_1','degreeList_1')

                        
                        rmsdVsRR_all(:,radio_rangeIdx,edgeAlgoIdx)=rmsds_1(:,minimum_degree);
                        linksVsRR_all(:,radio_rangeIdx,edgeAlgoIdx)=totalMeasurements_1(:,minimum_degree);
                        averageDeg_all(:,radio_rangeIdx,edgeAlgoIdx)=mean(degreeList_1,2);
            
                        % quartilesRmsd_1=quantile(rmsds_1(:,minimum_degree),3)
                        % rmsdVsRR_q1(radio_rangeIdx,edgeAlgoIdx)=quartilesRmsd_1(1);
                        % rmsdVsRR_q3(radio_rangeIdx,edgeAlgoIdx)=quartilesRmsd_1(3);
                        % rmsdVsRR_avg(radio_rangeIdx,edgeAlgoIdx)=mean(rmsds_1(:,minimum_degree));
            
                        % quartilesRmsd_1=quantile(rmsds_1(:,minimum_degree),3);
            
                    end
            
                    
                end
                
            
            rmsdVsRR_all
            subplot(length(choice_grad_only),3,3*plot_num+1)
            
            hold on
            for edgeAlgoIdx = 1:length(edgeAlgos)
                % plot(radio_ranges(2:end),rmsdVsRR_avg(2:end,edgeAlgoIdx),'o-','DisplayName',edgeAlgos(edgeAlgoIdx))
                
                data=rmsdVsRR_all(:,1:end,edgeAlgoIdx);
                data=double(data);
                bins=string(radio_ranges(1:end));
                boxchart(data,"DisplayName",edgeAlgosNames(edgeAlgoIdx),'BoxFaceAlpha',.05, ...
                    'BoxFaceColor',colors(edgeAlgoIdx))
                set(gca,'XTickLabel',radio_ranges(1:end))
            
            
                plot(mean(data),'O-',"Color",colors(edgeAlgoIdx),"DisplayName",...
                    edgeAlgosNames(edgeAlgoIdx)+" mean",'LineWidth',3)
            
            end
            legend('Location','northeast')
            xlabel("Radio Range")
            ylabel("RMSD error")
            ylim([0.03,0.13])
            title("RMSD error v/s RR; Density="+density+"; Using SDP: "+(grad_only~=1))
            grid on
            hold off
            
            subplot(length(choice_grad_only),3,3*plot_num+2)
            % plot_num=plot_num+1;
            hold on
            for edgeAlgoIdx = 1:length(edgeAlgos)
                % plot(radio_ranges(2:end),rmsdVsRR_avg(2:end,edgeAlgoIdx),'o-','DisplayName',edgeAlgos(edgeAlgoIdx))
                
                data=linksVsRR_all(:,1:end,edgeAlgoIdx);
                data=double(data);
                bins=string(radio_ranges(1:end));
                boxchart(data,"DisplayName",edgeAlgosNames(edgeAlgoIdx),'BoxFaceAlpha',.05, ...
                    'BoxFaceColor',colors(edgeAlgoIdx))
                set(gca,'XTickLabel',radio_ranges(1:end))
            
            
                plot(mean(data),'O-',"Color",colors(edgeAlgoIdx),"DisplayName",...
                    edgeAlgosNames(edgeAlgoIdx)+" mean",'LineWidth',3)
        
                % for temp = 1:length(radio_ranges(3:end))
                %     caption = sprintf('%0.1f', mean(data(:,temp)));
                %     text(temp+0.1*edgeAlgoIdx,mean(data(:,temp))+500*edgeAlgoIdx, caption, 'FontSize', 12);
                % end
            
            end
            legend('Location','northwest')
            xlabel("Radio Range")
            ylabel("Links Measured")
            title("Links Measured v/s RR; Density="+density+"; Using SDP: "+(grad_only~=1))
            grid on
            hold off

            subplot(length(choice_grad_only),3,3*plot_num+3)
            plot_num=plot_num+1;
            hold on
            for edgeAlgoIdx = 1:length(edgeAlgos)
                % plot(radio_ranges(2:end),rmsdVsRR_avg(2:end,edgeAlgoIdx),'o-','DisplayName',edgeAlgos(edgeAlgoIdx))
                
                data=averageDeg_all(:,1:end,edgeAlgoIdx);
                data=double(data);
                bins=string(radio_ranges(1:end));
                boxchart(data,"DisplayName",edgeAlgosNames(edgeAlgoIdx),'BoxFaceAlpha',.05, ...
                    'BoxFaceColor',colors(edgeAlgoIdx))
                set(gca,'XTickLabel',radio_ranges(1:end))
            
            
                plot(mean(data),'O-',"Color",colors(edgeAlgoIdx),"DisplayName",...
                    edgeAlgosNames(edgeAlgoIdx)+" mean",'LineWidth',3)
        
                % for temp = 1:length(radio_ranges(3:end))
                %     caption = sprintf('%0.1f', mean(data(:,temp)));
                %     text(temp+0.1*edgeAlgoIdx,mean(data(:,temp))+500*edgeAlgoIdx, caption, 'FontSize', 12);
                % end
            
            end
            legend('Location','northwest')
            
            xlabel("Radio Range")
            ylabel("Average Degree")
            title("Avg Degree v/s RR; Density="+density+"; Using SDP: "+(grad_only~=1))
            grid on
            hold off
        end
        sgtitle("Std: "+measurementErrorSD+" Mean: 0.02")
    end
end