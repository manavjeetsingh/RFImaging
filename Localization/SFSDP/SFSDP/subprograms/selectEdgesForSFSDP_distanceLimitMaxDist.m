%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [distanceMatrix,totalMeasurements,before_degreeList, degreeList] = selectEdgesForSFSDP_distanceLimitMaxDist(sDim,noOfSensors,distanceMatrix,xMatrix,minDegree,ubdForSenToAnchorEdge,rand_state,plot_graph)




% startingTime = tic; 
[rowSize,colSize] = size(distanceMatrix);
noOfAnchors = colSize - noOfSensors; 
   
aj_mat=distanceMatrix
aj_mat(colSize,colSize)=0; % changes the matrix size to colSize X colSize
%   wih new elements set to 0.
aj_mat=aj_mat+aj_mat'
G=graph(aj_mat);
edges_before=numedges(G);
if plot_graph==1
    % figure();
    % plot(G);
    % title("Before sensor link selection");
    plotSensorGraph(xMatrix,aj_mat,"Before Link Selection")
end 
before_degreeList=degree(G);
before_mat_size=size(aj_mat);


% Reordering sensors randomly --->
sensorDistMat = distanceMatrix(:,1:noOfSensors)+distanceMatrix(:,1:noOfSensors)'; %since the distanceMatrix is directed T1->T2!=0, T2->T1=0. Making it undirected.
% rand('state',3202); 
rand('state',rand_state); 
[temp,permutation] = sort(rand(1,noOfSensors));
[temp,invPermutation] = sort(permutation);

% sensorDistMat = sensorDistMat(permutation,permutation); 
% distanceMat2 = [sensorDistMat,distanceMatrix(permutation,noOfSensors+1:colSize)];
% distanceMat2 = triu(distanceMat2,1);

%Changes by MS: Making changes such that there is no randomization fo links.

distanceMat2 = [sensorDistMat,distanceMatrix(:,noOfSensors+1:colSize)];
distanceMat2 = triu(distanceMat2,1);

% <--- Reordering sensors randomly 
% clear distanceMatrix

% fprintf('##0 %6.1f\n',toc(startingTime));
% startingTime = tic; 

% Selecting edges between sensors and anchors 
% ---> 
senToAnchorDistMat = sparse(rowSize,noOfAnchors); 
countVector = sparse(1,rowSize); 
for r=1:noOfAnchors
    nzIdx = find(distanceMat2(:,noOfSensors+r)' > 0); 
    if ~isempty(nzIdx)
        for p=nzIdx
            if (countVector(p) < ubdForSenToAnchorEdge)
                senToAnchorDistMat(p,r) = distanceMat2(p,noOfSensors+r); 
                countVector(p) = countVector(p) + 1;
            end
        end
    end
end
% < ---
% Selecting edges between sensors and anchors 

% fprintf('##1 %6.1f\n',toc(startingTime));
% startingTime = tic; 

% Selecting edges between sensors and anchors 
% ---> 
countVector = min([countVector;repmat(sDim,1,noOfSensors)],[],1);
noOfEffectiveDegrees = sum(countVector);
totalDegrees = noOfSensors*minDegree; 
[nzzRowIdx,nzzColIdx,nzzValue] = find(distanceMat2(:,1:noOfSensors)'); 
% clear distanceMat2
noOfNzz = length(nzzRowIdx); 

distanceMeasurementsMat=zeros(size(aj_mat));

if noOfNzz == 0
    fprintf('## no edges between sensors\n');
    
else


    distanceMat2=zeros(rowSize,rowSize);
    degree_count=zeros(1,rowSize);
    visited_links=java.util.HashSet;
    

    for node_idx = 1:rowSize
        node_loc=xMatrix(:,node_idx);
        selected_links_nodes=[];
        unselected_links_nodes=[];

        for other_node_idx = 1:rowSize
            if length(selected_links_nodes)>=minDegree
                break
            end

            other_node_loc=xMatrix(:,other_node_idx);   
            % dist b/w node and other_node
            
            dist_1=max(distanceMatrix(node_idx,other_node_idx)...
                                ,distanceMatrix(other_node_idx,node_idx));
            if dist_1~=0
                distanceMeasurementsMat(node_idx,other_node_idx)=1;
                distanceMeasurementsMat(other_node_idx,node_idx)=1;
            end
            if dist_1 > 0.2
                continue
            end

            if aj_mat(node_idx,other_node_idx)~=0 && other_node_idx~=node_idx ...
                     && ~visited_links.contains(node_idx+" "+other_node_idx)
                
                %select the first link if no other link is selected yet
                if degree_count(node_idx)==0
                    selected_links_nodes=[selected_links_nodes,other_node_idx];
                    distanceMeasurementsMat(node_idx,other_node_idx)=1;
                    distanceMeasurementsMat(other_node_idx,node_idx)=1;
                    distanceMat2(node_idx,other_node_idx)=max(distanceMatrix(node_idx,other_node_idx)...
                                            ,distanceMatrix(other_node_idx,node_idx));

                    % mark the selected links into already visited links 
                    visited_links.add(node_idx+" "+other_node_idx);
                    visited_links.add(other_node_idx+" "+node_idx);
                    
                    degree_count(node_idx)=degree_count(node_idx)+1;
                    degree_count(other_node_idx)=degree_count(other_node_idx)+1;
                    continue
                else
                    add_link=true;

                    % Compare with all already selected nodes
                    for selected_links_idx = 1:length(selected_links_nodes)
                        other_other_node_idx=selected_links_nodes(selected_links_idx);
                        
                        distanceMeasurementsMat(node_idx,other_node_idx)=1;
                        distanceMeasurementsMat(other_node_idx,node_idx)=1;
                        % dist b/w node and other_node
                        dist_1=max(distanceMatrix(node_idx,other_node_idx)...
                                            ,distanceMatrix(other_node_idx,node_idx));
                        
                        distanceMeasurementsMat(node_idx,other_other_node_idx)=1;
                        distanceMeasurementsMat(other_other_node_idx,node_idx)=1;
                        % dist b/w node and other_other_node
                        dist_2=max(distanceMatrix(node_idx,other_other_node_idx)...
                                            ,distanceMatrix(other_other_node_idx,node_idx));
                        % dist b/w other_node and other_other_node
                        dist_3=max(distanceMatrix(other_other_node_idx,other_node_idx)...
                                            ,distanceMatrix(other_node_idx,other_other_node_idx));
                        % If other_node and other_other_node are not in
                        % radio range
                        if dist_3==0
                            continue
                        else
                            % Add the reading between other_node and
                            % other_other_node only if they are reachable
                            distanceMeasurementsMat(other_node_idx,other_other_node_idx)=1;
                            distanceMeasurementsMat(other_other_node_idx,other_node_idx)=1;
                        end

                        % if node_idx==47
                        %     fprintf(node_idx+" "+other_node_idx+" "+other_other_node_idx+"\n")
                        % end
                        
                        % if clash with any other already used link
                        % don't add it.
                        if ~should_select_node(dist_1, dist_2, dist_3,60,node_idx)
                            
                            unselected_links_nodes=[unselected_links_nodes,other_node_idx];
                            add_link=false;
                            break
                        end
                        
                    end
                    if add_link
                        selected_links_nodes=[selected_links_nodes,other_node_idx];
                        distanceMeasurementsMat(node_idx,other_node_idx)=1;
                        distanceMeasurementsMat(other_node_idx,node_idx)=1;
                        distanceMat2(node_idx,other_node_idx)=max(distanceMatrix(node_idx,other_node_idx)...
                                            ,distanceMatrix(other_node_idx,node_idx));
                        % mark the selected links into already visited links 
                        visited_links.add(node_idx+" "+other_node_idx);
                        visited_links.add(other_node_idx+" "+node_idx);
                        
                        degree_count(node_idx)=degree_count(node_idx)+1;
                        degree_count(other_node_idx)=degree_count(other_node_idx)+1;
                    end
                              
                end
            end

        end
        
        % If enough nodes are not selected, sample nodes uniformaly
        if degree_count(node_idx)<minDegree
            currentDegree=degree_count(node_idx);
            wanted_links=minDegree-currentDegree;
            step=max(floor(length(unselected_links_nodes)/wanted_links),1);
            unselected_links_nodes;
            selected_nodes=1:step:length(unselected_links_nodes);
            for row_idx=1:step:length(unselected_links_nodes)

                other_node_idx=unselected_links_nodes(row_idx);

                distanceMeasurementsMat(node_idx,other_node_idx)=1;
                distanceMeasurementsMat(other_node_idx,node_idx)=1;
                distanceMat2(node_idx,other_node_idx)=max(distanceMatrix(node_idx,other_node_idx)...
                                                ,distanceMatrix(other_node_idx,node_idx));
                %distanceMat2(other_node_idx,node_idx)=norm(node_loc-other_node_loc);

                % mark the selected links into already visited links 
                visited_links.add(node_idx+" "+other_node_idx);
                visited_links.add(other_node_idx+" "+node_idx);

                degree_count(node_idx)=degree_count(node_idx)+1;
                degree_count(other_node_idx)=degree_count(other_node_idx)+1;
            end
        end


    end


       
end
totalMeasurements=sum(distanceMeasurementsMat,'all')/2;
fprintf("Total distance measurements=%d\n",totalMeasurements)
    
    distanceMat2 = ...
        [sparse(distanceMat2),...
        senToAnchorDistMat];

    
    
    


% fprintf('##2 %6.1f\n',toc(startingTime));
% startingTime = tic; 


distanceMat2(:,1:noOfSensors) = distanceMat2(:,1:noOfSensors) + distanceMat2(:,1:noOfSensors)';


%distanceMatrix = [distanceMat2(invPermutation,invPermutation),distanceMat2(invPermutation,noOfSensors+1:colSize)];
distanceMatrix = triu(distanceMat2,1); 

full(distanceMatrix);

distanceMatrix = sparse(distanceMatrix);



fprintf("---------------------")
aj_mat=distanceMatrix;
aj_mat(colSize,colSize)=0;
aj_mat=aj_mat+aj_mat';
fprintf("---------------------")
G=graph(aj_mat);
edges_after=numedges(G);
after_deg=mean(degree(G))
degreeList=degree(G);
mat_size=size(aj_mat)
if plot_graph==1
    % % Replacing the following graph with the location aware graph
    % figure();
    % plot(G);
    % title("After sensor link selection");
    
    % Location sensitive sensor graph
    plotSensorGraph(xMatrix,aj_mat,"After Link Selection")
end
edges_before
edges_after
edges_reduction_percent=(edges_before-edges_after)/edges_before*100


% fprintf('##3 %6.1f\n',toc(startingTime));
% 
% XXXXX

return
%%%%% end of selectEdgesForSFSDP2 %%%%%
end

% dist_1: b/w node and other_node
% dist_2: b/w node and other_other_node
% dist_3: b/w other_node and other_other_node
% min_angle: min acceptable angle in degrees
function decision=should_select_node(dist_1, dist_2, dist_3, min_angle,node_idx)
    % Give false if the angle with the node is less than min_angle.
    decision=false;

    angle_rad=acos((dist_1^2+dist_2^2-dist_3^2)/(2*dist_1*dist_2));
    angle_deg=rad2deg(angle_rad);
    if angle_deg>=min_angle
        decision=true;
    end
    % if node_idx==47
    %     angle_deg
    %     fprintf(full(dist_1)+" "+full(dist_2)+" "+full(dist_3)+"\n")
    % end

    
end

function a = vecangle360(v1,v2,n)
    x = cross(v1,v2);
    c = sign(dot(x,n)) * norm(x);
    a = atan2d(c,dot(v1,v2));
end

function a = julius_angle(v1,v2,n)
    % Define the points
    v1=v1(1:2);
    v2=v2(1:2);
    
       
    % Calculate the angles of AB and BC relative to the x-axis
    angleAB = atan2(v1(2), v1(1));
    angleBC = atan2(v2(2), v2(1));
    
    % Calculate the angle between AB and BC in anti-clockwise direction
    angleDiff = angleBC - angleAB;
    
    % Normalize the angle to be between 0 and 2*pi
    angleDiff = mod(angleDiff, 2*pi);
    
    % Convert to degrees
    angleDegrees = rad2deg(angleDiff);
    a=angleDegrees;
    % Display the angle
    % disp(['The anti-clockwise angle between the lines is ', num2str(angleDegrees), ' degrees.']);
end