%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [distanceMatrix] = selectEdgesForSFSDP_angle(sDim,noOfSensors,distanceMatrix,xMatrix,minDegree,ubdForSenToAnchorEdge,rand_state,plot_graph)




% startingTime = tic; 
[rowSize,colSize] = size(distanceMatrix);
noOfAnchors = colSize - noOfSensors; 
   
aj_mat=distanceMatrix;
aj_mat(colSize,colSize)=0;
aj_mat=aj_mat+aj_mat';
G=graph(aj_mat);
edges_before=numedges(G);
if plot_graph==1
    % figure();
    % plot(G);
    % title("Before sensor link selection");
    plotSensorGraph(xMatrix,aj_mat,"Before Link Selection")
end 
deg=mean(degree(G))
before_mat_size=size(aj_mat)


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
if noOfNzz == 0
    fprinrf('## no edges between sensors\n');
    
else
    % Angle aware link selection.

    distanceMat2=zeros(rowSize,rowSize);
    degree_count=zeros(1,rowSize);
    visited_links=java.util.HashSet;
    k = 0;

    for node_idx = 1:rowSize
        node_loc=xMatrix(:,node_idx);
        link_angles=[]; % (other_node_idx,angle)

        for other_node_idx = 1:rowSize
            other_node_loc=xMatrix(:,other_node_idx);        
            if aj_mat(node_idx,other_node_idx)~=0 && other_node_idx~=node_idx ...
                     && ~visited_links.contains(node_idx+" "+other_node_idx)

                % Getting the angle between all links of the node
                % 1. For a point (x,y), take a reference line (x,y),(2,y).
                % 2. Get the angle of each link joining the node, and 
                % % the other_node with the reference line. This will give
                % % an array ref_angles, where length(ref_angles) = no. of
                % % links.
                % 3. Sort the ref_angles array.
                               
                % 1.
                ref_point_1=[100,node_loc(2)];
                % ref_point_2=[node_loc(1),node_loc(2)];
                v_1 = [ref_point_1(1),ref_point_1(2),0] - [node_loc(1),node_loc(2),0];
                v_2 = [other_node_loc(1),other_node_loc(2),0] ...
                        - [node_loc(1),node_loc(2),0];
                % 2.
                % angle in radians
                % theta = atan2(norm(cross(v_1, v_2)), dot(v_1, v_2));
                theta=vecangle360(v_1,v_2,[0,0,1]);
                theta=deg2rad(theta);
                theta=mod(theta,2*pi);
                theta_j=julius_angle(v_1,v_2);
                theta_j=deg2rad(theta_j);
                                
                assert(abs(theta-theta_j)<1e-10);

                
                link_angles=[link_angles;[other_node_idx,theta]];
                               

            end   
            
        end

        
        
        
        if isempty(link_angles)
            continue
        end

        % 3.
        sorted_link_angles=sortrows(link_angles,2);
        
        % Select minLinks-currentDeg 
        currentDegree=degree_count(node_idx);
        wanted_links=minDegree-currentDegree;
        % wanted_links
        if wanted_links<=0
            continue
        end
        step=max(floor(length(sorted_link_angles)/wanted_links),1);

        to_select=1:step:length(sorted_link_angles(:,1));
        
        for row_idx=1:step:length(sorted_link_angles(:,1))
            
            other_node_idx=sorted_link_angles(row_idx,1);
               
            other_node_loc=xMatrix(:,other_node_idx);
            
            distanceMat2(node_idx,other_node_idx)=max(distanceMatrix(node_idx,other_node_idx)...
                                            ,distanceMatrix(other_node_idx,node_idx));
            %distanceMat2(other_node_idx,node_idx)=norm(node_loc-other_node_loc);

            % mark the selected links into already visited links 
            visited_links.add(node_idx+" "+other_node_idx);
            visited_links.add(other_node_idx+" "+node_idx);
            
            degree_count(node_idx)=degree_count(node_idx)+1;
            degree_count(other_node_idx)=degree_count(other_node_idx)+1;
        end

        % 
        % if sum(degree_count)>=totalDegrees
        %     break
        % end
    end

    
    distanceMat2 = ...
        [sparse(distanceMat2),...
        senToAnchorDistMat];

    
    
    
end

% fprintf('##2 %6.1f\n',toc(startingTime));
% startingTime = tic; 


distanceMat2(:,1:noOfSensors) = distanceMat2(:,1:noOfSensors) + distanceMat2(:,1:noOfSensors)';


%distanceMatrix = [distanceMat2(invPermutation,invPermutation),distanceMat2(invPermutation,noOfSensors+1:colSize)];
distanceMatrix = triu(distanceMat2,1); 

% full(distanceMatrix)

distanceMatrix = sparse(distanceMatrix);



fprintf("---------------------")
aj_mat=distanceMatrix;
aj_mat(colSize,colSize)=0;
aj_mat=aj_mat+aj_mat';
fprintf("---------------------")
G=graph(aj_mat);
edges_after=numedges(G);
after_deg=mean(degree(G))
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