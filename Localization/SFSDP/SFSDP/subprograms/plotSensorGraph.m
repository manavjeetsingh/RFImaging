function plotSensorGraph(node_locations, aj_mat,title_text)
%PLOTSENSORGRAPH Summary of this function goes here
%   Detailed explanation goes here
    figure;
    [~,c]=size(node_locations);
    ref_angles=dictionary();
    %Adding a dummy value in the dict
    ref_angles{0}=[0,0,0];
    hold on
    % scatter([0,0,1,1],[0,1,0,1])
    
    done_links=java.util.HashSet;
    runs=0;
    node_degrees=zeros(1,c);
    node_max_angles=zeros(1,c);
    node_max_linkdist=zeros(1,c);
    all_link_dists=[];
    for node_idx = 1:c
        node_loc=node_locations(:,node_idx);
        max_linkdist=0;
        for other_node_idx = 1:c
            % Saving the degree of the nodes
            if aj_mat(node_idx,other_node_idx)~=0 && other_node_idx~=node_idx
                other_node_loc=node_locations(:,other_node_idx);

                % Updating the degree of the node
                node_degrees(node_idx)=node_degrees(node_idx)+1;
                
                % Saving the max dist link of the current node
                linkdist=norm(node_loc-other_node_loc);
                if linkdist>max_linkdist
                    max_linkdist=linkdist;
                end
                
            
                % Plot links in a non repetitive way
                if ~done_links.contains(node_idx+" "+other_node_idx)
                    all_link_dists=[all_link_dists,norm(node_loc-other_node_loc)];
                    
                    % Plotting the link on the sensor location aware graph.
                    plot([node_loc(1),other_node_loc(1)],[node_loc(2),other_node_loc(2)], ...
                    'Color', [0.2 0.5 0.9 0.5], 'LineWidth',0.5)
                    done_links.add(node_idx+" "+other_node_idx);
                    done_links.add(other_node_idx+" "+node_idx);
                    runs=runs+1;
                end

                % Getting the largest angle between all links of the node
                % 1. For a point (x,y), take a reference line (x,y),(2,y).
                % 2. Get the angle of each link joining the node, and 
                % % the other_node with the reference line. This will give
                % % an array ref_angles, where length(ref_angles) = no. of
                % % links.
                % 3. Sort the ref_angles array.
                % 4. Find the difference of consecutive angles, for example
                % % ((2)-(1))mod(2.pi), ((3)-(2))mod(2.pi) ... ((1)-(n))mod(2.pi)
                % 5. Sanity check, sum of consecutive angles = 2.pi
                % 6. Get the maximum out of consecutive angles.
                
                
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

                
                
                
                % rad2deg(theta)
                if isKey(ref_angles,node_idx)
                    ref_angles{node_idx}=[ref_angles{node_idx},theta];
                else
                    ref_angles{node_idx}=theta;
                end

            end   
            
        end

        if ~ref_angles.isKey(node_idx)
            continue
        end

        assert(length(ref_angles{node_idx})==node_degrees(node_idx))
        
        

        % 3.
        sorted_local_ref_angles=sort(ref_angles{node_idx});
        
        % 4.
        consecutive_angles=zeros(1,length(sorted_local_ref_angles));
        for angle_idx = 2:length(sorted_local_ref_angles)
            consecutive_angles(angle_idx-1) = mod(sorted_local_ref_angles(angle_idx) ...
                -sorted_local_ref_angles(angle_idx-1),2*pi);
        end
        consecutive_angles(end) = mod(sorted_local_ref_angles(1) - ...
            sorted_local_ref_angles(end), 2*pi);
        % 5.
        
        % Taking care of floating point errors
        if node_degrees(node_idx)>1
            assert(abs(sum(consecutive_angles)-2*pi)<=0.0001); 
        elseif node_degrees(node_idx)==1
            assert(sum(consecutive_angles)==0); 
        end

        node_max_angles(node_idx)=max(consecutive_angles);

        % noting the largest link distance for the node
        node_max_linkdist(node_idx)=max_linkdist;
    end
    title(title_text, "Links: " + runs)
    if numel(node_locations(1,:))>100
        scatter(node_locations(1,:),node_locations(2,:),10,'filled','blue')
    else
        for n = 1:numel(node_locations(1,:))
       
            text(node_locations(1,n),node_locations(2,n),num2str(n))
            
        end
    end
    hold off
  
    fprintf(title_text + ": Average node degree:"+mean(node_degrees))
    
    % figure;
    % hold on;
    % histogram(node_degrees);
    % title("Degree information", title_text+"Mean: "+mean(node_degrees)+" Std: " + std(node_degrees));
    % xlabel("Distribution of node degrees")
    % ylabel("Instance count")
    % hold off;


    % figure;
    % hold on;
    % histogram(node_max_linkdist,0:0.01:0.15);
    % title("Max Linkdist", title_text+"; Mean: "+mean(node_degrees)+" Std: " + std(node_degrees));
    % xlabel("Distribution of node max link dist")
    % ylabel("Max link dist count")
    % hold off;


    % figure;
    % hold on;
    % histogram(all_link_dists,0:0.01:0.15);
    % title("All Linkdist", title_text+"; Mean: "+mean(node_degrees)+" Std: " + std(node_degrees));
    % xlabel("Distribution of all link dist")
    % ylabel("All link dist count")
    % hold off;

    figure;
    hold on;
    node_max_angles_deg=rad2deg(node_max_angles);
    % node_max_angles_deg=node_max_angles
    histogram(node_max_angles_deg,0:20:360);
    title("Angle information", title_text+"; Mean: "+mean(node_max_angles_deg)+" Std: " + std(node_max_angles_deg));
    xlabel("Max angle per node in degrees")
    ylabel("Instance count")
    hold off;
    figure;
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

