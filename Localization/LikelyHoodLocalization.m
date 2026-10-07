function [X_hat_tags, orig_tag_idx, orig_anchor_idx, k_est_diff, dist_errs, se_errs] ...
    =LikelyHoodLocalization(pars, X_tags_orig, X_anchors, num_freqs, anch_type)
    k_est_diff=[];
    dist_errs=[];
    se_errs=[];
    X_tags=X_tags_orig;
    X_anchors_orig=X_anchors;
    orig_tag_idx=1:pars.num_tags;
    orig_anchor_idx=pars.num_tags+1:pars.num_tags+pars.num_anchors;
    X_hat_tags = zeros(size(X_tags));
    X_all= [X_tags X_anchors];
    
    num_tags_o=pars.num_tags;
    num_tags = pars.num_tags;
    num_anchors_o = pars.num_anchors;
    num_anchors = pars.num_anchors;
    scale = pars.scale;
    lambda=pars.lambda;
    radio_range=pars.radio_range;
    lim=pars.lim;
    main_runs=pars.main_runs;
     % num_anch_per_side_o=pars.num_anch_per_side_o;

    % Measuring distances only once, not in every iteration
    pars.groundDistances0(num_tags+1:num_tags+num_anchors,:)=0;
    pars.groundDistances(num_tags+1:num_tags+num_anchors,:)=0;
    pars.groundDistances=pars.groundDistances+pars.groundDistances';
    pars.dist_ests(num_tags+1:num_tags+num_anchors,:)=0;
    pars.dist_ests_k(num_tags+1:num_tags+num_anchors,:)=0;
    pars.seedK(num_tags+1:num_tags+num_anchors,:)=0;
    pars.mainfreqPhase(num_tags+1:num_tags+num_anchors,:)=0;
    % pars.SE(num_tags+1:num_tags+num_anchors,:)=0;

    dist = pars.groundDistances0+pars.groundDistances0';
    dist=dist*100;
    phases_without_error=mod((2*pi*dist*915*1e6/(3e8*100)),pi);

    dist_hat=pars.dist_ests+pars.dist_ests';
    % dist_hat=pars.dist_ests_k+pars.dist_ests_k';
    dist_hat=dist_hat*100;

    mainFreqPhase=pars.mainfreqPhase+pars.mainfreqPhase';
    
    % SE=pars.SE+pars.SE';
    % SE_perc=prctile(SE(pars.groundDistances~=0),100);

    k_est=pars.seedK+pars.seedK';

    
    % dist_hat_wrapped=zeros(num_tags+num_anchors,num_tags+num_anchors);
    % dist_hat_wrapped = dist_hat - lambda*floor(dist_hat/lambda); 
    dist_hat_wrapped=(3e8/(2*pi*915e6))*mainFreqPhase*100;
    
    % % !!!!!!!!!!!!!!!!!!!!!!remove the line below and uncomment the line
    % % above!!!!!!!!!!!!!!!!!!!!!!!!
    % dist_hat_wrapped=(3e8/(2*pi*915e6))*phases_without_error*100;
    % k_est=floor(dist/lambda);
    % dist_hat_t=dist;
    % dist_hat_t(dist_hat==0)=0;
    % dist_hat=dist_hat_t;

    for i=1:num_tags+num_anchors
       for j=1:num_tags+num_anchors
           if i>=j
               continue
           end

            % if pars.groundDistances(i,j)~=0
            %    if SE(i,j)>SE_perc
            %     k_est(i,j)=0;
            %     dist_hat(i,j)=0;
            %     dist_hat_wrapped(i,j)=0;
            %    else
            %        % k_est(i,j)=round((dist_hat(i,j)/100-(3e8/(2*pi*915e6))*mainFreqPhase(i,j))/(3e8/(2*915e6)));
            %        % k_est(j,i)=k_est(i,j);
            %        k_real=floor(dist(i,j)/lambda);
            %        k_est_diff=[k_est_diff k_est(i,j)-k_real];
            %        dist_errs=[dist_errs abs(dist(i,j)/100-dist_hat(i,j)/100)];
            %        se_errs=[se_errs SE(i,j)];
            %    end
            % end


            if pars.groundDistances(i,j)~=0



               % k_est(i,j)=round((dist_hat(i,j)/100-(3e8/(2*pi*915e6))*mainFreqPhase(i,j))/(3e8/(2*915e6)));
               % k_est(j,i)=k_est(i,j);
               k_real=floor(dist(i,j)/lambda);
               k_est_diff=[k_est_diff k_est(i,j)-k_real];
               dist_errs=[dist_errs abs(dist(i,j)/100-dist_hat(i,j)/100)];
               local_dhat=dist_hat_wrapped(i,j)/100+k_est(i,j)*(3e8/(2*915e6));
               % dist_errs=[dist_errs abs(dist(i,j)/100-local_dhat)];
               % se_errs=[se_errs SE(i,j)];

            end

       end
    end


    
    


    for run=1:5 % runs for iterative position correction


        conf = zeros(1,num_tags);
        for tn = 1:num_tags
            cnv1 = zeros(scale*lim,scale*lim);
            tn_converted=orig_tag_idx((tn));

            if pars.all_circles==1 %all circles
                
                for k=0:min(40,ceil(radio_range/lambda))
                    anchor_dist_hat=dist_hat(:, orig_anchor_idx);
                    anchor_dist_hat_wrapped=dist_hat_wrapped(:, orig_anchor_idx);
                    % anchor_k_est=k_est(:,orig_anchor_idx);
                    an_nos=find(anchor_dist_hat(tn_converted, :)~=0);
                    cnv1 = cnv1 + rgb2gray(insertShape(zeros(scale*lim,scale*lim),'circle',scale*[X_anchors(1,an_nos)',X_anchors(2,an_nos)',anchor_dist_hat_wrapped(tn_converted,an_nos)'+k*lambda],'color','red','LineWidth',2));
                end
            else
            
                anchor_dist_hat=dist_hat(:, orig_anchor_idx);
                anchor_dist_hat_wrapped=dist_hat_wrapped(:, orig_anchor_idx);
                anchor_k_est=k_est(:,orig_anchor_idx);
                an_nos=find(anchor_dist_hat(tn_converted, :)~=0); % minimum circles are set to 3
                
                cnv1 = cnv1 + rgb2gray(insertShape(zeros(scale*lim,scale*lim),'circle',scale*[X_anchors(1,an_nos)',X_anchors(2,an_nos)',anchor_dist_hat_wrapped(tn_converted,an_nos)'+anchor_k_est(tn_converted,an_nos)'*lambda],'color','red','LineWidth',2));
                cnv1 = cnv1 + rgb2gray(insertShape(zeros(scale*lim,scale*lim),'circle',scale*[X_anchors(1,an_nos)',X_anchors(2,an_nos)',anchor_dist_hat_wrapped(tn_converted,an_nos)'+max((anchor_k_est(tn_converted,an_nos)'-1),0)*lambda],'color','red','LineWidth',1));
                cnv1 = cnv1 + rgb2gray(insertShape(zeros(scale*lim,scale*lim),'circle',scale*[X_anchors(1,an_nos)',X_anchors(2,an_nos)',anchor_dist_hat_wrapped(tn_converted,an_nos)'+(anchor_k_est(tn_converted,an_nos)'+1)*lambda],'color','red','LineWidth',1));
                
                % For tags with circles set to be greater than 3 make two
                % more.
                an5=an_nos(find(pars.circle_ind(tn_converted, orig_anchor_idx(an_nos))>=5));
                cnv1 = cnv1 + rgb2gray(insertShape(zeros(scale*lim,scale*lim),'circle',scale*[X_anchors(1,an5)',X_anchors(2,an5)',anchor_dist_hat_wrapped(tn_converted,an5)'+max((anchor_k_est(tn_converted,an5)'-2),0)*lambda],'color','red','LineWidth',1));
                cnv1 = cnv1 + rgb2gray(insertShape(zeros(scale*lim,scale*lim),'circle',scale*[X_anchors(1,an5)',X_anchors(2,an5)',anchor_dist_hat_wrapped(tn_converted,an5)'+(anchor_k_est(tn_converted,an5)'+2)*lambda],'color','red','LineWidth',1));
                
                % For tags with circles set to be greater than 5 make two
                % more.
                an7=an_nos(find(pars.circle_ind(tn_converted, orig_anchor_idx(an_nos))>=7));
                cnv1 = cnv1 + rgb2gray(insertShape(zeros(scale*lim,scale*lim),'circle',scale*[X_anchors(1,an7)',X_anchors(2,an7)',anchor_dist_hat_wrapped(tn_converted,an7)'+max((anchor_k_est(tn_converted,an7)'-3),0)*lambda],'color','red','LineWidth',1));
                cnv1 = cnv1 + rgb2gray(insertShape(zeros(scale*lim,scale*lim),'circle',scale*[X_anchors(1,an7)',X_anchors(2,an7)',anchor_dist_hat_wrapped(tn_converted,an7)'+(anchor_k_est(tn_converted,an7)'+3)*lambda],'color','red','LineWidth',1));
                
            
            end

            % crd=floor(X_tags(:,tn_converted)*scale);
            % cnv2=cnv1;
            % cnv2(crd(2)-5:crd(2)+5,crd(1)-5:crd(1)+5)=1;

            % I = cnv1;
            % % I = imfilter(I, fspecial('disk',15), 'symmetric');
            % I = insertShape(I,'FilledCircle',scale*[X_tags(1,tn_converted)',X_tags(2,tn_converted)',5*ones(1,1)],'color','yellow');
            % I = insertShape(I,'FilledCircle',scale*[X_anchors(1,:)',X_anchors(2,:)',5*ones(num_anchors,1)],'color','cyan');
            % % I = insertShape(I,'circle',scale*[ancs(1,:)',ancs(2,:)',dist(1,:)'],'color','red');
            % hold on
            % axis off
            % ax = gca;
            % ax.YDir = 'normal';
            % imshow(I)
            % hold off
            
            
            
            

            cnv = cnv1;
            cnv = imfilter(cnv, fspecial('disk',15), 'symmetric');
            % cnv(:,1:50)=0;
            % cnv(:,1450:1500)=0;
            % cnv(1:50,:)=0;
            % cnv(1450:1500,:)=0;
            cnv = cnv./max(max(cnv));
            m=max(max(cnv)); [y,x]=find(cnv==m);
            x = mean(x);
            y = mean(y);
            if (isnan(x) || isnan(y))
                x=0;
                y=0;
            end
            cnv2 = rgb2gray(insertShape(cnv,'Filledcircle',[x,y,scale*50],'color','black','LineWidth',10));
            m2=max(max(cnv2));
            conf(tn) = m/m2;

            X_hat_tags(:,tn) = [x;y]./scale;

        end

        num = floor(length(conf)/10);
        [b, idx] = maxk(conf,num);
        X_anchors = [X_anchors X_hat_tags(:,idx)];
        orig_anchor_idx = [orig_anchor_idx orig_tag_idx(idx)];
        X_tags(:,idx) = [];
        orig_tag_idx(idx)=[];
        num_anchors = num_anchors+num;
        num_tags = num_tags-num;

        temp = X_tags_orig(:,idx);
        X_tags_orig(:,idx) = [];
        X_tags_orig = [X_tags_orig temp];

        temp = X_hat_tags(:,idx);
        X_hat_tags(:,idx) = [];
        X_hat_tags = [X_hat_tags temp];

        
        if anch_type==2
            folder_name='dat_random';
        elseif anch_type==6
            folder_name='dat_sides';
        end

        if pars.all_circles==1
            folder_name=folder_name+"allCircles";
        end

        folder_name
        if pars.savefile==true
            
            dir = strcat(folder_name,'F',num2str(num_freqs),'/_',num2str(run),'_run_no_',num2str(main_runs),'/na_',num2str(num_anchors_o),'_nt_',num2str(num_tags_o));
            if ~exist(dir, 'dir')
                mkdir(dir)
            end
            save(strcat(dir,'/data_12.mat'), "X_hat_tags", "X_tags_orig", "X_anchors_orig", "pars", "orig_tag_idx","orig_anchor_idx");
        end
    end
end

