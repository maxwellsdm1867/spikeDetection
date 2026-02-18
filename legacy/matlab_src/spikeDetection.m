function [trial,vars_skeleton] = spikeDetection(trial,inputToAnalyze,vars_initial,varargin)
% [trial,vars_skeleton] = spikeDetection(trial,inputToAnalyze,vars_initial,varargin)
% This function detects spikes based on the match to a template and the
% approx. size of the spike, from initial upswing to the peak.
% To use this function in non-interactive mode use:
%   [trial,~] = spikeDetection(trial,invec1,spikevars,'interact','no'); % fieldname will be 'spikes'

%% Initial vars example
%     vars_initial.fs = 10000;
%     vars_initial.spikeTemplateWidth = round(0.005*vars_initial.fs)+1;
%     vars_initial.len = 10000;
%     vars_initial.hp_cutoff = 897.6012;
%     vars_initial.lp_cutoff = 209.2370;
%     vars_initial.diff = 1;
%     vars_initial.peak_threshold = 3.7670e-04;
%     vars_initial.Distance_threshold= 11.0418;
%     vars_initial.spikeTemplate = [];
%     vars_initial.lastfilename = '';
%     vars_initial.likelyiflpntpeak = 33;
%     vars_initial.polarity = -1;
%     vars_initial.Amplitude_threshold = 0.7104;
%     
%     fstag = ['fs' num2str(vars_initial.fs)];
%     setacqpref('FlyAnalysis',['Spike_params_voltage_1_' fstag],vars_initial);
    
%%

p = inputParser;
p.addParameter('filter',[]);
p.addParameter('interact','yes',@(x) any(validatestring(x,{'yes','no'})));
p.addParameter('alt_spike_field','spikes', @(x) assert(ischar(x)));
p.addParameter('intertrial_spikes','no', @(x) any(validatestring(x,{'yes','no'})));
parse(p,varargin{:});

switch p.Results.interact
    case 'yes'
        InteractFlag = 1;
    case 'no'
        InteractFlag = 0;
end

% clean up vars_initial
vars_initial = cleanUpSpikeVarsStruct(vars_initial);
vars_initial.field = p.Results.alt_spike_field;

if InteractFlag
    fprintf('** Spike Detection running with params:\n')
    disp(vars_initial);
end

global vars;
vars = vars_initial;

if InteractFlag
    vars.trialname = trial.name;
end

switch p.Results.intertrial_spikes
    case 'yes'
        unfiltered_data = cat(2,trial.(inputToAnalyze),trial.intertrial.(inputToAnalyze));
    case 'no'
        unfiltered_data = trial.(inputToAnalyze);
end

if ~isempty(p.Results.filter)
    d1 = p.Results.filter;
    unfiltered_data = lowPassFilterMembraneVoltage(unfiltered_data,d1.SampleRate,d1);
elseif isfield(trial,'name')
    %unfiltered_data = filterMembraneVoltage(unfiltered_data,trial.params.sampratein);
end

start_point = round(.01*vars_initial.fs);
stop_point = length(unfiltered_data);
unfiltered_data = unfiltered_data(start_point+1:stop_point);
vars.len = length(unfiltered_data)-round(.01*vars_initial.fs);

vars.unfiltered_data = unfiltered_data;
vars.filtered_data = filterDataWithSpikes(vars);
if isfield(trial,'name')
    [~,vars.lastfilename] = fileparts(trial.name);
end

%% run detection first, ask if you need to look again.
if isfield(vars,'spikeTemplate') && ~isempty(vars.spikeTemplate) 
    spikeTemplate = vars.spikeTemplate;
    [spikes_detected,uncorrectedSpikes] = detectSpikes();
end

if InteractFlag
    %% if you're done, save the spikes... done
    newbutton = questdlg('Done with spike detection?','Spike detection','Yes');
    
    % rejigger the filter and then select some spikes
    while strcmp(newbutton,'No')
        % if it needs work, change the filters
        
        fprintf('** Spike filtering running with params:\n')
        disp(vars);
        
        filter_sliderGUI(vars.unfiltered_data);
        while ~waitforbuttonpress;end
        slidefig = gcf;
        slidefig.CloseRequestFcn = {@(hObject,eventdata,handles) delete(hObject)};
        close(gcf)
        
        selected_spikes = [];
        spikeTemplate = [];
        
        fprintf('** Spike filtering running with params:\n')
        disp(vars);
        
        % select suspect spikes to create a seedTemplate
        while isempty(selected_spikes)          % Wait while the user does this.
            getSeedTemplate
        end
        
        if isempty(spikeTemplate)
            detected_spike_locs = [];
            spikesBelowThresh = [];
            return;
        end
        vars.spikeTemplate = spikeTemplate;
        
        [spikes_detected,uncorrectedSpikes] = detectSpikes;
        newbutton = questdlg('Done with spike detection?','Spike detection','Yes');
    end
else
    newbutton = 'Yes';
end

% Save spikes
if strcmp(newbutton,'Yes')
    vars = cleanUpSpikeVarsStruct(vars);
    vars_skeleton = vars;
    fn = p.Results.alt_spike_field;

    if isempty(spikes_detected)
        trial.(fn) = spikes_detected;
        trial.([fn '_spikes_uncorrected']) = uncorrectedSpikes;
        trial.([fn(1:end-1) 'DetectionParams']) = vars;
        trial.([fn(1:end-1) 'SpotChecked']) = 0;
        if isfield(trial,'name')
            [~,vars.lastfilename] = fileparts(trial.name);
            save(trial.name, '-struct', 'trial');
            fprintf('Saved ''%s'' (0) and filter parameters saved: %s\n',fn,trial.name);
        else
            fprintf('Saved ''%s'' (%d) and filter parameters saved\n',fn,numel(trial.(fn)));
        end
        
        % Just close the detection figure, if it exists
        delete(findobj('type','figure','tag','distanceThreshFig'));
        return
    end
    
    % calls the field by the input field names
    trial.(fn) = spikes_detected + start_point;
    trial.([fn '_uncorrected']) = uncorrectedSpikes + start_point;
    trial.([fn(1:end-1) 'DetectionParams']) = vars;
    trial.([fn(1:end-1) 'SpotChecked']) = 0;
    if isfield(trial,'name')
        save(trial.name, '-struct', 'trial');
        fstag = ['fs' num2str(vars.fs)];
        setacqpref('FlyAnalysis',['Spike_params_' inputToAnalyze '_' fstag],vars);
        fprintf('Saved ''%s'' (%d) and filter parameters saved: %s\n',fn,numel(trial.(fn)),trial.name);
    else
        fprintf('Saved Spikes (%d) and filter parameters saved\n',numel(trial.(fn)));
    end
    if InteractFlag
        fprintf('** Spike Detection was run with params:\n')
        disp(vars);
        fprintf('**----------------------**\n')
    end
    
    return
end

if strcmp(newbutton,'Cancel')
    vars_skeleton = [];
end


    function varargout = detectSpikes
        %% get all the spike locs using the correct filt and thresh cvalues
        % Amazingly, the filtering operation in the filter_slider GUI hear
        % below were using different values for the filter poles (4 here
        % and 2 in the filterGUI). This meant that if you chose a single
        % template seed example, you would get different spikeTemplate
        % waveforms and the DTW distance was somewhat large. Oddly, you'd
        % think it would go the opposite way, such that the template was smoother 
        % than the target spikes. I'm now using 3 poles as a happy medium               

        %% First get details about the spike
        
        if vars.peak_threshold > 1E4*std(vars.filtered_data)
            vars.peak_threshold = 3*std(vars.filtered_data);
        end
        spike_locs = findSpikeLocations(vars, vars.filtered_data);
        
        if any(spike_locs~=unique(spike_locs))
            error('Why are there multiple peak at the same time?')
        end
        
        [detectedUFSpikeCandidates,...
            detectedSpikeCandidates,...
            norm_detectedSpikeCandidates,...
            targetSpikeDist,...
            spikeAmplitude,...
            window,...
            spikewindow] = ...
            getSquiggleDistanceFromTemplate(spike_locs,spikeTemplate,vars.filtered_data,vars.unfiltered_data,vars.spikeTemplateWidth,vars.fs,vars.field);
        
        vars.locs = spike_locs;


        %% Plot the trial
        % Create a figure that you can then click on to analyze spikes
        
        if InteractFlag
            
            disttreshfig = figure; clf; set(disttreshfig, 'Position', [140          80        1600         900],'color', 'w','tag','distanceThreshFig');
            disttreshfig.CloseRequestFcn = {@(hObject,eventdata,handles) disp('Hit a button')};
            disttreshfig.Name = vars.trialname;
            panl = panel(disttreshfig);
            
            vertdivisions = [2 1 4 4]; vertdivisions = num2cell(vertdivisions/sum(vertdivisions));
            panl.pack('v',vertdivisions)  % response panel, stimulus panel
            panl.margin = [20 20 10 10];
            panl.fontname = 'Arial';
            panl(1).marginbottom = 2;
            panl(2).margintop = 2;
            panl(2).marginbottom = 10;
            
            % Plot unfiltered data
            ax_main = panl(1).select(); ax_main.Tag = 'main';
            plot(ax_main,vars.unfiltered_data-mean(vars.unfiltered_data),'color',[.85 .33 .1],'tag','vars.unfiltered_data'), hold(ax_main,'on');
            axis(ax_main,'off');
            title(ax_main,regexprep(sprintf('\tWhen done moving threshold, hit ''space'''),'_','\\_'));
            
            % Plot filtered data
            ax_filtered = panl(2).select(); ax_filtered.Tag = 'filtered';
            plot(ax_filtered,vars.filtered_data-mean(vars.filtered_data),'color',[.0 .45 .74],'tag','filtered_data'), hold(ax_filtered,'on');
            axis(ax_filtered,'off');
            
            linkaxes([ax_main ax_filtered],'x');
        end
                
        % This is useful feedback to see what has been detected thus far,
        % but if there are no spikes, stop here.
        if ~isempty(targetSpikeDist)
            spikeWaveforms = detectedUFSpikeCandidates-repmat(detectedUFSpikeCandidates(1,:),size(detectedUFSpikeCandidates,1),1);

            if InteractFlag
                
                suspect = targetSpikeDist<vars.Distance_threshold & spikeAmplitude > vars.Amplitude_threshold;
                [good,weird,weirdbad,bad] = thegoodthebadandtheweird(targetSpikeDist,spikeAmplitude,vars.Distance_threshold,vars.Amplitude_threshold);
                
                panl(3).pack('h',{1/3 1/3 1/3})
                
                % Plot targetSpikeDist vs spikeAmplitude
                ax_hist = panl(3,1).select(); ax_hist.Tag = 'hist';
                title(ax_hist,'Click to change threshold'); xlabel(ax_hist,'DTW Distance');
                
                % hist_dots_out =
                plot(ax_hist,targetSpikeDist(~suspect),spikeAmplitude(~suspect),...
                    '.','color',[0.9290 0.6940 0.1250],'markersize',10,'tag','distance_hist_out'); hold(ax_hist,'on');
                % hist_dots_in =
                plot(ax_hist,targetSpikeDist(suspect),spikeAmplitude(suspect),...
                    '.','color',[.0 .45 .74],'markersize',10,'tag','distance_hist');
                ax_hist.UserData = [targetSpikeDist(:) spikeAmplitude(:) spike_locs(:)];
                hold(ax_hist,'on');
                
                % Threshold lines
                ampthreshlineY = [min(spikeAmplitude) max(spikeAmplitude)];
                if diff(ampthreshlineY)==0
                    ampthreshlineY = [min(spikeAmplitude) min(spikeAmplitude)+1];
                end
                plot(ax_hist,vars.Distance_threshold*[1 1],ampthreshlineY,'color',[1 0 0],'tag','dist_threshold');
                plot(ax_hist,[min(targetSpikeDist) max([max(targetSpikeDist) vars.Distance_threshold])],vars.Amplitude_threshold*[1 1],'color',[1 0 0],'tag','amp_threshold');
                
                % Plot good and bad detected waveforms
                ax_detect = panl(3,2).select(); ax_detect.Tag = 'detect';
                title(ax_detect,'Click anywhere to use blue line as template');
                hold(ax_detect,'on');
                
                ax_detect.UserData.window = window;
                ax_detect.UserData.spikewindow = spikewindow;
                
                %%
                
                if any(good)
                    goodSuspectSquiggles = plot(ax_detect,window,detectedSpikeCandidates(:,good),'tag','squiggles');
                    meansquiggle = plot(ax_detect,window,mean(detectedSpikeCandidates(:,good),2),'color',[.3 .6 1], 'linewidth', 1,'tag','potential_template');
                    set(goodSuspectSquiggles,'Color',[.8 .8 .8]);
                end
                
                if any(weird)
                    weirdSuspectSquiggles = plot(ax_detect,window,detectedSpikeCandidates(:,weird),'tag','weirdsquiggles');
                    set(weirdSuspectSquiggles,'Color',[0 0 0]);
                end
                template = plot(ax_detect,window,spikeTemplate,'color',[.7 .2 .3], 'linewidth', 1,'tag','initial_template');
                
                
                % Plot all detected spikes
                ax_detect_patch = panl(3,3).select(); ax_detect_patch.Tag = 'detect_patch';
                hold(ax_detect_patch,'on');
                spikeWaveform = smooth(mean(spikeWaveforms(:,suspect),2),vars.fs/2000);
                spikeWaveform_ = smoothAndDifferentiate(spikeWaveform,vars.fs/2000);
                
                if any(good) && any(weird)
                    plot(ax_detect_patch,spikewindow,spikeWaveforms(:,good),'color',[0 .8 .8],'tag','spikes');
                    plot(ax_detect_patch,spikewindow,spikeWaveforms(:,weird),'color',[0 0 0],'tag','weirdspikes');
                    smthwnd = (vars.fs/2000+1:length(spikewindow)-vars.fs/2000);
                    plot(ax_detect_patch,spikewindow,spikeWaveform,'color',[0 .3 1],'linewidth',2,'tag','goodspike');
                    plot(ax_detect_patch,spikewindow(smthwnd(2:end-1)),spikeWaveform_(smthwnd(2:end-1))/max(spikeWaveform_(smthwnd(2:end-1)))*max(spikeWaveform),'color',[0 .8 .4],'linewidth',2,'tag','spike_ddt');
                    
                    spikeTime = spikewindow(spikeWaveform_==max(spikeWaveform_));
                    spikePT = spikewindow(spikeWaveform==max(spikeWaveform));
                    spikePT = spikePT(1);
                else
                    spikeTime = 0;
                    spikePT = 0;
                end
                
                
                % Plot spikes
                suspect_ticks = raster(ax_main,spike_locs+spikePT,max(vars.unfiltered_data-mean(vars.unfiltered_data))+.02*diff([min(vars.unfiltered_data) max(vars.unfiltered_data)]));
                set(suspect_ticks,'color',[0 0 0],'linewidth',1,'tag','ticks','userdata',spikePT);
                set(suspect_ticks(~suspect),'color',[1 0 0],'linewidth',.5)
                
                % Divide detected events into spike suspects and non spike suspects
                panl(4).pack('h',{1/4 1/4 1/4 1/4});
                
                ax_fltrd_suspect = panl(4,1).select(); ax_fltrd_suspect.Tag = 'fltrd_suspect'; hold(ax_fltrd_suspect,'on');
                if any(good)
                    plot(ax_fltrd_suspect,window,detectedSpikeCandidates(:,good),'tag','squiggles_good','color',.8*[1 1 1]);
                end
                if any(weird)
                    plot(ax_fltrd_suspect,window,detectedSpikeCandidates(:,weird),'tag','squiggles_weird','color',.0*[1 1 1]);
                end
                
                ax_unfltrd_suspect = panl(4,2).select(); ax_unfltrd_suspect.Tag = 'unfltrd_suspect';  hold(ax_unfltrd_suspect,'on');
                if any(good)
                    plot(ax_unfltrd_suspect,spikewindow,spikeWaveforms(:,good),'tag','spikes_good','color',.8*[1 1 1]);
                end
                if any(weird)
                    plot(ax_unfltrd_suspect,spikewindow,spikeWaveforms(:,weird),'tag','spikes_weird','color',.0*[1 1 1]);
                end
                
                ax_fltrd_notsuspect = panl(4,3).select(); ax_fltrd_notsuspect.Tag = 'fltrd_notsuspect'; hold(ax_fltrd_notsuspect,'on')
                if any(bad)
                    plot(ax_fltrd_notsuspect,window,detectedSpikeCandidates(:,bad),'tag','squiggles_bad','color',[1 .7 .7]);
                end
                if any(weirdbad)
                    plot(ax_fltrd_notsuspect,window,detectedSpikeCandidates(:,weirdbad),'tag','squiggles_weirdbad','color',[.7 0 0]);
                end
                
                ax_unfltrd_notsuspect = panl(4,4).select(); ax_unfltrd_notsuspect.Tag = 'unfltrd_notsuspect'; hold(ax_unfltrd_notsuspect,'on')
                if any(bad)
                    plot(ax_unfltrd_notsuspect,spikewindow,spikeWaveforms(:,bad),'tag','spikes_notsuspect_','color',[1 .7 .7]);
                end
                if any(weirdbad)
                    plot(ax_unfltrd_notsuspect,spikewindow,spikeWaveforms(:,weirdbad),'tag','spikes_notsuspect_','color',[.7 0 0]);
                end
                
                if any(good)
                    uistack(meansquiggle,'top')
                end
                uistack(template,'top')
                
                %% Now update the threshold for the squiggles
                vars_0 = vars;
                %disttreshfig.CloseRequestFcn = {@(hObject,eventdata,handles) disp('Hit a button')};
                
                spikeThresholdUpdateGUI(disttreshfig,detectedSpikeCandidates,spikeWaveforms);
                %disttreshfig.CloseRequestFcn = [];
                delete(disttreshfig)
                % while ~waitforbuttonpress;end
                % uiwait();
                
                [~,...
                    ~,...
                    ~,...
                    targetSpikeDist,...
                    spikeAmplitude] = ...
                    getSquiggleDistanceFromTemplate(vars.locs,spikeTemplate,vars.filtered_data,vars.unfiltered_data,vars.spikeTemplateWidth,vars.fs,vars.field);
                
            end
            
            spikes = vars.locs;
            suspect = targetSpikeDist<vars.Distance_threshold & spikeAmplitude > vars.Amplitude_threshold; % goodspikeamp
            spikes = spikes(suspect);
                                    
            % This loop gets a corrected spike time for each spike. 
            % If the peak of the second derivative isn't useful, use the
            % average of all spikes you found to get a peak

            if length(spikes)>=1
                
                vars.locs = spikes;
                vars = estimateSpikeTimeFromInflectionPoint(vars,spikeWaveforms(:,suspect),targetSpikeDist(suspect));

                varargout = {vars.locs,vars.locs_uncorrected};
                
                return
                
            else % If there were no spikes at all
                
                spikes = [];
                spikes_uncorrected = spikes;
                varargout = {spikes,spikes_uncorrected};
                
                return
            end
        else
            % uiwait();
            spikes = [];
            spikes_uncorrected = spikes;
            varargout = {spikes,spikes_uncorrected};
        end
            
    end

    function getSeedTemplate()
        
        fprintf('** Seed template running with params:\n')
        disp(vars);
        
        fig = figure('position',[100 100 1200 900], 'NumberTitle', 'off', 'color', 'w');
        
        patchax = axes(fig,'units','normalized','position',[0.1300 0.8500 0.7750 0.1]);
        plot(patchax,(1:vars.len),vars.unfiltered_data(1:vars.len),'color',[0.8500    0.3250    0.0980]);
        ticks = raster(patchax,vars.locs,max(vars.unfiltered_data(1:vars.len))+.02*diff([max(vars.unfiltered_data(1:vars.len)),min(vars.unfiltered_data(1:vars.len))]));
        set(ticks,'tag','ticks');
        
        filtax = axes(fig,'units','normalized','position',[0.1300 0.105 0.7750 0.39]);
        set(fig,'toolbar','figure');
        
        plot_filt = (vars.filtered_data(1:vars.len)-mean(vars.filtered_data))/max(vars.filtered_data);
        plot_spikes = vars.locs(vars.locs<vars.len);
        plot_thresh = (vars.peak_threshold *std(vars.filtered_data)-mean(vars.filtered_data))/max(vars.filtered_data);
        
        hold(filtax,'off');
        axes(filtax);
        plot(filtax,plot_spikes, plot_filt(plot_spikes),'ro');hold on;
        plot(filtax,plot_filt,'k');hold on;
        plot(filtax,[1 vars.len],max(plot_filt)-[plot_thresh plot_thresh],'--','color',[.8 .8 .8]);%% uncomment to plot piezo signal or another channel
        
        squigglesax = axes(fig,'units','normalized','position',[0.1300    0.56    0.3550    0.235]); hold(squigglesax,'on');
        spikesax = axes(fig,'units','normalized','position',[0.544    0.56    0.36    0.235]); hold(spikesax,'on');
        
        cursorobj = datacursormode(fig);
        cursorobj.SnapToDataVertex = 'on'; % Snap to our plotted data, on by default
        title(filtax,'select template spikes (hold alt to select multiple), then hit enter');
        cursorobj.Enable = 'on';     % Turn on the data cursor, hold alt to select multiple points
        set(cursorobj,'UpdateFcn',{@labeldtips})
        
        fig.CloseRequestFcn = {@(hObject,eventdata,handles) disp('Hit a button')};

        while ~waitforbuttonpress;end 
        fig.CloseRequestFcn = {@(hObject,eventdata,handles) delete(hObject)};
        
        cursorobj.Enable = 'off';
        mypoints = getCursorInfo(cursorobj);
        
        if ~isempty(mypoints)
            for hh = 1:length(mypoints)
                template_center = mypoints(hh).Position(1);
                spikeTemplateSeed(hh,:) = vars.filtered_data(template_center+(-vars.spikeTemplateWidth:vars.spikeTemplateWidth));
            end
            
            selected_spikes = 0;
            if size(spikeTemplateSeed,1)>1
                % align the templates, may not have picked the peaks
                skootchedTemplate = spikeTemplateSeed;
                for r = 2:size(spikeTemplateSeed,1)
                    [c,lags] = xcorr(spikeTemplateSeed(1,:),spikeTemplateSeed(r,:));
                    skootch = lags(c==max(c));
                    switch sign(skootch)
                        case 1
                            skootchedTemplate(r,skootch+1:end) = spikeTemplateSeed(r,1:end-skootch);
                        case -1
                            skootchedTemplate(r,1:end+skootch) = spikeTemplateSeed(r,-skootch+1:end);
                    end
                end
                spikeTemplate = mean(skootchedTemplate,1);
                middle = find(spikeTemplate(vars.spikeTemplateWidth+ (-floor(vars.spikeTemplateWidth/2):floor(vars.spikeTemplateWidth/2)))==max(spikeTemplate(vars.spikeTemplateWidth+ (-floor(vars.spikeTemplateWidth/2):floor(vars.spikeTemplateWidth/2)))));
                middle = middle+floor(vars.spikeTemplateWidth/2)-1;
                spikeTemplate = spikeTemplate(middle+1+(-floor(vars.spikeTemplateWidth/2):floor(vars.spikeTemplateWidth/2)));
            else
                spikeTemplate = spikeTemplateSeed;
                middle = find(spikeTemplate(vars.spikeTemplateWidth+ (-floor(vars.spikeTemplateWidth/2):floor(vars.spikeTemplateWidth/2)))==max(spikeTemplate(vars.spikeTemplateWidth+ (-floor(vars.spikeTemplateWidth/2):floor(vars.spikeTemplateWidth/2)))));
                middle = middle+floor(vars.spikeTemplateWidth/2)-1;
                spikeTemplate = spikeTemplate(middle+1+(-floor(vars.spikeTemplateWidth/2):floor(vars.spikeTemplateWidth/2)));
            end
        else
            disp('no spikes selected');
            selected_spikes = 0;
            detected_spike_locs = [];
            spikesBelowThresh = [];
            spikeTemplate = [];
            spikeTemplateSeed = [];
        end
        
        close(fig)
        
        function output_txt = labeldtips(obj,event_obj,...
                xydata,labels,xymean)
            % Display an observation's Y-data and label for a data tip
            % obj          Currently not used (empty)
            % event_obj    Handle to event object
            % xydata       Entire data matrix
            % labels       State names identifying matrix row
            % xymean       Ratio of y to x mean (avg. for all obs.)
            % output_txt   Datatip text (character vector or cell array
            %              of character vectors)
            
            pos = get(event_obj,'Position');
            x = pos(1); y = pos(2);
            output_txt = {['X: ',num2str(x,4)]};
            
            window = -floor(vars.spikeTemplateWidth/2):floor(vars.spikeTemplateWidth/2);
            squiggle = vars.filtered_data(x+window);
            spike = vars.unfiltered_data(x+window-floor(find(window==0)/2));
            plot(squigglesax,window,squiggle);
            plot(spikesax,window-floor(find(window==0)/2),spike);
            
            % The portion of the example called Explore the Graph with the Custom Data
            % Cursor sets up data cursor mode and declares this function as a callback
            % using the following code:
        end
    end
end

%% %%%%%%%%%%%%    Other code detritus    %%%%%%%%%%%%%%%%%%%%

%% estimate spike probabilities at candidate locations
% spikeProbs = zeros(size(spike_locs));
% for i=1:length(spike_locs)
%     if min(spike_locs(i)+spike_params.spikeTemplateWidth/2,length(all_filtered_data)) - max(spike_locs(i)-spike_params.spikeTemplateWidth/2,0)< spike_params.spikeTemplateWidth
%         continue
%     else
%         spikeProbs(i) = exp( -(abs(targetSpikeDist(i)-mean(targetSpikeDist))) / (2*var(targetSpikeDist)) );
%     end
% end
        
