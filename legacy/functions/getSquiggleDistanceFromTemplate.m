function varargout = getSquiggleDistanceFromTemplate(spike_locs,spikeTemplate,fd,ufd,stw,fs,varargin)

if ~isempty(varargin)
    field = varargin{1};
else 
    field = 'spikes';
end

window = -floor(stw/2): floor(stw/2);
spikewindow = window-floor(stw/2);
smthwnd = (fs/2000+1:length(spikewindow)-fs/2000);

if isempty(spike_locs)
    varargout = {...
        [],...
        [],...
        [],...
        [],...
        [],...
        window,...
        spikewindow};
    return
end

% pool the detected spike candidates and do spike_params.spiketemplate matching
targetSpikeDist = zeros(size(spike_locs(:)));
norm_spikeTemplate = (spikeTemplate-min(spikeTemplate))/(max(spikeTemplate)-min(spikeTemplate));

detectedUFSpikeCandidates = nan(size(window(:),1),size(spike_locs(:),1));
detectedSpikeCandidates = detectedUFSpikeCandidates;
norm_detectedSpikeCandidates = detectedUFSpikeCandidates;

for i=1:length(spike_locs)
    % in the case of a single location, the template doesn't match
    % the one coming out of seed template matching
    if min(spike_locs(i)+stw/2,length(fd)) - max(spike_locs(i)-stw/2,0)< stw
        continue
    else
        curSpikeTarget = fd(spike_locs(i)+window);
        detectedUFSpikeCandidates(:,i) = ufd(spike_locs(i)+spikewindow); % all_filtered_data(max(spike_locs(i)-floor(spike_params.spikeTemplateWidth/2),0): min(spike_locs(i)+floor(spike_params.spikeTemplateWidth/2),length(all_filtered_data)));
        detectedSpikeCandidates(:,i) = curSpikeTarget; % all_filtered_data(max(spike_locs(i)-floor(spike_params.spikeTemplateWidth/2),0): min(spike_locs(i)+floor(spike_params.spikeTemplateWidth/2),length(all_filtered_data)));
        norm_curSpikeTarget = (curSpikeTarget-min(curSpikeTarget))/(max(curSpikeTarget)-min(curSpikeTarget));
        norm_detectedSpikeCandidates(:,i) = norm_curSpikeTarget;
        [targetSpikeDist(i), ~,~] = dtw_WarpingDistance(norm_curSpikeTarget, norm_spikeTemplate);
    end
end

vars.spikeTemplateWidth = stw;
vars.fs = fs;
vars.field = field;
[vars,spikeWaveform] = likelyInflectionPoint(vars,detectedUFSpikeCandidates,targetSpikeDist);
idx_f = round(stw/24);

s_hat = spikeWaveform(vars.likelyiflpntpeak:end-idx_f)- spikeWaveform(vars.likelyiflpntpeak);
s_hat = s_hat/sum(s_hat);
s_hat = s_hat(:);

spikeAmplitude = ...
    (detectedUFSpikeCandidates(vars.likelyiflpntpeak:end-idx_f,:) - ...
    repmat(detectedUFSpikeCandidates(vars.likelyiflpntpeak,:),length(vars.likelyiflpntpeak:stw-idx_f),1))' * s_hat;

if any(isnan(detectedUFSpikeCandidates(:)))
    error('some of the spikes are at the edge of the data');
end

varargout = {...
    detectedUFSpikeCandidates,...
    detectedSpikeCandidates,...
    norm_detectedSpikeCandidates,...
    targetSpikeDist,...
    spikeAmplitude,...
    window,...
    spikewindow};