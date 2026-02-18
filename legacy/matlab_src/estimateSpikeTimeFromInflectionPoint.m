function vars = estimateSpikeTimeFromInflectionPoint(vars,spikeWaveforms,targetSpikeDist)

[vars,spikeWaveform,spikeWaveform_] = likelyInflectionPoint(vars,spikeWaveforms,targetSpikeDist);

window = -floor(vars.spikeTemplateWidth/2): floor(vars.spikeTemplateWidth/2);
spikewindow = window-floor(vars.spikeTemplateWidth/2);

vars.locs_uncorrected = vars.locs;
spikes =  vars.locs;

ipps = nan(size(spikes));

%% debug
DEBUG = 0;
if DEBUG
    figure
    ax = subplot(1,1,1); ax.NextPlot = 'add';
    plot(ax,window,spikeWaveform,'Color',[.8 .8 .8])
    plot(ax,window,spikeWaveform_,'Color',[.5 1 .5])
end

%%

for i = 1:length(spikes)

    if targetSpikeDist(i)>vars.Distance_threshold
        % Don't correct the spike if it is above the distance
        % threshold and there is another spike nearby
        spikelocation_comparison = spikes;
        spikelocation_comparison = abs(spikelocation_comparison-vars.locs(i));
        if sum(spikelocation_comparison<vars.spikeTemplateWidth) > 1
            continue
        end
    end
    
    start_idx = vars.fs/10000*20; % 50 for fs - 50k, 10 for fs - 10k
    end_idx = vars.fs/10000*6;

    detectedSpikeWaveform = spikeWaveforms(:,i);

    %if ~(isfield(vars,'field') && contains(vars.field,'EMG'))
        detectedSpikeWaveform = smooth(detectedSpikeWaveform-detectedSpikeWaveform(1),vars.fs/2000);
        detectedSpikeWaveform_ = smoothAndDifferentiate(detectedSpikeWaveform,vars.fs/2000);
    %else
        detectedSpikeWaveform_ = Differentiate(detectedSpikeWaveform,vars.fs/4000);
    %end
    
    % normalize
    detectedSpikeWaveform = (detectedSpikeWaveform-min(detectedSpikeWaveform(start_idx+1:end-end_idx)))/diff([min(detectedSpikeWaveform(start_idx+1:end-end_idx)) max(detectedSpikeWaveform(start_idx+1:end-end_idx))]);
    detectedSpikeWaveform_ = (detectedSpikeWaveform_-min(detectedSpikeWaveform_(start_idx+1:end-end_idx)))/diff([min(detectedSpikeWaveform_(start_idx+1:end-end_idx)) max(detectedSpikeWaveform_(start_idx+1:end-end_idx))]);
            
    [pks,inflPntPeak] = findpeaks(detectedSpikeWaveform_(start_idx+1:end-end_idx),'MinPeakProminence',0.04*251/vars.spikeTemplateWidth);
    inflPntPeak = inflPntPeak+start_idx;
    
    if numel(inflPntPeak)>1
        inflPntPeak = inflPntPeak(abs(inflPntPeak-vars.likelyiflpntpeak)==min(abs(inflPntPeak-vars.likelyiflpntpeak)));
    end
    
    if length(inflPntPeak)==1 && inflPntPeak> vars.fs/10000*30 && inflPntPeak<length(detectedSpikeWaveform_)-end_idx
        ipps(i) = inflPntPeak;
        spikes(i) = spikes(i)+spikewindow(inflPntPeak);
    else
        % Peak of 2nd derivative is still undefined
        ipps(i) = vars.likelyiflpntpeak;
        spikes(i) = spikes(i)+spikewindow(vars.likelyiflpntpeak);
        
    end
    if DEBUG
        sp = plot(ax,window,detectedSpikeWaveform,'Color',[0 0 0]);
        dsp = plot(ax,window,detectedSpikeWaveform_,'Color',[0 .6 0]);
        spipps = plot(ax,window(ipps(i)),detectedSpikeWaveform_(ipps(i)),'marker','o','color',[1 0 0]);

            pause
        
        delete(sp)
        delete(dsp)
        delete(spipps)
    end

end

%%

[~,temp] = unique(spikes);
% duplicate indices
duplicate_idxs = setdiff(1:size(spikes(:), 1), temp);
% duplicate values
for idx = 1:length(duplicate_idxs)
    duplicate_value = spikes(duplicate_idxs(idx));
    repind = find(spikes==duplicate_value);
    for ridx = 1:length(repind)
        spikes(repind(ridx)) = spikes(repind(ridx))+ridx-1;
    end
end
if length(unique(spikes))~=length(spikes)
    warning('Still some duplicate spike values in estimateSpikeTimeFromInflectionPoint')
end

vars.locs = spikes;
vars.spikeWaveform = spikeWaveform;
vars.spikeWaveform_ = spikeWaveform_;

if DEBUG
    close(ax.Parent)
end

end