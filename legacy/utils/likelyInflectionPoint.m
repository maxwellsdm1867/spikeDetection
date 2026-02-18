function [vars,spikeWaveform,spikeWaveform_] = likelyInflectionPoint(vars,spikeWaveforms,targetSpikeDist)

% normalize and find a peak of the second derivative
idx_i = round(vars.spikeTemplateWidth/6);
idx_f = round(vars.spikeTemplateWidth/24);
idx_m = round(vars.spikeTemplateWidth*4/5);

window = -floor(vars.spikeTemplateWidth/2): floor(vars.spikeTemplateWidth/2);
spikewindow = window-floor(vars.spikeTemplateWidth/2);
smthwnd = (vars.fs/2000+1:length(spikewindow)-vars.fs/2000);

% Find the best estimate of the spike shape and it's 2nd derivative
goodspikes = targetSpikeDist<quantile(targetSpikeDist,.25);
if sum(goodspikes)<4
    if length(goodspikes)==1
        goodspikes = 1;
    else
        [~,o] = sort(targetSpikeDist);
        cnt = 1;
        while sum(goodspikes)<floor(length(goodspikes)/2) && sum(goodspikes)<4
            goodspikes(o(cnt)) = 1;
            cnt = cnt+1;
        end
    end
end
spikeWaveform = nanmean(spikeWaveforms(:,goodspikes),2);
spikeWaveform = spikeWaveform-min(spikeWaveform);
spikeWaveform = spikeWaveform/max(spikeWaveform);

% if ~(isfield(vars,'field') && contains(vars.field,'EMG'))
    spikeWaveform = smooth(spikeWaveform-spikeWaveform(1),round(vars.fs/4000));
    spikeWaveform_ = smoothAndDifferentiate(spikeWaveform,round(vars.fs/2000));
% else
%     spikeWaveform_ = Differentiate(spikeWaveform,vars.fs/4000);
% end
spikeWaveform_ = spikeWaveform_-spikeWaveform_(smthwnd(1));
spikeWaveform_ = (spikeWaveform_-min(spikeWaveform_(idx_i:end-idx_f)))/diff([min(spikeWaveform_(idx_i:end-idx_f)) max(spikeWaveform_(idx_i:end-idx_f))]);

% find the peak of the second derivative that is not the end point
[pks,inflPntPeak_ave] = findpeaks(spikeWaveform_(idx_i+1:end-idx_f),'MinPeakProminence',0.014*251/vars.spikeTemplateWidth);
if isempty(pks)
    expn = 1; % just progressively lower the min peak prominence
    while isempty(pks)
        [pks,inflPntPeak_ave] = findpeaks(spikeWaveform_(idx_i+1:end-idx_f),'MinPeakProminence',0.014*251/vars.spikeTemplateWidth/(2^expn));
        expn = expn+1;
    end
end
inflPntPeak_ave = inflPntPeak_ave+idx_i;

pks = pks(abs(inflPntPeak_ave-idx_m)==min(abs(inflPntPeak_ave-idx_m)));
inflPntPeak_ave = inflPntPeak_ave(abs(inflPntPeak_ave-idx_m)==min(abs(inflPntPeak_ave-idx_m)));
if numel(inflPntPeak_ave) > 1 
    % if by chance the peaks are equidistant from the estimated mid point
    inflPntPeak_ave = inflPntPeak_ave(pks==max(pks));
end
vars.likelyiflpntpeak = inflPntPeak_ave; 