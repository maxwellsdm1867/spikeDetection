function spikeWaveform_ = Differentiate(spikeWaveform,smthw)

spikeWaveform_ = diff(spikeWaveform-spikeWaveform(1));
spikeWaveform_ = diff(spikeWaveform_-spikeWaveform_(1));
spikeWaveform_(1:3) = mean(spikeWaveform_(1:20));
spikeWaveform_ = smooth(spikeWaveform_-spikeWaveform_(1),max([smthw,5]));
spikeWaveform_ = spikeWaveform_-spikeWaveform_(1);
%spikeWaveform_ = [0;spikeWaveform_; spikeWaveform_(end)+diff(spikeWaveform_(end-1:end))];
spikeWaveform_ = [0;0;spikeWaveform_]; 