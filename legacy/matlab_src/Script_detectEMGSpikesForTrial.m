% Run EMG spike detection on a trial
% assuming the trial is loaded
sgn = 1;
spikevars = getacqpref('FlyAnalysis',['Spike_params_current_2_flipped_fs', num2str(trial.params.sampratein)]);
% setacqpref('FlyAnalysis',['Spike_params_current_2_flipped_fs', num2str(trial.params.sampratein)],spikevars);
 
trial.current_2_flipped = sgn*trial.current_2; 
[trial,spikevars] = spikeDetection(trial,'current_2_flipped',spikevars,'alt_spike_field','EMGspikes');
