clc, clear, close all

% Scenario !!! Must match estimation
channel_snr_range = -10:2.5:15;
precoding_offset  = 20; % Data transmission is dampened by this much
snr_range         = (-10:0.5:2) - precoding_offset;
num_packets       = 100000;

% Load some channels
target_channels = 'ours_known';

% Load from pre-saved files
if strcmp(target_channels, 'l1_unknown')
    target_file  = '../results_l1_saved_channels_seed9999/CDL-C_spacing0.50/l1_results_lmbda3.00e-02.mat';
    boost_factor = 1; % To match variance scaling!
elseif strcmp(target_channels, 'l1_known')
    target_file  = '../results_l1_known_SNR_saved_channels_seed9999/CDL-C_spacing0.50/l1_results_lmbda3.00e-01.mat';
    boost_factor = 1; % To match variance scaling! 
elseif strcmp(target_channels, 'ours_unknown')
    target_file = '../results_ours_saved_channels_seed9999/CDL-C_noise3.0e-04.mat';
    boost_factor = 1;
elseif strcmp(target_channels, 'ours_known')
    target_file = '../results_ours_saved_channels_SNR_known_seed9999/CDL-C_noise1.0e-03.mat';
    boost_factor = 1;
end
contents    = load(target_file);

% Fetch channels
real_channels      = double(contents.ideal_H) * boost_factor;
est_full_channels  = double(contents.est_H) * boost_factor;

% Target a specific alpha
alpha_array  = [1, 0.8, 0.6];
target_alpha = 1;
% For each data SNR, fetch channels estimated at the corresponding channel
% SNR
est_channels = zeros(size(squeeze(est_full_channels(target_alpha, :, :, :, :))));
for snr_idx = 1:numel(snr_range)
    [~, closest_idx] = min( ...
        abs(snr_range(snr_idx) + precoding_offset - channel_snr_range));
    est_channels(snr_idx, :, :, :) = ...
        squeeze(est_full_channels(target_alpha, closest_idx, :, :, :));
end

% Move channels to last
if strcmp(target_channels, 'l1_unknown') || strcmp(target_channels, 'l1_known')
    real_channels = permute(real_channels, [3, 2, 1]);
    est_channels  = permute(est_channels,  [1, 4, 3, 2]);
else
    real_channels = conj(permute(real_channels, [2, 3, 1]));
    est_channels  = conj(permute(est_channels,  [1, 3, 4, 2]));
end

% More parameters
run_ideal   = true;
mod_size    = 2;
num_streams = 4;
num_tx = size(real_channels, 2);
num_rx = size(real_channels, 1);

% Run the stuff
[bler_ideal, ber_ideal, bler_est, ber_est] = ...
    testPackets(real_channels, est_channels, mod_size, num_tx, num_rx, ...
    num_streams, num_packets, snr_range, run_ideal);

% Save to file
save(sprintf('./e2e_results/%s_alpha%.2f.mat', ...
    target_channels, alpha_array(target_alpha)), 'bler_ideal', 'ber_ideal', ...
    'bler_est', 'ber_est', 'target_alpha', 'snr_range', 'alpha_array', ...
    'precoding_offset');
