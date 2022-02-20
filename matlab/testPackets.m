%%
% 'real_channels' = ideally known channels (same at all data SNR)
% 'est_channels'  = saved outputs of any algorithm !!! per data SNR point !!!
% size(real_channels)= [Nt, Nr, Batch]
% size(est_channels) = [len(snr_range), Nt, Nr, Batch]
% 
% 'mod_size' = number of bits per symbol. Only '2' = QPSK
% supported because of ML soft bit estimation implementation
% 
% 'precoding'   = digital precoding strategy, only 'random' supported
% 'num_streams' = number of i.i.d. data streams (spatial multiplexing)
% 'num_packets' = number of codewords of size (324, 648) to simulate
% 'snr_range'   = vector of data SNR values
% 'run_ideal'   = whether to also actually run with 'real_channels'
%%
function [bler_ideal, ber_ideal, bler_est, ber_est] = ...
    testPackets(real_channels, est_channels, mod_size, num_tx, num_rx, ...
    precoding, num_streams, num_packets, snr_range, run_ideal)

% Modulation order fixed to QPSK
mod_order = 2^mod_size;
% Interleaver and bit seeds - no reason to ever change in principle
bit_seed   = 1234;
inter_seed = 1111;

% Code parameters - default is a systematic code of size (324, 648)
% Taken from IEEE 802.11n: HT LDPC matrix definitions
% You can change this according to your needs
Z = 27;
rotmatrix = ...
    [0 -1 -1 -1 0 0 -1 -1 0 -1 -1 0 1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
    22 0 -1 -1 17 -1 0 0 12 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1;
    6 -1 0 -1 10 -1 -1 -1 24 -1 0 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1;
    2 -1 -1 0 20 -1 -1 -1 25 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1;
    23 -1 -1 -1 3 -1 -1 -1 0 -1 9 11 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1;
    24 -1 23 1 17 -1 3 -1 10 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1;
    25 -1 -1 -1 8 -1 -1 -1 7 18 -1 -1 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1;
    13 24 -1 -1 0 -1 8 -1 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1;
    7 20 -1 16 22 10 -1 -1 23 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1;
    11 -1 -1 -1 19 -1 -1 -1 13 -1 3 17 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1;
    25 -1 8 -1 23 18 -1 14 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0;
    3 -1 -1 -1 16 -1 -1 2 25 5 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0];

H_code = zeros(size(rotmatrix)*Z);
Zh = diag(ones(1,Z),0);

% Convert Z to binary parity check matrix
for r=1:size(rotmatrix,1)
    for c=1:size(rotmatrix,2)
        rotidx = rotmatrix(r,c);
        if (rotidx > -1)
            Zt = circshift(Zh,[0 rotidx]);
        else
            Zt = zeros(Z);
        end
        limR = (r-1)*Z+1:r*Z;
        limC = (c-1)*Z+1:c*Z;
        H_code(limR, limC) = Zt;
    end
end

% Encoder/decoder objects
hEnc = comm.LDPCEncoder('ParityCheckMatrix', sparse(H_code));
hDec = comm.LDPCDecoder('ParityCheckMatrix', sparse(H_code), ...
    'DecisionMethod', 'Soft decision', ...
    'IterationTerminationCondition', 'Parity check satisfied', ...
    'MaximumIterationCount', 50);

% System parameters
K = size(H_code, 1); % Default = 324
N = size(H_code, 2); % Default = 648

% Number of channels used per packet (codeword) - must be integer
num_channels = floor(N / (mod_size * num_streams));

% Compute number of MIMO transmission required to fill one codeword
if mod(N, mod_size * num_streams) > 0
    % Add padding bits if division is not exact
    padded_llrs = N - ...
        floor(N / (mod_size * num_streams)) * (mod_size * num_streams);
else
    padded_llrs = 0;
end

% Generate a set of randomly sampled complex Gaussian precoders
% TODO: More types of precoding, some even use 'est_channels' itself
if strcmp(precoding, 'random')
    Pt = 1/sqrt(2) * randn(num_tx, num_streams, size(real_channels, 3)) + ...
         1/sqrt(2) * 1j * randn(num_tx, num_streams, size(real_channels, 3));
elseif strcmp(precoding, 'dft')
    error('Not yet implemented!')
elseif strcmp(precoding, 'svd')
    error('Not yet implemented!')
end

% Interleaver
rng(inter_seed);
P = randperm(N);
R(P) = 1:N;
rng(bit_seed)

% Fetch constellation and auxiliary tables for fast LLR computation
bitmap = de2bi(0:(mod_order-1)).';
constellation = qammod(bitmap, mod_order, 'InputType', 'bit', ...
    'UnitAveragePower', true);

% Performance metrics
ber_ideal  = zeros(numel(snr_range), num_packets);
ber_est    = zeros(numel(snr_range), num_packets);
bler_ideal = zeros(numel(snr_range), num_packets);
bler_est   = zeros(numel(snr_range), num_packets);

% For each SNR value
for snr_idx = 1:numel(snr_range)
    noise_power = 10 ^ (-snr_range(snr_idx)/10);
    
    % Fetch estimated channels
    local_est_H = squeeze(est_channels(snr_idx, :, :, :));    
    
    % For each packet
    for run_idx = 1:num_packets
        % Random bits
        payload_bits = randi([0 1], K, 1);
        bitsEnc      = hEnc(payload_bits);
        % Interleave bits
        bitsInt = bitsEnc(P);
        
        % Modulate bits
        x = qammod(bitsInt(1:end-padded_llrs), mod_order, ...
            'InputType', 'bit', 'UnitAveragePower', true);
        
        % Reshape to batched MIMO size
        x_mimo = reshape(x, [num_streams, 1, num_channels]);

        % Pick a random set of channel indices
        if size(local_est_H, 3) >= num_channels
            local_perm = randperm(size(local_est_H, 3));
        else
            local_perm = randi(size(local_est_H, 3), [1, num_channels]);
        end
        % Subsample a random set of channels and precoders
        perm_H_ideal = real_channels(:, :, local_perm(1:num_channels));
        perm_H_est   = local_est_H(:, :, local_perm(1:num_channels));
        perm_P       = Pt(:, :, local_perm(1:num_channels));
        
        % Form received streams with included precoders
        perm_Hp_ideal = tmult(perm_H_ideal, perm_P);
        perm_Hp_est   = tmult(perm_H_est, perm_P);
        
        % Apply precoding
        xp = tmult(perm_P, x_mimo);

        % Generate noise
        n = 1/sqrt(2) * sqrt(noise_power) * ...
            (randn(num_rx, 1, num_channels) + ...
            1i * randn(num_rx, 1, num_channels));
        % MIMO channel
        y = tmult(perm_H_ideal, xp) + n;
        
        % Estimate soft bits with ideally known channels
        if run_ideal
            llr_ideal = ComputeLLRMIMO( ...
                constellation, bitmap, mod_size, y, num_streams, ...
                perm_Hp_ideal, noise_power, 'ml', 0);
            % Flatten correctly
            llr_ideal = permute(llr_ideal, [2, 1, 3]);
            llr_ideal = llr_ideal(:);
            % Pad with erasures
            llr_ideal = [llr_ideal; zeros(padded_llrs, 1)]; %#ok<AGROW>
            % Deinterleave bits
            llr_ideal = double(llr_ideal(R));
    
            % Clip estimated LLR
            clip_value = 6; % Usually a good idea in practice
            llr_ideal(abs(llr_ideal) >= clip_value) = ...
                clip_value * sign(llr_ideal(abs(llr_ideal) >= clip_value));
            % Uncertain LLR values may sometimes be NaN due to division by zero
            llr_ideal(isnan(llr_ideal)) = 0;
        else
            llr_ideal = 0;
        end
        
        % Estimate soft bits with estimated channels
        llr_est = ComputeLLRMIMO( ...
            constellation, bitmap, mod_size, y, num_streams, ...
            perm_Hp_est, noise_power, 'ml', 0);
        % Flatten correctly
        llr_est = permute(llr_est, [2, 1, 3]);
        llr_est = llr_est(:);
        % Pad with erasures
        llr_est = [llr_est; zeros(padded_llrs, 1)]; %#ok<AGROW>
        % Deinterleave bits
        llr_est = double(llr_est(R));
        
        % Clip estimated LLR
        clip_value = 6; % Usually a good idea in practice
        llr_est(abs(llr_est) >= clip_value) = ...
            clip_value * sign(llr_est(abs(llr_est) >= clip_value));
        % Uncertain LLR values may sometimes be NaN due to division by zero
        llr_est(isnan(llr_est)) = 0;
        
        % Decode packets
        if run_ideal
            % Only if we want to
            payload_ideal = hDec(llr_ideal);
            payload_ideal = (sign(-payload_ideal) +1) / 2;
        else
            payload_ideal = zeros(size(payload_bits));
        end
        % This is the one coming from the algorithm
        payload_est   = hDec(llr_est);
        payload_est   = (sign(-payload_est) +1) / 2;
        
        % Mark block errors
        ber_ideal(snr_idx, run_idx) = ...
            mean(payload_bits ~= payload_ideal);
        bler_ideal(snr_idx, run_idx) = ...
            ~all(payload_bits == payload_ideal);
        ber_est(snr_idx, run_idx) = ...
            mean(payload_bits ~= payload_est);
        bler_est(snr_idx, run_idx) = ...
            ~all(payload_bits == payload_est);
        
        % Progress
        progressbar(double(run_idx) / double(num_packets), []);
    end
    % Progress
    progressbar([], snr_idx / numel(snr_range));
end
progressbar(1, 1)

% Average results across packets
ber_ideal  = mean(ber_ideal, 2);
ber_est    = mean(ber_est, 2);
bler_ideal = mean(bler_ideal, 2);
bler_est   = mean(bler_est, 2);

end