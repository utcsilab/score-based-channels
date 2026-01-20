% clc, clear, close all

% Target channel
target_channels = {'CDL-A', 'CDL-B', 'CDL-C', 'CDL-D'};

% For each channel
for target_idx = 1:numel(target_channels)
    target_channel = target_channels{target_idx};

    % Load training data and determine scaling
    if strcmp(target_channel, 'CDL-D')
        target_file  = '../data/CDL-D_Nt64_Nr16_ULA0.5_seed1234.mat';
    else
        target_file  = sprintf('../data/%s_Nt64_Nr16_ULA0.50_seed1234.mat', target_channel);
    end
    contents     = load(target_file);
    val_H        = contents.output_h;
    scale_factor = sqrt(mean(abs(val_H) .^ 2, 'all'));
    
    % Load data
    if strcmp(target_channel, 'CDL-D')
        target_file  = '../data/CDL-D_Nt64_Nr16_ULA0.5_seed4321.mat';
    else
        target_file  = sprintf('../data/%s_Nt64_Nr16_ULA0.50_seed4321.mat', target_channel);
    end
    contents    = load(target_file);
    val_H       = contents.output_h;
    % Get first subcarrier of first symbol
    val_H       = squeeze(val_H(:, 1, :, :));
    % Conjugate transpose
    val_H       = conj(permute(val_H, [1 3 2]));
    
    % Normalize CSI power
    val_H = val_H / scale_factor;
    
    % Helper IDFT matrices
    left_idft  = conj(dftmtx(size(val_H, 2))) / size(val_H, 2);
    right_idft = conj(dftmtx(size(val_H, 3))) / size(val_H, 3);
    
    % Load measurement matrices
    target_file = '../data/measurementP_seed4321.mat';
    contents    = load(target_file);
    val_P       = contents.val_P;
    
    % Downselect samples
    num_samples = 100;
    val_H = val_H(1:num_samples, :, :);
    val_P = val_P(1:num_samples, :, :);
    
    % alpha and SNRs under test
    alpha_range = 0.6;
    snr_range   = -30:2.5:15;
    
    % EM-GM-AMP configuration
    optEM.heavy_tailed = true;
    optEM.noise_dim    = 'col';
    optEM.sig_dim      = 'col';
    optEM.maxEMiter    = 200;
    optEM.maxTol       = 1e-1;
    optEM.robust_gamp  = true;
    
    % Advanced parameters
    optEM.maxBethe = false;
    optEM.hiddenZ  = false;
    
    % GAMP base configuration
    optGAMP.nit = 10;
    
    % Outputs
    oracle_log   = zeros(numel(alpha_range), numel(snr_range), size(val_H, 1));
    complete_log = 1000 * ones(numel(alpha_range), numel(snr_range), ...
        optEM.maxEMiter, size(val_H, 1));
    best_log     = zeros(numel(alpha_range), numel(snr_range), size(val_H, 1));
    
    progressbar(0, 0);
    
    % For each alpha level
    for alpha_idx = 1:numel(alpha_range)
        % Select a number of rows from P
        num_pilots = int32(floor(size(val_P, 2) * alpha_range(alpha_idx)));
        local_P    = val_P(:, 1:num_pilots, :);
        % For each SNR level
        for snr_idx = 1:numel(snr_range)
            % Dynamic tolerance
            local_snr    = snr_range(snr_idx);
            local_noise  = 10 .^ (-local_snr / 10);
            
            % Update noise power (ideally known)
    %         optEM.SNRdB = local_snr - 10 * log10(64);
    
            % For each sample
            for sample_idx = 1:size(val_H, 1)
                % Get samples
                local_A = squeeze(local_P(sample_idx, :, :));
                local_H = squeeze(val_H(sample_idx, :, :));
    
                % Flatten x and create expanded A
                flat_H      = local_H(:);
                flat_H_freq = fft2(local_H);
                flat_H_freq = flat_H_freq(:);
                full_A      = double(kron(right_idft, local_A * left_idft));
    
                % Generate measurements and check errors
                square_Y  = double(local_A * local_H);
                flat_Y    = double(full_A * flat_H_freq);
                max_error = max(abs(square_Y(:) - flat_Y) .^ 2);
                if max_error > 1e-8
                    error('Kronecker flattening incorrect!');
                end
    
                % Add noise
                noisy_flat_Y = flat_Y + sqrt(local_noise/2) * ( ...
                    randn(size(flat_Y)) + 1i * randn(size(flat_Y)));
    
                % Solve
                [H_hat, EMfin, estHist, optEMfin, optGAMPfin] = EMGMAMP(noisy_flat_Y, full_A, optEM, optGAMP);
                % Reshape to matrix, convert to spatial domain and flat again
                H_hat = reshape(H_hat, size(local_H));
                H_hat = ifft2(H_hat);
                H_hat = H_hat(:);
    
                % Store error
                oracle_log(alpha_idx, snr_idx, sample_idx) = ...
                    sum(abs(H_hat - flat_H) .^ 2, 'all') / ...
                    sum(abs(flat_H) .^ 2, 'all');
                % Store error at all intermediate EM steps
                all_H_hat = reshape(estHist.xhat, [size(local_H), size(estHist.xhat, 2)]);
                % Extremely inefficient 2D FFT
                for sub_idx = 1:size(all_H_hat, 3)
                    all_H_hat(:, :, sub_idx) = ifft2(all_H_hat(:, :, sub_idx));
                end
                % Re-flatten
                all_H_hat = reshape(all_H_hat, [], size(all_H_hat, 3));
                % Errors
                complete_log(alpha_idx, snr_idx, 1:size(all_H_hat, 2), sample_idx) = ...
                    sum(abs(all_H_hat - flat_H) .^ 2, 1) / sum(abs(flat_H .^ 2), 'all');
                best_log(alpha_idx, snr_idx, sample_idx) = ...
                    min(complete_log(alpha_idx, snr_idx, 1:size(all_H_hat, 2), sample_idx));
                
                % Progress
                progressbar([], [], sample_idx / size(val_H, 1));
    
            end
            % Progress
            progressbar([], snr_idx / numel(snr_range), []);
    
        end
        % Progress
        progressbar(alpha_idx / numel(alpha_range), [], []);
    end
    
    progressbar(1, 1)

    % Save results to file
    filename = sprintf('emgmAMP_results_aug7/%s_nit%d.mat', target_channel, optGAMP.nit);
    save(filename, 'complete_log', 'oracle_log', 'best_log');
end