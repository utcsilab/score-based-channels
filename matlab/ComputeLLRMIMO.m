% Auxiliary function that computes full precision, exact LLR values with
% CSI knowledge
% Bitmap needs to match the symbols in 'constellation'
function llrOut = ComputeLLRMIMO(constellation, bitmap, mod_size, y, ...
    num_streams, h, noise_power, ...
    algorithm, alg_params)

% !!! A lot of these are carryover and may not work
% Only 'ml' was tested

% Initialize
llrOut = zeros(size(h, 2), mod_size, size(h, 3));

% Different algorithms
if strcmp(algorithm, 'zf-sic')
    % Get cell QR
    [q_cell, r_cell] = cellfun(@qr, num2cell(h, [1 2]), 'UniformOutput', false);
    
    % Convert to array and perform Q hermitian multiplication
    y_hat = tmult(conj(cell2mat(q_cell)), y, [1 0]);
    r_mat = cell2mat(r_cell);
    
    % Initialize hard candidates
    x_candidate = zeros(size(h, 2), 1, size(h, 3));
    
    % For each transmitted stream
    for stream_idx = size(h, 2):-1:1
        % Get r vector corresponding to past
        r_past    = r_mat(stream_idx, stream_idx+1:end, :);
        % Get r scalar corresponding to present
        r_current = r_mat(stream_idx, stream_idx, :);

        % Subtract past candidates
        y_hat_zf = y_hat(stream_idx, :, :) - tmult(r_past, x_candidate(stream_idx+1:end, :, :));
        % Squeeze - everything is now scalar
        y_hat_zf  = squeeze(y_hat_zf);
        r_current = squeeze(r_current);
        
        % For each bit, compute zero set / one set
        for bit_idx = 1:mod_size
            zeroSymbols = constellation(bitmap(bit_idx, :) == 0);
            oneSymbols  = constellation(bitmap(bit_idx, :) == 1);

            % Treat remainder as AWGN channel
            zeroDiffs = sum(exp(-abs(y_hat_zf - r_current .* zeroSymbols) .^ 2 / (2*noise_power)), 2);
            oneDiffs  = sum(exp(-abs(y_hat_zf - r_current .* oneSymbols) .^ 2 / (2*noise_power)), 2);
            % Compute LLR
            llrOut(stream_idx, bit_idx, :) = log(zeroDiffs ./ oneDiffs);
        end
        
        % Estimate current (hard) symbol, given past symbols
        hard_dist     = abs(y_hat_zf - r_current .* constellation) .^ 2;
        [~, hard_idx] = min(hard_dist, [], 2);
        
        % Fill candidate vector
        x_candidate(stream_idx, :, :) = constellation(hard_idx);
    end
elseif strcmp(algorithm, 'mmse')
    % Compute auxiliary matrices
    g = tmult(conj(h), h, [1, 0]);
    a = g + noise_power * eye(size(h, 2));
    % Manual inversion
    a_inv = zeros(size(a));
    for idx = 1:size(a, 3)
        a_inv(:, :, idx) = inv(a(:, :, idx));
    end
    y_matched = tmult(conj(h), y, [1, 0]);
    
    % Inverted symbols
    s = tmult(a_inv, y_matched);
    
    % Effective channel gains
    mu = tmult(a_inv, conj(h), [0, 1]);
    
    % Output LLR values
 
elseif strcmp(algorithm, 'sphere')
    % Instantiate sphere decoder object
    decoder = comm.SphereDecoder('Constellation', constellation.', ...
        'BitTable', bitmap.', 'InitialRadius', 'ZF Solution');
    
    % Permute
    h_reshaped = permute(h, [3 2 1]);
    y_reshaped = squeeze(permute(y, [3 1 2]));
    % Batch decode
    output_llr = decoder(y_reshaped, h_reshaped);
    % Downscale by sigma
    output_llr = output_llr / noise_power;
    
    % Reshape and permute
    llrOut = reshape(output_llr, [mod_size size(h, 3) size(h, 2)]);
    llrOut = permute(llrOut, [3, 1, 2]);
    % Negate
    llrOut = -llrOut;
elseif strcmp(algorithm, 'm-algorithm')
    % Get cell QR
    [q_cell, r_cell] = cellfun(@qr, num2cell(h, [1 2]), 'UniformOutput', false);
    
    % Convert to array and perform Q hermitian multiplication
    y_hat = tmult(conj(cell2mat(q_cell)), y, [1 0]);
    r_mat = cell2mat(r_cell);
    
    % Get a modulation object
    demod_object = comm.GeneralQAMDemodulator('Constellation', constellation, 'BitOutput', 1);
    
    % For each channel, call the M-algorithm
    for channel_idx = 1:size(h, 3)
        % Call
        [local_llr, local_bits, ~] = ...
            soma(y_hat(:, :, channel_idx), r_mat(:, :, channel_idx), ...
            alg_params, size(h, 2), 2 ^ mod_size, ...
            constellation, noise_power, demod_object, bitmap.');
        % Fill LLR
        llrOut(:, :, channel_idx) = reshape(local_llr.', mod_size, []).';
    end
elseif strcmp(algorithm, 'ml')
    if num_streams == 2
        % Inflate dimensions
        h = reshape(h, [size(h, 1), size(h, 2), 1, 1, size(h, 3)]);
        y = reshape(y, [size(y, 1), size(y, 2), 1, 1, size(y, 3)]);

        % Return canonical transmit vectors
        [s_grid_x, s_grid_y] = meshgrid(constellation, constellation);
        s_qam = cat(3, s_grid_x, s_grid_y);
        
        % Permute
        s_qam = permute(s_qam, [3, 1, 2]);
        % Inflate dimensions
        s_qam = reshape(s_qam, [size(s_qam, 1), 1, size(s_qam, 2), size(s_qam, 3)]);

        % Single-shot matrix product
        y_candidate = tmult(h, s_qam);
        % Single-shot norm and similarity
        error_norm = squeeze(sum(abs(y - y_candidate) .^ 2, [1, 2]));
        exp_sim    = exp(- 1. / noise_power * error_norm);

        % For each bit
        for bit_idx = 1:mod_size
            % Compute zero set / one set
            zeroSymbols = bitmap(bit_idx, :) == 0;
            oneSymbols  = bitmap(bit_idx, :) == 1;

            % Collapse axis on both dimensions
            llrOut(2, bit_idx, :) = log(squeeze(sum(exp_sim(zeroSymbols, :, :), [1, 2])) ./ ...
                squeeze(sum(exp_sim(oneSymbols, :, :), [1, 2])));

            llrOut(1, bit_idx, :) = log(squeeze(sum(exp_sim(:, zeroSymbols, :), [1, 2])) ./ ...
                squeeze(sum(exp_sim(:, oneSymbols, :), [1, 2])));
        end
    elseif num_streams == 4
        % Inflate dimensions
        h = reshape(h, [size(h, 1), size(h, 2), 1, 1, 1, 1, size(h, 3)]);
        y = reshape(y, [size(y, 1), size(y, 2), 1, 1, 1, 1, size(y, 3)]);

        % Return canonical transmit vectors
        [s_grid_x, s_grid_y, s_grid_z, s_grid_t] = ndgrid(constellation, constellation, constellation, constellation);
        s_qam = cat(5, s_grid_x, s_grid_y, s_grid_z, s_grid_t);
        
        % Permute
        s_qam = permute(s_qam, [5, 1, 2, 3, 4]);
        % Inflate dimensions
        s_qam = reshape(s_qam, [size(s_qam, 1), 1, size(s_qam, 2), size(s_qam, 3), size(s_qam, 4), size(s_qam, 5)]);
        
        % Single-shot matrix product
        y_candidate = tmult(h, s_qam);
        % Single-shot norm and similarity
        error_norm = squeeze(sum(abs(y - y_candidate) .^ 2, [1, 2]));
        exp_sim    = exp(- 1. / noise_power * error_norm);
                
        % For each bit
        for bit_idx = 1:mod_size
            % Compute zero set / one set
            zeroSymbols = bitmap(bit_idx, :) == 0;
            oneSymbols  = bitmap(bit_idx, :) == 1;
            
            % Collapse axis on all dimensions
            llrOut(1, bit_idx, :) = log(squeeze(sum(exp_sim(zeroSymbols, :, :, :, :), [1, 2, 3, 4])) ./ ...
                squeeze(sum(exp_sim(oneSymbols, :, :, :, :), [1, 2, 3, 4])));

            llrOut(2, bit_idx, :) = log(squeeze(sum(exp_sim(:, zeroSymbols, :, :, :), [1, 2, 3, 4])) ./ ...
                squeeze(sum(exp_sim(:, oneSymbols, :, :, :), [1, 2, 3, 4])));

            llrOut(3, bit_idx, :) = log(squeeze(sum(exp_sim(:, :, zeroSymbols, :, :), [1, 2, 3, 4])) ./ ...
                squeeze(sum(exp_sim(:, :, oneSymbols, :, :), [1, 2, 3, 4])));

            llrOut(4, bit_idx, :) = log(squeeze(sum(exp_sim(:, :, :, zeroSymbols, :), [1, 2, 3, 4])) ./ ...
                squeeze(sum(exp_sim(:, :, :, oneSymbols, :), [1, 2, 3, 4])));

        end
    
    elseif num_streams == 8
        % Inflate dimensions
        h = reshape(h, [size(h, 1), size(h, 2), 1, 1, 1, 1, 1, 1, 1, 1, size(h, 3)]);
        y = reshape(y, [size(y, 1), size(y, 2), 1, 1, 1, 1, 1, 1, 1, 1, size(y, 3)]);

        % Return canonical transmit vectors
        [s_grid_x, s_grid_y, s_grid_z, s_grid_t, ...
            s_grid_u, s_grid_v, s_grid_w, s_grid_p] = ndgrid(...
            constellation, constellation, constellation, constellation, ...
            constellation, constellation, constellation, constellation);
        s_qam = cat(9, s_grid_x, s_grid_y, s_grid_z, s_grid_t, ...
            s_grid_u, s_grid_v, s_grid_w, s_grid_p);
        
        % Permute
        s_qam = permute(s_qam, [9, 1, 2, 3, 4, 5, 6, 7, 8]);
        % Inflate dimensions
        s_qam = reshape(s_qam, [size(s_qam, 1), 1, size(s_qam, 2), size(s_qam, 3), size(s_qam, 4), size(s_qam, 5), ...
            size(s_qam, 6), size(s_qam, 7), size(s_qam, 8), size(s_qam, 9)]);
        
        % Single-shot matrix product
        y_candidate = tmult(h, s_qam);
        % Single-shot norm and similarity
        error_norm = squeeze(sum(abs(y - y_candidate) .^ 2, [1, 2]));
        exp_sim    = exp(- 1. / noise_power * error_norm);
                
        % For each bit
        for bit_idx = 1:mod_size
            % Compute zero set / one set
            zeroSymbols = bitmap(bit_idx, :) == 0;
            oneSymbols  = bitmap(bit_idx, :) == 1;
            
            % Collapse axis on all dimensions
            llrOut(1, bit_idx, :) = log(squeeze(sum(exp_sim(zeroSymbols, :, :, :, :, :, :, :, :), [1, 2, 3, 4, 5, 6, 7, 8])) ./ ...
                squeeze(sum(exp_sim(oneSymbols, :, :, :, :, :, :, :, :), [1, 2, 3, 4, 5, 6, 7, 8])));

            llrOut(2, bit_idx, :) = log(squeeze(sum(exp_sim(:, zeroSymbols, :, :, :, :, :, :, :), [1, 2, 3, 4, 5, 6, 7, 8])) ./ ...
                squeeze(sum(exp_sim(:, oneSymbols, :, :, :, :, :, :, :), [1, 2, 3, 4, 5, 6, 7, 8])));

            llrOut(3, bit_idx, :) = log(squeeze(sum(exp_sim(:, :, zeroSymbols, :, :, :, :, :, :), [1, 2, 3, 4, 5, 6, 7, 8])) ./ ...
                squeeze(sum(exp_sim(:, :, oneSymbols, :, :, :, :, :, :), [1, 2, 3, 4, 5, 6, 7, 8])));

            llrOut(4, bit_idx, :) = log(squeeze(sum(exp_sim(:, :, :, zeroSymbols, :, :, :, :, :), [1, 2, 3, 4, 5, 6, 7, 8])) ./ ...
                squeeze(sum(exp_sim(:, :, :, oneSymbols, :, :, :, :, :), [1, 2, 3, 4, 5, 6, 7, 8])));
            
            llrOut(5, bit_idx, :) = log(squeeze(sum(exp_sim(:, :, :, :, zeroSymbols, :, :, :, :), [1, 2, 3, 4, 5, 6, 7, 8])) ./ ...
                squeeze(sum(exp_sim(:, :, :, :, oneSymbols, :, :, :, :), [1, 2, 3, 4, 5, 6, 7, 8])));

            llrOut(6, bit_idx, :) = log(squeeze(sum(exp_sim(:, :, :, :, :, zeroSymbols, :, :, :), [1, 2, 3, 4, 5, 6, 7, 8])) ./ ...
                squeeze(sum(exp_sim(:, :, :, :, :, oneSymbols, :, :, :), [1, 2, 3, 4, 5, 6, 7, 8])));

            llrOut(7, bit_idx, :) = log(squeeze(sum(exp_sim(:, :, :, :, :, :, zeroSymbols, :, :), [1, 2, 3, 4, 5, 6, 7, 8])) ./ ...
                squeeze(sum(exp_sim(:, :, :, :, :, :, oneSymbols, :, :), [1, 2, 3, 4, 5, 6, 7, 8])));

            llrOut(8, bit_idx, :) = log(squeeze(sum(exp_sim(:, :, :, :, :, :, :, zeroSymbols, :), [1, 2, 3, 4, 5, 6, 7, 8])) ./ ...
                squeeze(sum(exp_sim(:, :, :, :, :, :, :, oneSymbols, :), [1, 2, 3, 4, 5, 6, 7, 8])));

        end
    end
    
end

end