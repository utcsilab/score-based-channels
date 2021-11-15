clc, close, clear

% Meta-parameters
spacing_array = [1/2];
profile_array = ["CDL-B", "CDL-C", "CDL-D"];

% Channel parameters
hparams                     = struct;
hparams.DelaySpread         = 30e-9;
hparams.CarrierFrequency    = 40e9;
hparams.MaximumDopplerShift = 5;
hparams.SampleRate          = 15.36e6;

% System parameters
Nt = 64; Nr = 16;
hparams.Nt = Nt; hparams.Nr = Nr;
% Other parameters
global_seed          = 9999;
num_channels         = 200;
hparams.selected_sc  = 10;
hparams.selected_gap = 24;

% For each spacing
for local_spacing = spacing_array
    % For each profile
    for local_profile = profile_array
        % Changing parameters
        hparams.DelayProfile = local_profile;
        hparams.TxSpacing    = local_spacing;
        hparams.RxSpacing    = hparams.TxSpacing;
        
        % Channel
        [cdl, output_h] = genChannels(num_channels, Nt, Nr, hparams, global_seed);

        % Save to file
        filename = sprintf('../data_mixed/%s_Nt%d_Nr%d_ULA%.2f_seed%d.mat', ...
            hparams.DelayProfile, Nt, Nr, hparams.TxSpacing, global_seed);
        save(filename, 'output_h', 'hparams', '-v7.3');
    end
end

