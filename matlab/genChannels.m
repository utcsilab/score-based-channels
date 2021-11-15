function [cdl, output_h] = genChannels(num_channels, Nt, Nr, hparams, ...
    global_seed)

% Base channel
cdl = nrCDLChannel;
cdl.DelayProfile        = hparams.DelayProfile;
cdl.DelaySpread         = hparams.DelaySpread;
cdl.CarrierFrequency    = hparams.CarrierFrequency;
cdl.MaximumDopplerShift = hparams.MaximumDopplerShift;
cdl.SampleRate          = hparams.SampleRate;

% Array
cdl.TransmitAntennaArray.Size           = [Nt, 1, 1, 1, 1];
cdl.TransmitAntennaArray.ElementSpacing = [hparams.TxSpacing, 1, 1, 1];
cdl.ReceiveAntennaArray.Size            = [Nr, 1, 1, 1, 1];
cdl.ReceiveAntennaArray.ElementSpacing  = [hparams.RxSpacing, 1, 1, 1];

% Generate a fake input
T = cdl.SampleRate * 1e-3;
x = complex(randn(T, Nt), randn(T, Nt));
% Selected subcarrier and symbol range
sc_range = 1:hparams.selected_gap:...
    ((hparams.selected_sc-1)*hparams.selected_gap+1);
sym_range = 1:hparams.selected_sc; % Make sure no overflow here

% Outputs
output_h = zeros(num_channels, hparams.selected_sc, Nr, Nt);

% Progress
progressbar(0);

% For each realization
for idx = 1:num_channels
    % Force seed
    release(cdl);
    cdl.Seed = global_seed * (num_channels + idx);
    reset(cdl);
    
    % Pass through channel and get subframe info
    [~, pathGains, sampleTimes] = cdl(x); % Note that output is not used
    pathFilters                 = getPathFilters(cdl);
    offset = nrPerfectTimingEstimate(pathGains, pathFilters);
    
    % Perfect CSI
    NRB   = 25;
    SCS   = 15;
    nSlot = 0;
    local_h = nrPerfectChannelEstimate(pathGains, pathFilters, ...
        NRB, SCS, nSlot, offset, sampleTimes);
    
    % Store in collection
    % Designated subcarrier in each designated symbol
    for sub_idx = 1:hparams.selected_sc
        output_h(idx, sub_idx, :, :) = ...
            squeeze(local_h(sc_range(sub_idx), sym_range(sub_idx), :, :));
    end
    
    % Progress
    progressbar(idx / num_channels)
end

end