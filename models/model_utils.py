import torch
import torch.nn as nn


def count_params(model, trainable=True):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


class Bilinear_Interpolation(nn.Module):
    def __init__(self, scene_size=100):
        super(Bilinear_Interpolation, self).__init__()
        self.scene_size = scene_size

    def forward(self, episode_idx, sequence, feature_map, oom_val):
        """
        inputs
        episode_idx: [A]
        sequence : [A X Td X 2]
        feature_map: [B X Ce X 100 X 100]
        oom_val: padding value
        outputs
        local_featrue_bt: [A X Td X Ce]
        sequence_mapCS: [A X Td X 2]
        """
        # Detect total agents
        total_agents = sequence.size(0)
        # Detect sequence length
        seq_len = sequence.size(1)

        if feature_map.device != sequence.device:
          feature_map = feature_map.to(sequence.device)

        # Pad the feature_map with oom_val
        pad = (1, 1, 1, 1)
        feature_map_padded = F.pad(feature_map, pad, mode='constant', value=oom_val) # [A X Ce X 102 X 102]

        # Change to map CS
        sequence_mapCS = (sequence + 56.0) / 112.0 * 100.0 + 1.0

        # Merge Agents-Time dimensions
        sequence_mapCS_bt = sequence_mapCS.reshape(-1, 2) # [A*Td, 2]
        x = sequence_mapCS_bt[:, 0:1] # [A*Td, 1]
        y = sequence_mapCS_bt[:, 1:] # [A*Td, 1]

        # Qunatize x and y
        floor_mapCS_bt = torch.floor(sequence_mapCS_bt)
        ceil_mapCS_bt = torch.ceil(sequence_mapCS_bt)

        # Clamp by range [0, 101]
        floor_mapCS_bt = torch.clamp(floor_mapCS_bt, 0, 101)
        ceil_mapCS_bt = torch.clamp(ceil_mapCS_bt, 0, 101)
        x1 = floor_mapCS_bt[:, 0:1]
        y1 = floor_mapCS_bt[:, 1:]
        x2 = ceil_mapCS_bt[:, 0:1]
        y2 = ceil_mapCS_bt[:, 1:]

        # Make integers for indexing
        x1_int = x1.long().squeeze()
        x2_int = x2.long().squeeze()
        y1_int = y1.long().squeeze()
        y2_int = y2.long().squeeze()

        # Generate duplicated batch indexes for prediction length
        # batch_idx_array = [0,0,..,0,1,1,...,1,A-1,A-1,...,A-1]
        # of length (Td * A)
        batch_idx_array = episode_idx.repeat_interleave(seq_len)

        # Get the four quadrants around (x, y)
        q11 = feature_map_padded[batch_idx_array, :, y1_int, x1_int]
        q12 = feature_map_padded[batch_idx_array, :, y1_int, x2_int]
        q21 = feature_map_padded[batch_idx_array, :, y2_int, x1_int]
        q22 = feature_map_padded[batch_idx_array, :, y2_int, x2_int]

        # Perform bilinear interpolation
        local_featrue_flat = (q11 * ((x2 - x) * (y2 - y)) +
                              q21 * ((x - x1) * (y2 - y)) +
                              q12 * ((x2 - x) * (y - y1)) +
                              q22 * ((x - x1) * (y - y1))
                              ) # (A*Td) X Ce

        local_featrue_bt = local_featrue_flat.reshape((total_agents, seq_len, -1))

        return local_featrue_bt, sequence_mapCS
