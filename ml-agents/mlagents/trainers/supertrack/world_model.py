from typing import Any, Dict, List, Optional, Tuple, Union, cast

from mlagents.trainers.settings import NetworkSettings
from mlagents.torch_utils import torch, nn, default_device
from mlagents.trainers.supertrack.supertrack_utils import NUM_T_BONES
from mlagents.trainers.torch_entities.encoders import Normalizer, VectorInput
from mlagents.trainers.torch_entities.layers import Initialization, LinearEncoder, linear_layer


# NORMALIZATION_SIZE = pos + vel = NUM_T_BONES * (3 + 3) = 16 * 6 = 96
NORMALIZATION_SIZE = 96 
class WorldModelNetwork(nn.Module):

    def __init__(
            self,
            network_settings: NetworkSettings,
    ):
        super().__init__()
        self.network_settings = network_settings
        h_size = network_settings.hidden_units
        input_size = network_settings.input_size 
        output_size = network_settings.output_size
        _layers = []

        # Normalize inputs if required
        if network_settings.normalize:
            self.normalizer = Normalizer(NORMALIZATION_SIZE)

        _layers += [LinearEncoder(
            input_size,
            self.network_settings.num_layers - 1,
            h_size,
            Initialization.KaimingHeNormal,
            1,
            network_settings.activation_function)]        

        _layers += [linear_layer(h_size, output_size)]

        self.layers = nn.Sequential(*_layers)

    # def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    #     return self.layers(inputs)

    def forward(self, local_pos : torch.Tensor, # [batch_size, NUM_T_BONES * 3]
                local_rots_6d: torch.Tensor,    # [batch_size, NUM_T_BONES * 6] 
                local_vels: torch.Tensor,       # [batch_size, NUM_T_BONES * 3]
                local_rot_vels: torch.Tensor,   # [batch_size, NUM_T_BONES * 3]
                local_up_dir: torch.Tensor,     # [batch_size, 3]
                kin_rot_t: torch.Tensor,        # [batch_size, NUM_T_BONES * 6]
                kin_rvel_t: torch.Tensor,       # [batch_size, NUM_T_BONES * 3]
                update_normalizer: bool = False,
    ) -> torch.Tensor:
        if self.network_settings.normalize:
            normalizable_inputs = torch.cat((local_pos, local_vels), dim=-1)
            if update_normalizer:
                self.normalizer.update(normalizable_inputs)
            normalizable_inputs = self.normalizer(normalizable_inputs)
            inputs = torch.cat((normalizable_inputs,
                local_rots_6d,
                local_rot_vels, 
                local_up_dir,   
                kin_rot_t,        
                kin_rvel_t), dim=-1
            )
        else:
            inputs = torch.cat((local_pos,
                local_rots_6d,
                local_vels,
                local_rot_vels, 
                local_up_dir,   
                kin_rot_t,        
                kin_rvel_t), dim=-1
            )
        return self.layers(inputs)
    
    def export(self, output_filepath: str): 
        """
        Exports self to .onnx format

        :param output_filepath: file path to output the model (without file suffix)
        """
        onnx_output_path = f"{output_filepath}.onnx"
        dummy_pos = torch.randn(1, NUM_T_BONES * 3,  device=default_device())
        dummy_rots = torch.randn(1, NUM_T_BONES * 6,  device=default_device())
        dummy_vels = torch.randn(1, NUM_T_BONES * 3,  device=default_device())
        dummy_rvels = torch.randn(1, NUM_T_BONES * 3,  device=default_device())
        dummy_up_dir = torch.randn(1, 3,  device=default_device())
        dummy_kin_rot_t = torch.randn(1, NUM_T_BONES * 6,  device=default_device())
        dummy_kin_rvel_t = torch.randn(1, NUM_T_BONES * 3,  device=default_device())
        dummy_input = (dummy_pos, dummy_rots, dummy_vels, dummy_rvels, dummy_up_dir, dummy_kin_rot_t, dummy_kin_rvel_t)
        # dummy_input = torch.randn(1, self.network_settings.input_size, device=default_device())

        self.eval()
        torch.onnx.export(
            self,
            dummy_input,
            onnx_output_path,
            # input_names=["Sim character state and PD targets"],
            output_names=["Local accels & rot accels"]
        )
        return onnx_output_path
