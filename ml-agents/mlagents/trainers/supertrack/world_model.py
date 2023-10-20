from typing import Any, Dict, List, Optional, Tuple, Union, cast

from mlagents.trainers.settings import NetworkSettings
from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch_entities.layers import Initialization, LinearEncoder, linear_layer



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
        # if input_size is -1:
        #     input_size = 700 # Temp override to test SAC 
        # if self.network_settings.output_size is -1:
        #     output_size = 48 # Temp override to test SAC
        # if (input_size == -1):
        #     raise Exception("SuperTrack World Model created without input_size designated in yaml file")

        _layers = []

        # Normalize inputs if required
        if network_settings.normalize:
            _layers += [nn.LayerNorm(input_size, elementwise_affine=True)]
            # _layers += [VectorInput(input_size, True)]


        _layers += [LinearEncoder(
            input_size,
            self.network_settings.num_layers - 1,
            h_size,
            Initialization.KaimingHeNormal,
            1,
            network_settings.activation_function)]        

        _layers += [linear_layer(h_size, output_size)]

        self.layers = nn.Sequential(*_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)