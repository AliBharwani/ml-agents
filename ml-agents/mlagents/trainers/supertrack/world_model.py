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
        _layers = []

        # Normalize inputs if required
        # if network_settings.normalize:
        #     _layers += [nn.LayerNorm(input_size, elementwise_affine=True)]
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