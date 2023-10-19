import os
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid
from mlagents_envs import logging_util

import json

logger = logging_util.get_logger(__name__)


# Create the UnityConfigSideChannel class
class UnityConfigSideChannel(SideChannel):

    def __init__(self, output_dir) -> None:
        super().__init__(uuid.UUID("d3930bb0-2df3-4080-90f7-9801df5b4a9f"))
        self.output_dir = output_dir

    def request_configuration(self) -> None:
        """
        Requests the configuration from the Unity environment.
        """
        msg = OutgoingMessage()
        msg.write_bool(True)
        self.queue_message_to_send(msg)

    def on_message_received(self, msg: IncomingMessage) -> None:
        run_options_path = os.path.join(self.output_dir, "unity_config.json")
        try:
            with open(run_options_path, "w") as f:
                parsed_json = json.loads(msg.read_string())
                json.dump(parsed_json, f, indent=4)
        except FileNotFoundError:
            logger.warning(
                f"Unable to save unity config to {run_options_path}. Make sure the directory exists"
            )
