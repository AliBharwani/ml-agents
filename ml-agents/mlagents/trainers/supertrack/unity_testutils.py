# Test utils for Supertrack C# code

import argparse
import json
import pdb
from mlagents.trainers.supertrack.supertrack_utils import NUM_BONES, NUM_T_BONES, CharState, PDTargets, SuperTrackDataField, SupertrackUtils
import torch


import pytorch3d.transforms as pyt

def generate_local_testdata(read_from_existing=False):
    if read_from_existing:
        raise NotImplementedError() 
        with open("local_test.json", "r") as json_file:
            test_data = json.load(json_file)
        
        cur_pos = torch.tensor(test_data["inputs"]["cur_pos"])
        cur_rots = torch.tensor(test_data["inputs"]["cur_rots"])
        cur_vels = torch.tensor(test_data["inputs"]["cur_vels"])
        cur_rvels = torch.tensor(test_data["inputs"]["cur_rvels"])
    else:
        cur_pos = torch.rand(NUM_T_BONES, 3)
        cur_rots = pyt.random_quaternions(NUM_T_BONES)
        cur_vels = torch.rand(NUM_T_BONES, 3)
        cur_rvels = torch.rand(NUM_T_BONES, 3)
    inputs = [cur_pos, cur_rots, cur_vels, cur_rvels]
    clones = [torch.clone(t) for t in inputs]
    # Assume 'local' is your function that generates the output tensors
    output = SupertrackUtils.local(cur_pos, cur_rots, cur_vels, cur_rvels, include_quat_rots=True, unzip_to_batchsize=False)
    # check that local is in place by asserting tensors are the same
    for input, clone in zip(inputs, clones):
        assert torch.allclose(input, clone)
    
    local_pos, local_rots_6d, local_vels, local_rot_vels, heights, local_up_dir, local_quat_rots = output

    def process_tensor(t):
        return torch.flatten(t).tolist()
    test_data = {
        "inputs": {
            "cur_pos": process_tensor(cur_pos),
            "cur_rots": process_tensor(cur_rots),
            "cur_vels": process_tensor(cur_vels),
            "cur_rvels": process_tensor(cur_rvels),
        },
        "outputs": {
            "local_pos": process_tensor(local_pos),
            "local_rots_6d": process_tensor(local_rots_6d),
            "local_vels": process_tensor(local_vels),
            "local_rot_vels": process_tensor(local_rot_vels),
            "heights": process_tensor(heights),
            "local_up_dir": process_tensor(local_up_dir),
            "local_quat_rots": process_tensor(local_quat_rots),
        }
    }

    # Write the test data to a JSON file
    with open("local_test.json", "w") as json_file:
        json.dump(test_data, json_file, indent=4)


def _create_fake_charstate():
    positions = torch.rand(NUM_BONES, 3)
    rotations = pyt.random_quaternions(NUM_BONES)
    velocities = torch.rand(NUM_BONES, 3)
    rot_velocities = torch.rand(NUM_BONES, 3)
    heights = torch.rand(NUM_BONES)
    up_dir = torch.rand(3)
    return CharState(positions,
                        rotations,
                        velocities,
                        rot_velocities, 
                        heights,
                        up_dir)

def _create_fake_pdtargets():    
    rotations = pyt.random_quaternions(NUM_BONES)
    rot_velocities = torch.rand(NUM_BONES, 3)
    return PDTargets(rotations, rot_velocities)

def _create_fake_SuperTrackDataField():
    return SuperTrackDataField(_create_fake_charstate(),_create_fake_charstate(), _create_fake_pdtargets(), _create_fake_pdtargets())

def generate_policy_testdata():
    st_data = _create_fake_SuperTrackDataField()
    policy_output, global_drift = SupertrackUtils.process_raw_observations_to_policy_input([st_data], True)
    result = torch.cat([t.squeeze(0) for t in [*policy_output, global_drift]], dim=0)
    test_data = {
        "inputs": st_data.to_json(),
        "outputs": {
            "result": torch.flatten(result).tolist(),
        }
    }

    # Write the test data to a JSON file
    with open("process_supertrack_obs_test.json", "w") as json_file:
        json.dump(test_data, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_from_existing", action="store_true", help="Read data from existing JSON file")
    parser.add_argument("--local", action="store_true", help="Read data from existing JSON file")
    parser.add_argument("--policy", action="store_true", help="Read data from existing JSON file")
    args = parser.parse_args()
    if args.local:
        generate_local_testdata(read_from_existing=args.read_from_existing)
    if args.policy:
        generate_policy_testdata()
