import collections
import dataclasses
import logging
import math
import pathlib
import json
import imageio
import numpy as np
import os
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
# install vlabench in venv first
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler
from VLABench.envs import load_env
from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.base import Policy 
from VLABench.robots import *
from VLABench.tasks import *

VLABENCH_DUMMY_ACTION = [0.0] * 6 + [0.04, 0.04]
VLABENCH_ENV_RESOLUTION = 480  # resolution used to render training data

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    replan_steps: int = 5

    #################################################################################################################
    # VLABench environment-specific parameters
    #################################################################################################################
    tasks: str = None
    eval_track: str = None
    n_episode: int = 50  # Number of rollouts per task
    intention_score_threshold: float=0.2
    #################################################################################################################
    # Utils
    #################################################################################################################
    metrics: str = "success_rate intention_score progress_score"
    save_dir: str = None  # Path to save videos
    visulization: bool = True
    seed: int = 7  # Random Seed (for reproducibility

class Pi0(Policy):
    def __init__(self, client, replan_steps=5):
        self.model = client
        self.replan_steps=replan_steps
        self.action_plan = collections.deque(maxlen=replan_steps)
    
    def reset(self):
        self.action_plan.clear()
    
    def predict(self, obs, **kwargs):
        if len(self.action_plan) == 0:
            second_image, _, image, image_wrist = obs["rgb"]
            state = obs["ee_state"]
            last_action = obs["last_action"].copy()
            pos, quat, gripper_state = state[:3], state[3:7], state[-1]
            ee_euler = quaternion_to_euler(quat)
            pos -= np.array([0, -0.4, 0.78])
            state = np.concatenate([pos, ee_euler, np.array(gripper_state).reshape(-1)])
            # last_action = np.concatenate([last_action, np.array(gripper_state).reshape(-1)])
            policy_input = {
                "observation/image": image,
                "observation/second_image": second_image,
                "observation/wrist_image": image_wrist,
                "observation/state": state,
                "prompt": obs["instruction"]
            }
            action_chunk = self.model.infer(policy_input)["actions"]
            assert (
                len(action_chunk) >= self.replan_steps
            ), f"We want to replan every {self.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
            self.action_plan.extend(action_chunk[: self.replan_steps])
        action = self.action_plan.popleft()
        target_pos, target_euler, gripper = action[:3], action[3:6], action[-1]
        if gripper >= 0.1:
            gripper_state = np.ones(2)*0.04
        else:
            gripper_state = np.zeros(2)
        target_pos = target_pos.copy()
        target_pos += np.array([0, -0.4, 0.78])
        return target_pos, target_euler, gripper_state
    
    @property
    def name(self):
        return "pi0"

def main(args:Args) -> None:
    save_dir = args.save_dir
    assert save_dir is not None
    if args.eval_track is not None:
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/evaluation/tracks", f"{args.eval_track}.json"), "r") as f:
            episode_configs = json.load(f)
            tasks = list(episode_configs.keys())
        save_dir = os.path.join(save_dir, args.eval_track)
        os.makedirs(save_dir, exist_ok=True)
    if args.tasks is not None:
        tasks = args.tasks.split(" ")
    metrics = args.metrics.split(" ")
    assert isinstance(tasks, list)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    policy = Pi0(
        client=client,
        replan_steps=args.replan_steps
    )
    
    evaluator = Evaluator(
        tasks=tasks,
        n_episodes=args.n_episode,
        episode_config=episode_configs,
        max_substeps=1,   
        save_dir=save_dir,
        visulization=args.visulization,
        metrics=metrics,
        # intention_score_threshold=args.intention_score_threshold
    )
    

    evaluator.evaluate(policy)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)