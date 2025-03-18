from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import argparse
import numpy as np

def load_dataset(name="joey/vlabench_base"):
    dataset = LeRobotDataset(
        repo_id=name,
        local_files_only=True
    )
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="joey/vlabench_base")
    parser.add_argument("--n-step", default=10, type=int)
    args = parser.parse_args()
    
    dataset = load_dataset(args.repo_id)
    print("Dataset Size:", len(dataset))
    state = []
    action = []
    for i in range(args.n_step):
        example_data = dataset.__getitem__(i)
        state.append(example_data['ee_state'].numpy())
        action.append(example_data['actions'].numpy())
    for key, value in example_data.items():
        print(key, ":", value.shape)
    state = np.array(state)
    action = np.array(action)
    print(action[:-1] - state[1:])