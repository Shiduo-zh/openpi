from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

if __name__ == "__main__":
    checkpoint_to_download = ["pi0_base", "pi0_fast_base"]
    for ckpt in checkpoint_to_download:
        # config = config.get_config(ckpt)
        checkpoint_dir = download.maybe_download(f"s3://openpi-assets/checkpoints/{ckpt}")
        print(f"The checkpoint directory of {ckpt} is {checkpoint_dir}")