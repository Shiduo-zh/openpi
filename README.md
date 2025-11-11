# openpi adapted to VLABench

openpi holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

Currently, this repo contains two types of models:
- the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based diffusion vision-language-action model (VLA)
- the [π₀-FAST model](https://www.physicalintelligence.company/research/fast), an autoregressive VLA, based on the FAST action tokenizer.

For both models, we provide _base model_ checkpoints, pre-trained on 10k+ hours of robot data, and examples for using them out of the box or fine-tuning them to your own datasets.

This is an experiment: $\pi_0$ was developed for our own robots, which differ from the widely used platforms such as [ALOHA](https://tonyzhaozh.github.io/aloha/) and [DROID](https://droid-dataset.github.io/), and though we are optimistic that researchers and practitioners will be able to run creative new experiments adapting $\pi_0$ to their own platforms, we do not expect every such attempt to be successful. All this is to say: $\pi_0$ may or may not work for you, but you are welcome to try it and see!

<span style="font-size:16px"> 🚨 <span style="color:#AB4459;">**NOTICE:**</span>The repository is a fork that adapts the source repository to VLABench's training and evaluation, and is used as a submodule of VLABench. Please refer to [here](examples/vlabench/README.md) for pi0 evaluation on vlabench. For finetuning, please refer to the script `train_vlabench_primitive.sh`.</span>

## Create uv environment in your conda env
Suppose you have create env ``vlabench`` following the instrution in [VLABench](https://github.com/OpenMOSS/VLABench).
Now,
```sh
conda activate vlabench
pip install uv

GIT_LFS_SKIP_SMUDGE=1 uv sync
```
This will create a venv in openpi directory.

## Add your own config
You can diy your training config in [here](src/openpi/training/config.py), such as, create a new TrainConfig named `vlabench_test`.

## Compute norm stats
Then you should compute the corresponing data norm stats by running
```sh
uv run scripts/compute_norm_stats.py --config-name vlabench_test
``` 
This will create a norm_stats.json in assets/vlabench_test

## Train the model
After getting the norm stats, you can train your policies by:

```sh
bash train.sh vlabench_test
```
You should replace ``vlabench_test`` by the config name you create.

## Evaluate the model
After training, you will get some model checkpoints in ``checkpoints`` directory. Then, run the multi-gpu evaluation in ``vlabench`` conda env by:
```sh
bash run_eval.sh vlabench_test checkpoint_path --track xx --task xx
```
You will get the metric.json in `evaluation_results` and a figure auto drawed in that directory.
