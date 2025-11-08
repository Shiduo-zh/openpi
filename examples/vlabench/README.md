```bash
# Create virtual environment
uv venv --python 3.10 examples/vlabench/.venv
source examples/vlabench/.venv/bin/activate
uv pip sync examples/vlabench/requirements.txt
uv pip install -e packages/openpi-client
uv pip install -e /remote-home1/sdzhang/project/VLABench
export PYTHONPATH=$PYTHONPATH:/remote-home1/sdzhang/project/VLABench

# Run the simulation
python examples/vlabench/eval.py 
```

```bash
uv run scripts/serve_policy.py --env VLABENCH policy:checkpoint --policy.config=pi0_fast_vlabench_lora --policy.dir=checkpoints/pi0_fast_vlabench_lora/pi0_fast_lora_vlabench_primitive/29999
```

An example is 
```bash
python examples/vlabench/eval.py --args.port 8000 --args.eval_track track_1_in_distribution --args.n-episode 50 --args.save_dir /inspire/hdd/global_user/gongjingjing-25039/ytchen/openpi/evaluate_results --args.visulization --args.tasks "add_condiment insert_flower select_book select_drink select_chemistry_tube select_mahjong select_toy select_fruit select_painting select_poker"
```