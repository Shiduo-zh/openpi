# python examples/vlabench/eval.py --args.host=10.176.52.118 --args.tasks="add_condiment" --args.episode_config_path="/remote-home1/sdzhang/project/VLABench/track_1_new.json" --args.save_dir="data/vlabench/pi0_fast_lora/track_1"
# python examples/vlabench/eval.py --args.host=10.176.52.118 --args.tasks="select_painting" --args.episode_config_path="/remote-home1/sdzhang/project/VLABench/track_4.json" --args.save_dir="data/vlabench/pi0_fast_lora/track_4"
python examples/vlabench/eval.py --args.host=10.176.52.111 --args.tasks="texas_holdem play_math_game" --args.episode_config_path="/remote-home1/sdzhang/project/VLABench/track_7.json" --args.save_dir="data/vlabench/pi0_fast_lora/track_7" --args.n_episode=10