"""Two-level curriculum for the velocity-command variant (small_yaw_vel):
level0_vel learns to fly to the target in an empty box, level1_vel adds the
obstacle set and all loss terms -- same shape as run_nav_level1.py, with the
*_vel configs (velocity_world_frame dynamics, 0.8-1.5 m/s targets).

Iteration counts are CLI-overridable so a wrapper (e.g. superfly's
scripts/train_depthnav.py) can size the run without editing this file.
"""

import argparse

from depthnav.scripts.runner import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level0-iters", type=int, default=500)
    parser.add_argument("--level1-iters", type=int, default=20000)
    args = parser.parse_args()

    config_keys = (
        "env.scene_kwargs.load_geodesics",
        "env.scene_kwargs.path",
        "train_bptt.iterations",
    )

    # for tensorboard
    run_params = {
        "level0_vel": (False, "configs/box_2", args.level0_iters),
        "level1_vel": (True, "configs/level_1", args.level1_iters),
    }
    base_config_files = [
        "examples/navigation/train_cfg/nav_empty_vel.yaml",
        "examples/navigation/train_cfg/nav_levelX_vel.yaml",
    ]
    run_experiment(
        script="depthnav/scripts/train_bptt.py",
        experiment_dir="examples/navigation/logs/level1_vel",
        config_keys=config_keys,
        run_params=run_params,
        base_config_files=base_config_files,
        policy_config_file="examples/navigation/policy_cfg/small_yaw_vel.yaml",
        eval_configs=[
            "examples/navigation/eval_cfg/nav_level1_vel.yaml",
        ],
        eval_csvs=[
            "examples/navigation/logs/level1_vel/nav_level_1_vel.csv",
        ],
        curriculum=True,
        max_retries=5,
    )
