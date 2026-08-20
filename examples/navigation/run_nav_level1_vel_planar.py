"""Two-level curriculum for the PLANAR velocity-command variant
(small_yaw_vel_planar): the depthnav analog of diffaero's pmv_planar --
same shape as run_nav_level1_vel.py with the *_vel_planar configs (vz
forced to zero by the activation, spawn altitude pinned to the target's,
gravity-z randomization off).
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
        "level0_vel_planar": (False, "configs/box_2", args.level0_iters),
        "level1_vel_planar": (True, "configs/level_1", args.level1_iters),
    }
    base_config_files = [
        "examples/navigation/train_cfg/nav_empty_vel_planar.yaml",
        "examples/navigation/train_cfg/nav_levelX_vel_planar.yaml",
    ]
    run_experiment(
        script="depthnav/scripts/train_bptt.py",
        experiment_dir="examples/navigation/logs/level1_vel_planar",
        config_keys=config_keys,
        run_params=run_params,
        base_config_files=base_config_files,
        policy_config_file="examples/navigation/policy_cfg/small_yaw_vel_planar.yaml",
        eval_configs=[
            "examples/navigation/eval_cfg/nav_level1_vel_planar.yaml",
        ],
        eval_csvs=[
            "examples/navigation/logs/level1_vel_planar/nav_level_1_vel_planar.csv",
        ],
        curriculum=True,
        max_retries=5,
    )
