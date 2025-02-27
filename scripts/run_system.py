"""Run a particular SLAM system configuration."""

import pickle
from argparse import ArgumentParser
from pathlib import Path

from gtsam import Pose2
from yaml import safe_load

import ground_texture_systems

if __name__ == "__main__":
    arg_parser = ArgumentParser(
        description="Load multi-session data.",
    )
    arg_parser.add_argument("system", help="The system type.")
    arg_parser.add_argument(
        "param_file",
        help="The settings for this particular run.",
    )
    args = arg_parser.parse_args()
    with Path(args.param_file).open() as param_stream:
        params = safe_load(stream=param_stream)
    with Path("intermediate_data/measurements.pkl").open(
        mode="rb",
    ) as measurement_stream:
        measurements: list[list[ground_texture_systems.Measurement]] = (
            pickle.load(file=measurement_stream)
        )
    system = ground_texture_systems.construct_system(
        system=args.system,
        parameters=params["system"],
    )
    final_pose_estimates: list[list[Pose2]] = []
    for session_measurements in measurements:
        for measurement in session_measurements:
            if measurement.measurement_id == 0:
                pose_prior = measurement.actual_pose
                pose_prior_variance = 1e-5
            else:
                pose_prior = None
                pose_prior_variance = None
            system.insert_frame(
                image=measurement.image,
                pose=pose_prior,
                variance=pose_prior_variance,
            )
        # Get the pose estimates for the most recent session.
        final_pose_estimates.append(system.pose_estimates[-1])
        system.new_session()
    # Store the poses.
    pose_file = Path(f"intermediate_data/poses/{args.system}.pkl")
    pose_file.parent.mkdir(parents=True, exist_ok=True)
    with pose_file.open(mode="wb") as pose_stream:
        pickle.dump(obj=final_pose_estimates, file=pose_stream)
