"""Gather the pose estimates and images for the video."""

import dataclasses
import pickle
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

from gtsam import Pose2
from numpy.typing import NDArray
from yaml import safe_load

import ground_texture_systems
from ground_texture_systems.systems.base import Base


@dataclasses.dataclass
class FrameData:
    """Helper class to store all data associated with one frame."""

    image: NDArray
    session_actual: list[Pose2]
    session_poses: dict[str, list[Pose2]]


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        description="Load multi-session data.",
    )
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
    systems: dict[str, Base] = {}
    for name in ["original", "kld"]:
        systems[name] = ground_texture_systems.construct_system(
            system=name,
            parameters=params["system"],
        )
    frame_data: list[FrameData] = []
    for session_measurements in measurements:
        # Build the list of ground truth poses for storing.
        session_ground_truths: list[Pose2] = []
        for measurement in session_measurements:
            session_ground_truths.append(measurement.actual_pose)
            if measurement.measurement_id == 0:
                pose_prior = measurement.actual_pose
                pose_prior_variance = 1e-5
            else:
                pose_prior = None
                pose_prior_variance = None
            system_poses: dict[str, list[Pose2]] = {}
            for name, system in systems.items():
                system.insert_frame(
                    image=measurement.image,
                    pose=pose_prior,
                    variance=pose_prior_variance,
                )
                # Get the pose estimates. They are binned by session, so grab
                # the latest and add it.
                system_poses[name] = system.pose_estimates[-1]
            # Copy so that references don't screw up the results.
            frame_data.append(
                FrameData(
                    image=deepcopy(x=measurement.image),
                    session_actual=deepcopy(x=session_ground_truths),
                    session_poses=deepcopy(x=system_poses),
                ),
            )
        for system in systems.values():
            system.new_session()
    frame_file = Path("intermediate_data/frame_data.pkl")
    frame_file.parent.mkdir(parents=True, exist_ok=True)
    with frame_file.open(mode="wb") as frame_stream:
        pickle.dump(obj=frame_data, file=frame_stream)
