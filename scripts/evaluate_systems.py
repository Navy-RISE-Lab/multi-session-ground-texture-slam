"""Evaluate the accuracy of all systems."""

from itertools import chain
from json import dump
from pathlib import Path
from pickle import load

import numpy as np
from gtsam import Pose2
from numpy.typing import NDArray

import ground_texture_systems


def rmse_orientation(actual: NDArray, estimated: NDArray) -> float:
    """Calculate the RMSE for orientation estimates.

    This must take in to account angle wrap.

    Parameters
    ----------
    actual
        The ground truth orientations, as a single vector.
    estimated
        The estimated orientations, as a single vector.

    Returns
    -------
        The RMSE error.
    """
    # This gets the smallest difference between two angles, when considering on
    # the domain [-pi, pi)
    orientation_errors = np.arctan2(
        np.sin(actual - estimated),
        np.cos(actual - estimated),
    )
    return np.sqrt(np.mean(a=np.square(orientation_errors)))


def rmse_position(actual: NDArray, estimated: NDArray) -> float:
    """Calculate the RMSE for position estimates.

    Parameters
    ----------
    actual
        The ground truth positions, as an Nx2 array.
    estimated
        The estimated positions, as an Nx2 array.

    Returns
    -------
        The RMSE error.
    """
    position_errors = np.linalg.norm(x=actual - estimated, axis=1)
    return np.sqrt(np.mean(a=np.square(position_errors)))


if __name__ == "__main__":
    # Read in ground truth data from the measurement files.
    with Path("intermediate_data/measurements.pkl").open(
        mode="rb",
    ) as measurement_stream:
        measurements: list[list[ground_texture_systems.Measurement]] = load(
            file=measurement_stream,
        )
    # This might be able to be a nested list comprehension.
    actual_pose_list = (
        (
            measurement.actual_pose.x(),
            measurement.actual_pose.y(),
            measurement.actual_pose.theta(),
        )
        for measurement in chain.from_iterable(measurements)
    )
    actual_pose = np.array(object=list(actual_pose_list))
    # For each system under test, load the estimated poses and use to calculate
    # accuracies.
    metrics: dict[str, dict[str, float]] = {
        "Position": {},
        "Orientation": {},
    }
    system_poses_folder = Path("intermediate_data/poses")
    for system_pose_file in system_poses_folder.glob(pattern="*.pkl"):
        system_name = system_pose_file.stem
        with system_pose_file.open(mode="rb") as pose_stream:
            system_poses: list[list[Pose2]] = load(file=pose_stream)
        estimated_pose = np.zeros_like(a=actual_pose)
        for i, pose in enumerate(iterable=chain.from_iterable(system_poses)):
            estimated_pose[i, 0] = pose.x()
            estimated_pose[i, 1] = pose.y()
            estimated_pose[i, 2] = pose.theta()
        metrics["Position"][system_name] = rmse_position(
            actual=actual_pose[:, 0:2],
            estimated=estimated_pose[:, 0:2],
        )
        metrics["Orientation"][system_name] = np.rad2deg(
            rmse_orientation(
                actual=actual_pose[:, 2],
                estimated=estimated_pose[:, 2],
            ),
        )
    metric_file = Path("output/metrics.json")
    metric_file.parent.mkdir(parents=True, exist_ok=True)
    with metric_file.open(mode="w") as metrics_stream:
        dump(obj=metrics, fp=metrics_stream)
