"""Generate qualitative insights into how well KLD works."""

import dataclasses
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import gtsam
import numpy as np
import shapely
from yaml import safe_load

import ground_texture_systems
from ground_texture_systems.systems.kld_data import KLDData, LoopInsight


def _construct_camera(params: dict[str, Any]) -> gtsam.PinholeCameraCal3_S2:
    camera_pose = np.genfromtxt(
        fname=params["system"]["camera"]["pose"],
        delimiter=" ",
    )
    camera_matrix = np.genfromtxt(
        fname=params["system"]["camera"]["intrinsic_matrix"],
        delimiter=" ",
    )
    image_pose_camera = np.array(
        object=[
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    image_pose_robot = gtsam.Pose3(mat=camera_pose @ image_pose_camera)
    camera_k_gtsam = gtsam.Cal3_S2(
        fx=camera_matrix[0, 0],
        fy=camera_matrix[1, 1],
        s=camera_matrix[0, 1],
        u0=camera_matrix[0, 2],
        v0=camera_matrix[1, 2],
    )
    return gtsam.PinholeCameraCal3_S2(
        pose=image_pose_robot,
        K=camera_k_gtsam,
    )


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
    system: KLDData = ground_texture_systems.construct_system(
        system="kld_data",
        parameters=params["system"],
    )  # type: ignore[PyLance]
    # Feed in the images. We don't care about the results.
    # At the same time, convert the field of view to the world frame to allow
    # overlap calculations later on.
    fields_of_view: dict[int, shapely.geometry.Polygon] = {}
    actual_poses: dict[int, gtsam.Pose2] = {}
    camera_model = _construct_camera(params=params)
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
            # Create the field of view in the world frame using the image
            # corners and ground truth pose.
            points_pixel = np.array(
                object=[
                    [0.0, 0.0],
                    [measurement.image.shape[1], 0.0],
                    [measurement.image.shape[1], measurement.image.shape[0]],
                    [0.0, measurement.image.shape[0]],
                ],
            ).transpose()
            # Backproject doesn't work on an array of points, so calculate
            # serially.
            points_robot = np.ones_like(a=points_pixel)
            for point_index in range(points_robot.shape[1]):
                point_pixel = points_pixel[:, point_index]
                point_robot = camera_model.backproject(
                    p=point_pixel,
                    depth=camera_model.pose().z(),
                )
                points_robot[:, point_index] = point_robot[0:2]
            points_world = measurement.actual_pose.transformFrom(points_robot)
            fields_of_view[measurement.index] = shapely.geometry.Polygon(
                shell=points_world.T,
            )
            actual_poses[measurement.index] = measurement.actual_pose
        # Get the pose estimates for the most recent session.
        system.new_session()
    # Retrieve and add actual pose and loop info to the insights.
    loop_insights: dict[tuple[int, int], LoopInsight] = {}
    for (
        current_index,
        candidate_index,
    ), insight in system.loop_insights.items():
        # Use the pre-calculated fields of view to find if there is overlap
        # or not.
        intersection = fields_of_view[current_index].intersection(
            other=fields_of_view[candidate_index],
        )
        actual_transform = actual_poses[current_index].between(
            p2=actual_poses[candidate_index],
        )
        loop_insights[(current_index, candidate_index)] = dataclasses.replace(
            insight,
            actual_transform=actual_transform,
            actual_loop=intersection.area > 0.0,
        )
    loop_file = Path("intermediate_data/loop_insights.pkl")
    loop_file.parent.mkdir(parents=True, exist_ok=True)
    with loop_file.open(mode="wb") as loop_stream:
        pickle.dump(obj=loop_insights, file=loop_stream)
