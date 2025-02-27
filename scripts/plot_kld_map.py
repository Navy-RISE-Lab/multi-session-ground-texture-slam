"""Plot a heat map of KLD scores for health monitoring."""

import pickle
from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from pickle import load

import numpy as np
from gtsam import Pose2
from matplotlib import colors
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from yaml import safe_load

import ground_texture_systems


def _define_center_pose(
    camera_matrix: NDArray,
    camera_pose: NDArray,
    image: NDArray,
    estimated_pose: Pose2,
) -> tuple[float, float]:
    """Convert the image center point to the world frame.

    This will allow a geometrically accurate health map to be plotted
    regardless of robot configuration. This becomes particularly useful if the
    camera is significantly offset from the robot center.

    Parameters
    ----------
    camera_matrix
        The intrinsic matrix of the camera, as a 3x3.
    camera_pose
        The 3D pose of the camera, as measured from the robot's frame of
        reference, as a 4x3 homogenous transform.
    image
        The image, used to get the center point location.
    estimated_pose
        The estimated pose of the robot in the world when the image was taken.

    Returns
    -------
        The X and Y coordinates of the center pixel in the world frame.
    """
    # Define the center pixel using the image size.
    center_pixel = np.array(
        object=[[image.shape[1] / 2.0], [image.shape[0] / 2.0], [1.0]],
    )
    center_image = np.linalg.inv(a=camera_matrix) @ center_pixel
    # Add the known depth.
    center_image = np.insert(arr=center_image, obj=3, values=[1.0], axis=0)
    center_image[0:3, :] *= camera_pose[2, 3]
    # Convert from image to camera to robot to world.
    image_pose_camera = np.array(
        object=[
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    center_robot = camera_pose @ image_pose_camera @ center_image
    # This should be approximately in [x, y, 0, 1] form. The estimated pose
    # is a 3x3 homogenous matrix. So just pull out the x and y to transform.
    center_world = estimated_pose.matrix() @ center_robot[[0, 1, 3]]
    # This should now be of the form [x, y, 0, 1] so pull out the x and y.
    return center_world[0], center_world[1]


if __name__ == "__main__":
    # Get the camera matrices via parameters to plot score vs image center.
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
    # Measurements contain the image, and therefore the image dimensions.
    camera_pose = np.genfromtxt(
        fname=params["system"]["camera"]["pose"],
        delimiter=" ",
    )
    camera_matrix = np.genfromtxt(
        fname=params["system"]["camera"]["intrinsic_matrix"],
        delimiter=" ",
    )
    with Path("intermediate_data/measurements.pkl").open(
        mode="rb",
    ) as measurement_stream:
        measurements: list[list[ground_texture_systems.Measurement]] = (
            pickle.load(file=measurement_stream)
        )
    # Get the estimated poses according to KLD.
    with Path("intermediate_data/poses/kld.pkl").open(
        mode="rb",
    ) as pose_stream:
        estimated_poses: list[list[Pose2]] = load(file=pose_stream)
    with Path("intermediate_data/kld_scores.pkl").open(
        mode="rb",
    ) as kld_stream:
        kld_scores: list[list[float]] = load(file=kld_stream)
    output_folder = Path("output/health")
    output_folder.mkdir(parents=True, exist_ok=True)
    # Use a consistent color gradient across sessions.
    color_norm = colors.Normalize(vmin=0.0, vmax=ceil(max(max(kld_scores))))
    for session_id, (
        session_measurements,
        session_kld_scores,
        session_estimated_poses,
    ) in enumerate(
        iterable=zip(measurements, kld_scores, estimated_poses, strict=True),
    ):
        # Use the estimated pose, camera matrices, and image size to get the
        # image center.
        image_x: list[float] = []
        image_y: list[float] = []
        for measurement, estimated_pose in zip(
            session_measurements,
            session_estimated_poses,
            strict=True,
        ):
            result_x, result_y = _define_center_pose(
                camera_matrix=camera_matrix,
                camera_pose=camera_pose,
                image=measurement.image,
                estimated_pose=estimated_pose,
            )
            image_x.append(result_x)
            image_y.append(result_y)
        plt.figure()
        plt.scatter(
            x=image_x,
            y=image_y,
            c=session_kld_scores,
            norm=color_norm,
        )
        plt.title(label=f"KLD Score: Session {session_id}")
        plt.xlabel(xlabel="X Position [m]")
        plt.ylabel(ylabel="Y Position [m]")
        plt.colorbar(label="KLD Score")
        plt.axis("equal")
        plt.gca().set(xlim=(0.0, 5.0), ylim=(0.0, 5.0))
        plt.savefig((output_folder / f"{session_id}.png").as_posix(), dpi=600)
        plt.close()
