"""Read in the data."""

from argparse import ArgumentParser
from pathlib import Path
from pickle import dump

import cv2
import numpy as np
from gtsam import Pose2
from numpy.typing import NDArray
from yaml import safe_load

from ground_texture_systems import Measurement


def load_session(filename: Path) -> tuple[list[NDArray], list[Pose2]]:
    """_Load a single session's worth of data.

    This includes both the images and ground truth information.

    Parameters
    ----------
    filename
        The file containing the list of images and poses, in HD Ground format.
        The file can be either relative or absolute.

    Returns
    -------
        A tuple containing the list of images and poses. The images are
        as-read, while the poses are converted to GTSAM's Pose2 type.
    """
    # All image files are relative paths, relative to the location of this
    # file. So extract its directory to assemble the absolute file name.
    base_directory = filename.resolve().parent
    images: list[NDArray] = []
    poses: list[Pose2] = []
    with filename.resolve().open() as path_stream:
        lines = path_stream.readlines()
    for i in range(0, len(lines), 2):
        # If the pose starts with a '*', skip, since that indicates unreliable
        # data.
        if "*" not in lines[i + 1]:
            image_file = lines[i].strip()
            image_file = base_directory / image_file
            image = cv2.imread(filename=image_file.as_posix())
            images.append(image)
            pose_array = np.fromstring(
                string=lines[i + 1].strip(),
                dtype=float,
                count=9,
                sep=" ",
            )
            pose_array = pose_array.reshape((3, 3))
            # Convert the pose from a homogenous transform into Pose2
            pose = Pose2(
                x=pose_array[0, 2],
                y=pose_array[1, 2],
                theta=np.arctan2(pose_array[1, 0], pose_array[0, 0]),
            )
            poses.append(pose)
    return images, poses


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
    measurements: list[list[Measurement]] = []
    index = 0
    for session_id, path_file in enumerate(iterable=params["paths"]):
        session_measurements: list[Measurement] = []
        images, poses = load_session(filename=Path(path_file))
        for measurement_id, (image, pose) in enumerate(
            iterable=zip(images, poses, strict=True),
        ):
            session_measurements.append(
                Measurement(
                    index=index,
                    session_id=session_id,
                    measurement_id=measurement_id,
                    image=image,
                    actual_pose=pose,
                ),
            )
            index += 1
        measurements.append(session_measurements)
    measurement_file = Path("intermediate_data/measurements.pkl")
    measurement_file.parent.mkdir(parents=True, exist_ok=True)
    with measurement_file.open(mode="wb") as measurement_stream:
        dump(obj=measurements, file=measurement_stream)
