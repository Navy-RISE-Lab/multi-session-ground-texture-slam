"""Plot maps for all systems."""

from argparse import ArgumentParser
from pathlib import Path
from pickle import load

from gtsam import Pose2
from matplotlib import pyplot as plt
from yaml import safe_load

import ground_texture_systems

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
        all_params = safe_load(stream=param_stream)
    params = all_params["plotting"]
    # Read in ground truth data from the measurement files.
    with Path("intermediate_data/measurements.pkl").open(
        mode="rb",
    ) as measurement_stream:
        measurements: list[list[ground_texture_systems.Measurement]] = load(
            file=measurement_stream,
        )
    all_actual_x: list[list[float]] = []
    all_actual_y: list[list[float]] = []
    for session_measurements in measurements:
        all_actual_x.append(
            [
                measurement.actual_pose.x()
                for measurement in session_measurements
            ],
        )
        all_actual_y.append(
            [
                measurement.actual_pose.y()
                for measurement in session_measurements
            ],
        )
    # Read in and make plots for each system in the intermediate_data/poses
    # folder.
    system_poses_folder = Path("intermediate_data/poses")
    for system_pose_file in system_poses_folder.glob(pattern="*.pkl"):
        system_name = system_pose_file.stem
        system_label = params["labels"][system_name]
        with system_pose_file.open(mode="rb") as pose_stream:
            system_poses: list[list[Pose2]] = load(file=pose_stream)
        output_folder = Path(f"output/maps/{system_name}/")
        output_folder.mkdir(parents=True, exist_ok=True)
        for session_id, (actual_x, actual_y) in enumerate(
            iterable=zip(all_actual_x, all_actual_y, strict=True),
        ):
            estimated_x = [pose.x() for pose in system_poses[session_id]]
            estimated_y = [pose.y() for pose in system_poses[session_id]]
            plt.figure()
            plt.scatter(x=actual_x, y=actual_y, label="Actual")
            plt.scatter(x=estimated_x, y=estimated_y, label="Estimated")
            plt.title(label=f"{system_label}: Session {session_id}")
            plt.xlabel(xlabel="X Position [m]")
            plt.ylabel(ylabel="Y Position [m]")
            plt.legend()
            plt.axis("equal")
            plt.gca().set(xlim=(0.0, 5.0), ylim=(0.0, 5.0))
            plt.savefig(
                (output_folder / f"{session_id}.png").as_posix(),
                dpi=600,
            )
            plt.close()
