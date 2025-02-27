"""Plot all ground truth paths on a single map."""

from pathlib import Path
from pickle import load

from matplotlib import pyplot as plt

import ground_texture_systems

if __name__ == "__main__":
    # Read in ground truth data from the measurement files.
    with Path("intermediate_data/measurements.pkl").open(
        mode="rb",
    ) as measurement_stream:
        measurements: list[list[ground_texture_systems.Measurement]] = load(
            file=measurement_stream,
        )
    output_file = Path("output/ground_truth.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for i, session_measurements in enumerate(iterable=measurements):
        session_x = [
            measurement.actual_pose.x() for measurement in session_measurements
        ]
        session_y = [
            measurement.actual_pose.y() for measurement in session_measurements
        ]
        plt.scatter(x=session_x, y=session_y, marker=".", label=f"Session {i}")
    plt.title(label="Ground Truth Poses for Each Path")
    plt.xlabel(xlabel="X Position [m]")
    plt.ylabel(ylabel="Y Position [m]")
    plt.legend()
    plt.axis("equal")
    plt.gca().set(xlim=(0.0, 5.0), ylim=(0.0, 5.0))
    plt.savefig(output_file.as_posix(), dpi=600)
    plt.close()
