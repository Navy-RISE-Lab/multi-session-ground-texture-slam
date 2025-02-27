"""Time how long a system takes per frame."""

import pickle
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

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
    time_results: dict[int, float] = {}
    # Create enough systems for averaging.
    total_systems = params["timing"]["samples_per_data_point"]
    systems = [
        ground_texture_systems.construct_system(
            system=args.system,
            parameters=params["system"],
        )
        for _ in range(total_systems)
    ]
    frames_processed = 0
    for session_measurements in measurements:
        for measurement in session_measurements:
            frames_processed += 1
            start_time = perf_counter()
            for system in systems:
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
            end_time = perf_counter()
            time_results[frames_processed] = (
                end_time - start_time
            ) / total_systems
        for system in systems:
            system.new_session()
    # Store the results.
    time_file = Path(f"intermediate_data/times/{args.system}.pkl")
    time_file.parent.mkdir(parents=True, exist_ok=True)
    with time_file.open(mode="wb") as time_stream:
        pickle.dump(obj=time_results, file=time_stream)
