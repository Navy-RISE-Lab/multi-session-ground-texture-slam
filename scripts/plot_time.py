"""Create metrics and plots for the time of all timed systems."""

from argparse import ArgumentParser
from pathlib import Path
from pickle import load

import pandas as pd
from matplotlib import pyplot as plt
from yaml import safe_load

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
    # Read in and make plots for each system in the intermediate_data/times
    # folder. There is probably a more pythonic way to do this, but this will
    # work.
    # All the systems should have the same length values, but this will handle
    # it if they don't for some reason.
    results: dict[str, pd.Series] = {}
    system_poses_folder = Path("intermediate_data/times")
    for system_pose_file in system_poses_folder.glob(pattern="*.pkl"):
        system_name = system_pose_file.stem
        system_label = params["labels"][system_name]
        with system_pose_file.open(mode="rb") as time_stream:
            system_results: dict[int, float] = load(file=time_stream)
        results[system_label] = pd.Series(data=system_results)
    result_df = pd.DataFrame(data=results)
    output_csv = Path("output/times/times.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(path_or_buf=output_csv, index_label="Length")
    # Make the plot as well.
    plt.figure()
    result_df.plot(style=".-")
    plt.title(label="Average Time per Frame")
    plt.xlabel(xlabel="Total Number of Frames")
    plt.ylabel(ylabel="Average Time per Frame [s]")
    plt.savefig(Path("output/times/times.png"), dpi=600)
    plt.close()
