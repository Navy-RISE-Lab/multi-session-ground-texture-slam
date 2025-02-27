"""Uses the video data to create a video."""

import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
from gather_video_data import FrameData
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import figaspect
from matplotlib.image import AxesImage
from yaml import safe_load


def draw_frame(
    frame: int,
    frame_list: list[FrameData],
    plots: tuple[
        PathCollection,
        PathCollection,
        AxesImage,
        PathCollection,
        PathCollection,
    ],
) -> tuple[
    PathCollection,
    PathCollection,
    AxesImage,
    PathCollection,
    PathCollection,
]:
    """Generate a single frame of data.

    This will update all the plots with the given frame data. This should be
    passed into FuncAnimation with a partial to map the arguments.

    Parameters
    ----------
    frame
        The index for the data to draw.
    frame_list
        The total list of frame information.
    draws
        The tuple of plots to update.

    Returns
    -------
        The updated tuple of plots.
    """
    actual_x = [pose.x() for pose in frame_list[frame].session_actual]
    actual_y = [pose.y() for pose in frame_list[frame].session_actual]
    original_x = [
        pose.x() for pose in frame_list[frame].session_poses["original"]
    ]
    original_y = [
        pose.y() for pose in frame_list[frame].session_poses["original"]
    ]
    kld_x = [pose.x() for pose in frame_list[frame].session_poses["kld"]]
    kld_y = [pose.y() for pose in frame_list[frame].session_poses["kld"]]
    plots[0].set_offsets(offsets=np.stack(arrays=[actual_x, actual_y]).T)
    plots[1].set_offsets(offsets=np.stack(arrays=[original_x, original_y]).T)
    # Convert BGR to RGB.
    plots[2].set_data(A=frame_list[frame].image[:, :, ::-1])
    plots[3].set_offsets(offsets=np.stack(arrays=[actual_x, actual_y]).T)
    plots[4].set_offsets(offsets=np.stack(arrays=[kld_x, kld_y]).T)
    return plots


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
    with Path("intermediate_data/frame_data.pkl").open(
        mode="rb",
    ) as frame_stream:
        frame_data: list[FrameData] = pickle.load(file=frame_stream)
    ax0: Axes
    ax1: Axes
    ax2: Axes
    figure, (ax0, ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=figaspect(9.0 / 16.0),
    )
    # Left contains the original vs actual.
    actual_draw0 = ax0.scatter(x=[], y=[], label="Actual")
    original_draw = ax0.scatter(x=[], y=[], label="Estimated")
    ax0.set_xlim(left=0.0, right=5.0)
    ax0.set_ylim(bottom=0.0, top=5.0)
    ax0.set_aspect(aspect="equal")
    ax0.set_xlabel(xlabel="X Position [m]")
    ax0.set_ylabel(ylabel="Y Position [m]")
    ax0.set_title(label=params["plotting"]["labels"]["original"])
    ax0.legend()
    # Middle contains the ground texture image.
    image_draw = ax1.imshow(X=frame_data[0].image[:, :, ::-1])
    ax1.set_title(label="Ground Texture Image")
    ax1.axis("off")
    # Right contains the KLD vs actual.
    actual_draw2 = ax2.scatter(x=[], y=[], label="Actual")
    kld_draw = ax2.scatter(x=[], y=[], label="Estimated")
    ax2.set_xlim(left=0.0, right=5.0)
    ax2.set_ylim(bottom=0.0, top=5.0)
    ax2.set_aspect(aspect="equal")
    ax2.set_xlabel(xlabel="X Position [m]")
    ax2.set_ylabel(ylabel="Y Position [m]")
    ax2.set_title(label=params["plotting"]["labels"]["kld"])
    ax2.legend()
    # Adjust the figure size so the plots aren't scrunched.
    figure.set_figheight(val=1.5 * figure.get_figheight())
    figure.set_figwidth(val=1.5 * figure.get_figheight())
    figure.tight_layout()
    # Add some overall info.
    figure.suptitle(
        t="SLAM System Comparison",
        fontsize="xx-large",
        fontweight="bold",
    )
    animation = FuncAnimation(
        fig=figure,
        func=partial(
            draw_frame,
            frame_list=frame_data,
            plots=(
                actual_draw0,
                original_draw,
                image_draw,
                actual_draw2,
                kld_draw,
            ),
        ),
        frames=len(frame_data),
        interval=125,
    )
    output_video = Path("output/video.mp4")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    animation.save(filename=output_video.as_posix())
