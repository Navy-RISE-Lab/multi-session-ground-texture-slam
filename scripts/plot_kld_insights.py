"""Plot the loop graphs."""

import itertools
from pathlib import Path
from pickle import load

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from ground_texture_systems.systems.kld_data import LoopInsight

if __name__ == "__main__":
    with Path("intermediate_data/loop_insights.pkl").open(
        mode="rb",
    ) as loop_stream:
        loop_insights: dict[tuple[int, int], LoopInsight] = load(
            file=loop_stream,
        )
    output_folder = Path("output/loop_insights")
    output_folder.mkdir(parents=True, exist_ok=True)
    # Make confusion matrices for original and KLD
    truth: list[bool] = []
    original: list[bool] = []
    kld: list[bool] = []
    for insight in loop_insights.values():
        # Only include loops where the change in covariance could make a
        # difference. So exclude all that get cut off at prior thresholds.
        # Also, exclude entries without actual data instead of None.
        if (
            insight.window_threshold
            and insight.bow_threshold
            and insight.match_threshold
            and insight.actual_loop is not None
            and insight.cov_original_threshold is not None
            and insight.cov_kld_threshold is not None
        ):
            # Only record loop pairs that have all three values.
            truth.append(insight.actual_loop)
            original.append(insight.cov_original_threshold)
            kld.append(insight.cov_kld_threshold)
    ConfusionMatrixDisplay.from_predictions(
        y_true=truth,
        y_pred=original,
        labels=[False, True],
        display_labels=["No Loop", "Loop"],
        colorbar=False,
    )
    plt.title(label="Loop Classification for Original SLAM System")
    plt.savefig((output_folder / "original.png").as_posix(), dpi=600)
    plt.close()
    ConfusionMatrixDisplay.from_predictions(
        y_true=truth,
        y_pred=kld,
        labels=[False, True],
        display_labels=["No Loop", "Loop"],
        colorbar=False,
    )
    plt.title(label="Loop Classification for KLD SLAM System")
    plt.savefig((output_folder / "kld.png").as_posix(), dpi=600)
    plt.close()
    # Make a plot of the accuracy of falsely removed loop closures vs correct
    # loop closures.
    false_errors = [
        np.linalg.norm(
            x=insight.estimated_transform.translation()
            - insight.actual_transform.translation(),
        ).item()
        for insight in loop_insights.values()
        if insight.actual_loop
        and insight.cov_original_threshold
        and not insight.cov_kld_threshold
        and insight.estimated_transform is not None
        and insight.actual_transform is not None
    ]
    true_errors = [
        np.linalg.norm(
            x=insight.estimated_transform.translation()
            - insight.actual_transform.translation(),
        ).item()
        for insight in loop_insights.values()
        if insight.actual_loop
        and insight.cov_original_threshold
        and insight.cov_kld_threshold
        and insight.estimated_transform is not None
        and insight.actual_transform is not None
    ]
    plt.boxplot(
        x=(false_errors, true_errors),
        tick_labels=("False Negatives", "True Positives"),
        notch=False,
        showfliers=False,
    )
    plt.title(
        label=(
            "Position Errors for Loop Closures Kept"
            " or Discarded by the KLD System."
        ),
    )
    plt.xlabel(xlabel="Loop Closure Type")
    plt.ylabel(ylabel="Position Accuracy [m]")
    plt.savefig(
        (output_folder / "position_error.png").as_posix(),
        dpi=600,
    )
    plt.close()
    # Make a matrix showing inter-session loop closures.
    # This is inefficient, but figure out how many sessions there are from
    # just the loop insight data.
    session_ids: list[int] = []
    for insight in loop_insights.values():
        if insight.first_session_id not in session_ids:
            session_ids.append(insight.first_session_id)
    session_loops = np.zeros(
        shape=(len(session_ids), len(session_ids)),
        dtype=np.int32,
    )
    for insight in loop_insights.values():
        # Only include ones that are considered valid loops by KLD
        if (
            insight.window_threshold
            and insight.bow_threshold
            and insight.match_threshold
            and insight.cov_kld_threshold
            and insight.estimated_transform is not None
        ):
            session_loops[
                insight.first_session_id,
                insight.second_session_id,
            ] += 1
    figure, axis = plt.subplots()
    # Double the color scale so that the highest values are readable with the
    # text labels on them.
    axis.matshow(
        Z=session_loops.transpose(),
        cmap="Blues",
        vmin=0,
        vmax=1.5 * np.max(a=session_loops),
    )
    for first_session, second_session in itertools.product(
        session_ids,
        repeat=2,
    ):
        axis.text(
            x=first_session,
            y=second_session,
            s=f"{session_loops[first_session, second_session]}",
            va="center",
            ha="center",
        )
    axis.set_title(label="Number of Loops Made Between Sessions")
    axis.xaxis.set_ticks_position(position="bottom")
    axis.set_xlabel(xlabel="Current Session Index")
    axis.set_ylabel(ylabel="Previous Session Index")
    figure.savefig(
        fname=(output_folder / "inter_session_loops.png").as_posix(),
        dpi=600,
    )
    plt.close(fig=figure)
    # Make a Pandas CSV
    insight_list = []
    for (current_index, candidate_index), insight in loop_insights.items():
        addition = {
            "Current": current_index,
            "Candidate": candidate_index,
            "Truth": insight.actual_loop,
            "Window": insight.window_threshold,
            "BoW": insight.bow_threshold,
            "Match": insight.match_threshold,
            "Original": insight.cov_original_threshold,
            "KLD": insight.cov_kld_threshold,
        }
        if insight.actual_transform is None:
            addition["Actual X"] = None
            addition["Actual Y"] = None
            addition["Actual T"] = None
        else:
            addition["Actual X"] = insight.actual_transform.x()
            addition["Actual Y"] = insight.actual_transform.y()
            addition["Actual T"] = np.rad2deg(
                insight.actual_transform.theta(),
            )
        if insight.estimated_transform is None:
            addition["Estimated X"] = None
            addition["Estimated Y"] = None
            addition["Estimated T"] = None
        else:
            addition["Estimated X"] = insight.estimated_transform.x()
            addition["Estimated Y"] = insight.estimated_transform.y()
            addition["Estimated T"] = np.rad2deg(
                insight.estimated_transform.theta(),
            )
        insight_list.append(addition)
    insight_df = pd.DataFrame(data=insight_list)
    insight_df.to_csv(
        path_or_buf=(output_folder / "loops.csv").as_posix(),
        index=False,
    )
