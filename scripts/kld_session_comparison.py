"""Does a session-wise comparison of KL Divergence to show efficacy."""

from itertools import product
from pathlib import Path
from pickle import load

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

import ground_texture_systems

if __name__ == "__main__":
    with Path("intermediate_data/measurements.pkl").open(
        mode="rb",
    ) as measurement_stream:
        measurements: list[list[ground_texture_systems.Measurement]] = load(
            file=measurement_stream,
        )
    # For each session, bundle all the images together and create a set of 3
    # histograms - 1 per channel.
    histograms: list[list[NDArray]] = []
    for session_measurements in measurements:
        all_images = np.array(
            object=[measurement.image for measurement in session_measurements],
        )
        session_histograms: list[NDArray] = []
        for channel in range(3):
            pixel_values = all_images[:, :, :, channel].flatten() / 255.0
            hist_results = np.histogram(
                a=pixel_values,
                bins=256,
                range=(0.0, 1.0),
                density=True,
            )
            histogram = hist_results[0] * np.diff(a=hist_results[1])
            session_histograms.append(histogram)
        histograms.append(session_histograms)
    # Now calculate the KL Score for each combination of values. KLD is not
    # necessarily symmetric, so order matters.
    kld_scores = np.zeros(shape=(len(histograms), len(histograms)))
    for (first_id, first_histogram), (second_id, second_histogram) in product(
        enumerate(iterable=histograms),
        repeat=2,
    ):
        channel_scores: list[float] = []
        for channel in range(3):
            hist1 = first_histogram[channel]
            hist2 = second_histogram[channel]
            epsilon = 1e-20
            kld_score = np.sum(
                a=np.where(
                    hist2 != 0,
                    np.where(
                        hist1 != 0,
                        hist1 * np.log((hist1 + epsilon) / (hist2 + epsilon)),
                        0,
                    ),
                    0,
                ),
            )
            channel_scores.append(kld_score)
        kld_scores[first_id, second_id] = np.mean(channel_scores)
    # Plot the results.
    output_file = Path("output/kld_compare.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.matshow(A=kld_scores)
    for i, j in product(
        range(kld_scores.shape[0]),
        range(kld_scores.shape[1]),
    ):
        plt.text(
            x=j,
            y=i,
            s=f"{kld_scores[i, j]:0.1f}",
            ha="center",
            va="center",
        )
    plt.ylabel(ylabel="First Session ID")
    plt.xlabel(xlabel="Second Session ID")
    plt.savefig(output_file.as_posix(), dpi=600)
    plt.close()
