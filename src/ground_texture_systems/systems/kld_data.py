"""A reconfiguration of KLD to store extra diagnostic data."""

import dataclasses
from pathlib import Path
from pickle import dump

import gtsam
import numpy as np
from numpy.typing import NDArray

from ground_texture_systems.systems import base


@dataclasses.dataclass
class LoopInsight:
    """Data concerning the result of loop closure estimation."""

    first_index: int
    first_session_id: int
    second_index: int
    second_session_id: int
    window_threshold: bool
    bow_threshold: bool
    match_threshold: bool | None
    cov_original_threshold: bool | None
    cov_kld_threshold: bool | None
    estimated_transform: gtsam.Pose2 | None
    actual_transform: gtsam.Pose2 | None = None
    actual_loop: bool | None = None


class KLDData(base.Base):
    """SLAM using KLD to adjust loop closure confidence."""

    def __init__(self, common_options: base.CommonParameters) -> None:
        super().__init__(common_options=common_options)
        self._baseline_images: list[NDArray] = []
        self._baseline_histograms: list[NDArray] = []
        # These are logged only for illustration purposes.
        self._kld_scores: list[list[float]] = []
        # Start the list for the first session.
        self._kld_scores.append([])
        # Loop closure confusion matrix data.
        self.loop_insights: dict[tuple[int, int], LoopInsight] = {}

    def insert_frame(
        self,
        image: NDArray,
        pose: gtsam.Pose2 | None = None,
        variance: float | None = None,
    ) -> int:
        """Add a new measurement frame to the system.

        Parameters
        ----------
        image
            The image taken at this frame.
        pose, optional
            The pose at this frame, if known. By default None, meaning no prior
            estimate.
        variance, optional
            The variance of the pose, if known. By default None, meaning no
            prior estimate.

        Returns
        -------
            the unique index of this frame.

        Raises
        ------
        RuntimeError
            Raised if odometry fails.
        """
        current_index = self._parse_image(image=image)
        # Store the images if this is still the baseline first session
        if self._current_session == 0:
            self._baseline_images.append(image)
        # If there is a previous frame, do odometry.
        if current_index > 0:
            odom_pose, odom_factor = self._odometry(
                current_index=current_index,
            )
            self._graph.add(factor=odom_factor)
        # If a known prior is provided, this will be an additional factor
        # and will replace the pose estimate.
        if pose is not None and variance is not None:
            odom_pose = pose
            self._graph.add(
                factor=self._create_prior(
                    current_index=current_index,
                    pose=odom_pose,
                    variance=variance,
                ),
            )
        self._evaluate_loop(current_index=current_index, image=image)
        self._pose_estimates.insert_pose2(j=current_index, pose2=odom_pose)
        self._optimize()
        return current_index

    def new_session(self) -> int:
        """Signal the start of a new session.

        Returns
        -------
            The unique ID of the session. This increases monotonically (i.e. 0,
            1, 2, ...).
        """
        # If this is the first session, construct the baseline histograms for
        # each channel.
        if self._current_session == 0:
            image_array = np.array(object=self._baseline_images)
            for channel in range(3):
                channel_array = image_array[:, :, :, channel]
                self._baseline_histograms.append(
                    self._create_histogram(image=channel_array),
                )
        # For illustration only: write/overwrite the list of KLD scores
        with Path("intermediate_data/kld_scores.pkl").open(
            mode="wb",
        ) as kld_stream:
            dump(obj=self._kld_scores, file=kld_stream)
        self._kld_scores.append([])
        return super().new_session()

    def _create_histogram(self, image: NDArray) -> NDArray:
        """Convert an image into a normalized histogram.

        The histogram assumes 8-bit, so there are 256 bars.

        Parameters
        ----------
        image
            The image to convert. This can be either a grayscale image or
            single channel.

        Returns
        -------
            The normalized histogram.
        """
        # Normalize the pixel values.
        pixel_values = image.flatten() / 255.0
        # Make a histogram.
        result = np.histogram(
            a=pixel_values,
            bins=256,
            range=(0.0, 1.0),
            density=True,
        )
        # The histogram provides the "height". Multiply by the diff, which is
        # the bin "width" to get the area for that bin.
        return result[0] * np.diff(a=result[1])

    def _evaluate_loop(self, current_index: int, image: NDArray) -> None:
        """Look for loop closures across all previous frames.

        This will go through each previous frame and see if there is a valid
        loop closure that passes all threshold criteria. If it is, the factor
        is added to the graph.

        Parameters
        ----------
        current_index
            The index to make loop closures against.
        image
            The image captures at the current frame.
        """
        bag_of_words_query: dict[int, float] = (
            self._bag_of_words.query_database(
                self._descriptors[current_index],
            )
        )
        # Calculate the KLD score no matter what. This will increase
        # calculation time, but aids in health map plotting.
        if self._current_session > 0:
            image_histograms = [
                self._create_histogram(image=image[:, :, channel])
                for channel in range(3)
            ]
            kld_scores = [
                self._kld(
                    hist1=image_histograms[channel],
                    hist2=self._baseline_histograms[channel],
                )
                for channel in range(3)
            ]
            kld_score = np.mean(kld_scores).item()
        else:
            kld_score = 0.0
        self._kld_scores[-1].append(kld_score)
        # Iterate through all possible loops. Calculate everything for
        # diagnostic information.
        for candidate_index, bow_score in sorted(
            bag_of_words_query.items(),
            key=lambda items: items[1],
            reverse=True,
        ):
            window_threshold = (
                current_index - candidate_index >= self._sliding_window
            )
            bow_threshold = bow_score >= self._threshold_bag_of_words
            match_result = self._match_features(
                first_index=current_index,
                second_index=candidate_index,
            )
            if match_result is not None:
                match_threshold = (
                    match_result[0].shape[0] >= self._threshold_keypoint
                )
                try:
                    estimated_vector, covariance = (
                        self._transform_estimator.estimate_transform(
                            match_result[0],
                            match_result[1],
                        )
                    )
                    estimated_transform = gtsam.Pose2(
                        x=estimated_vector[0],
                        y=estimated_vector[1],
                        theta=estimated_vector[2],
                    )
                except (RuntimeError, ValueError):
                    estimated_transform = None
                    covariance = None
            else:
                match_threshold = None
                estimated_transform = None
                covariance = None
            if estimated_transform is not None and covariance is not None:
                cov_original_score = np.log10(
                    np.max(a=np.linalg.eigvals(covariance)),
                )
                # Use the KLD score to offset the covariance.
                # Add 1 so perfect is unity, not zero.
                covariance *= kld_score + 1
                cov_kld_score = np.log10(
                    np.max(a=np.linalg.eigvals(covariance)),
                )
                cov_original_threshold = (
                    cov_original_score <= self._threshold_covariance
                )
                cov_kld_threshold = cov_kld_score <= self._threshold_covariance
            else:
                cov_original_threshold = None
                cov_kld_threshold = None
            self.loop_insights[(current_index, candidate_index)] = LoopInsight(
                first_index=current_index,
                first_session_id=self._session_map[current_index],
                second_index=candidate_index,
                second_session_id=self._session_map[candidate_index],
                window_threshold=window_threshold,
                bow_threshold=bow_threshold,
                match_threshold=match_threshold,
                cov_original_threshold=cov_original_threshold,
                cov_kld_threshold=cov_kld_threshold,
                estimated_transform=estimated_transform,
            )
            # Still only add factors if all thresholds pass.
            if (
                window_threshold
                and bow_threshold
                and match_threshold
                and cov_kld_threshold
                and estimated_transform is not None
                and covariance is not None
            ):
                noise_model = gtsam.noiseModel.Gaussian.Covariance(
                    covariance,
                )
                factor = gtsam.BetweenFactorPose2(
                    key1=current_index,
                    key2=candidate_index,
                    relativePose=estimated_transform,
                    noiseModel=noise_model,
                )
                self._graph.add(factor=factor)

    def _kld(self, hist1: NDArray, hist2: NDArray) -> float:
        """Calculate KL Divergence for two normalized histograms.

        Specifically, this is the divergence of hist1 from hist2.

        Parameters
        ----------
        hist1
            The first histogram.
        hist2
            The second histogram.

        Returns
        -------
            The score.
        """
        # Add a tiny number to avoid divide by zero.
        epsilon = 1e-20
        return np.sum(
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
