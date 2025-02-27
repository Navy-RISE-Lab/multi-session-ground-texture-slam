"""SLAM using joint intensity histogram to eliminate loop closures."""

import cv2
import gtsam
import numpy as np
from gtsam import Pose2
from numpy.typing import NDArray

from ground_texture_systems.systems import base


class JIH(base.Base):
    """SLAM using joint intensity histograms."""

    def __init__(self, common_options: base.CommonParameters) -> None:
        super().__init__(common_options=common_options)
        # Store the images for comparison.
        self._images: list[NDArray] = []
        self._session_map: dict[int, int] = {}

    def insert_frame(
        self,
        image: NDArray,
        pose: Pose2 | None = None,
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
        self._session_map[current_index] = self._current_session
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
        self._evaluate_loop(current_index=current_index)
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
        return super().new_session()

    def _calculate_jih(self, index1: int, index2: int) -> float:
        # Get both images and convert to grayscale.
        image1 = cv2.cvtColor(
            src=self._images[index1],
            code=cv2.COLOR_BGR2GRAY,
        )
        image2 = cv2.cvtColor(
            src=self._images[index2],
            code=cv2.COLOR_BGR2GRAY,
        )
        # Get the joint intensity histogram.
        histogram, _, _ = np.histogram2d(
            x=image1.flatten(),
            y=image2.flatten(),
            range=((0, 256), (0, 256)),
        )
        symmetric_part = 0.5 * (histogram + histogram.T)
        anti_symmetric_part = 0.5 * (histogram - histogram.T)
        symmetric_norm = np.linalg.norm(x=symmetric_part, ord="fro")
        anti_symmetric_norm = np.linalg.norm(x=anti_symmetric_part, ord="fro")
        score = 0.5 * (
            (symmetric_norm - anti_symmetric_norm)
            / (symmetric_norm + anti_symmetric_norm)
            + 1
        )
        return score.item()

    def _evaluate_loop(self, current_index: int) -> None:
        """Look for loop closures across all previous frames.

        This will go through each previous frame and see if there is a valid
        loop closure that passes all threshold criteria. If it is, the factor
        is added to the graph.

        Parameters
        ----------
        current_index
            The index to make loop closures against.
        """
        bag_of_words_query: dict[int, float] = (
            self._bag_of_words.query_database(
                self._descriptors[current_index],
            )
        )
        # Iterate through the scores from highest to lowest. Once the score
        # drops below the threshold, stop iterating.
        # Also, make sure the index is outside of the window.
        for candidate_index, bow_score in sorted(
            bag_of_words_query.items(),
            key=lambda items: items[1],
            reverse=True,
        ):
            if current_index - candidate_index < self._sliding_window:
                continue
            if bow_score < self._threshold_bag_of_words:
                break
            # Only run if the candidate and current are from different
            # sessions.
            current_session = self._session_map[current_index]
            candidate_session = self._session_map[candidate_index]
            match_result = self._match_features(
                first_index=current_index,
                second_index=candidate_index,
            )
            if match_result is None:
                continue
            keypoint_score = match_result[0].shape[0]
            if keypoint_score < self._threshold_keypoint:
                continue
            try:
                estimated_vector, covariance = (
                    self._transform_estimator.estimate_transform(
                        match_result[0],
                        match_result[1],
                    )
                )
                if current_session != candidate_session:
                    jih_score = self._calculate_jih(
                        index1=current_index,
                        index2=candidate_index,
                    )
                    # Scores are 1 for symmetrical, 0 for not.
                    covariance *= 1 / (jih_score + 1e-20)
                covariance_score = np.log10(
                    np.max(a=np.linalg.eigvals(covariance)),
                )
                if covariance_score > self._threshold_covariance:
                    continue
                estimated_transform = gtsam.Pose2(
                    x=estimated_vector[0],
                    y=estimated_vector[1],
                    theta=estimated_vector[2],
                )
                noise_model = gtsam.noiseModel.Gaussian.Covariance(covariance)
                factor = gtsam.BetweenFactorPose2(
                    key1=current_index,
                    key2=candidate_index,
                    relativePose=estimated_transform,
                    noiseModel=noise_model,
                )
                self._graph.add(factor=factor)
            except (RuntimeError, ValueError):
                continue

    def _parse_image(self, image: NDArray) -> int:
        """Store feature detection on an image.

        Parameters
        ----------
        image
            The image to process and store.

        Returns
        -------
            The unique index for this image, which can be used to look up
            features and descriptors in the associated lists.
        """
        self._images.append(image)
        return super()._parse_image(image=image)
