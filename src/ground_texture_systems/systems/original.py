"""Traditional SLAM system."""

import gtsam
import numpy as np
from numpy.typing import NDArray

from ground_texture_systems.systems import base


class Original(base.Base):
    """Traditional SLAM system."""

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
