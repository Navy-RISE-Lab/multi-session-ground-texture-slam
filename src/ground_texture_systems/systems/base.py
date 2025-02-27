"""Base class with common components."""

import abc
from dataclasses import dataclass

import cv2
import gtsam
from numpy.typing import NDArray

from ground_texture_systems import components


@dataclass
class CommonParameters:
    """All parameters in common across systems."""

    bag_of_words_options: components.BagOfWordsOptions
    image_parser_options: components.ImageParserOptions
    keypoint_matcher_options: components.KeypointMatcherOptions
    transform_estimator_options: components.TransformEstimatorOptions
    sliding_window: int
    threshold_bag_of_words: float = 0.0
    threshold_keypoints: int = 0
    threshold_covariance: float = 1e5


class Base(abc.ABC):
    """Base SLAM system with common elements and components."""

    def __init__(self, common_options: CommonParameters) -> None:
        super().__init__()
        self._transform_estimator = components.create_transform_estimator(
            options=common_options.transform_estimator_options,
        )
        self._image_parser = components.create_image_parser(
            options=common_options.image_parser_options,
        )
        self._keypoint_matcher = components.create_keypoint_matcher(
            options=common_options.keypoint_matcher_options,
        )
        self._bag_of_words = components.create_bag_of_words(
            options=common_options.bag_of_words_options,
        )
        self._sliding_window = common_options.sliding_window
        self._threshold_bag_of_words = common_options.threshold_bag_of_words
        self._threshold_keypoint = common_options.threshold_keypoints
        self._threshold_covariance = common_options.threshold_covariance
        self._graph = gtsam.NonlinearFactorGraph()
        self._pose_estimates = gtsam.Values()
        self._features: list[NDArray] = []
        self._descriptors: list[NDArray] = []
        self._current_session: int = 0
        self._session_map: dict[int, int] = {}

    @abc.abstractmethod
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
            The unique index of this frame.
        """

    @abc.abstractmethod
    def new_session(self) -> int:
        """Signal the start of a new session.

        Returns
        -------
            The unique ID of the session.
        """
        self._current_session += 1
        return self._current_session

    @property
    def pose_estimates(self) -> list[list[gtsam.Pose2]]:
        """Get the most current pose estimates, binned by session.

        Returns
        -------
            The set of pose estimates. Each inner list is one session.
        """
        estimates: list[list[gtsam.Pose2]] = [
            [] for _i in range(self._current_session + 1)
        ]
        for index in range(len(self._features)):
            pose = self._pose_estimates.atPose2(j=index)
            session_id = self._session_map[index]
            estimates[session_id].append(pose)
        return estimates

    def _create_prior(
        self,
        current_index: int,
        pose: gtsam.Pose2,
        variance: float,
    ) -> gtsam.PriorFactorPose2:
        """Create a prior factor from a known pose and confidence.

        Parameters
        ----------
        current_index
            The index of the pose.
        pose
            The estimated pose.
        variance
            The variance for the covariance of the pose estimate.

        Returns
        -------
            The factor, ready to be added to the graph.
        """
        noise_model = gtsam.noiseModel.Isotropic.Variance(
            dim=3,
            varianace=variance,
        )
        return gtsam.PriorFactorPose2(
            key=current_index,
            prior=pose,
            noiseModel=noise_model,
        )

    def _match_features(
        self,
        first_index: int,
        second_index: int,
    ) -> tuple[NDArray, NDArray] | None:
        """Perform feature matching between two frames.

        Parameters
        ----------
        first_index
            The index of the first frame.
        second_index
            The index of the second frame.

        Returns
        -------
            If a match can be made, returns two arrays of features. If no
            match, returns None.
        """
        try:
            result = self._keypoint_matcher.find_matched_keypoints(
                self._features[first_index],
                self._descriptors[first_index],
                self._features[second_index],
                self._descriptors[second_index],
            )
        except RuntimeError:
            result = None
        return result

    def _odometry(
        self,
        current_index: int,
    ) -> tuple[gtsam.Pose2, gtsam.BetweenFactorPose2]:
        """Estimate the transform between a frame and the frame just before it.

        Parameters
        ----------
        current_index
            The frame to estimate.

        Returns
        -------
            The estimated pose of the frame and the factor between this frame
            and the one right before it.

        Raises
        ------
        RuntimeError
            Raised if a valid transform cannot be found.
        """
        previous_index = current_index - 1
        match_result = self._match_features(
            first_index=previous_index,
            second_index=current_index,
        )
        if match_result is None:
            msg = "Odometry failed somehow!"
            raise RuntimeError(msg)
        try:
            estimated_vector, covariance = (
                self._transform_estimator.estimate_transform(
                    match_result[0],
                    match_result[1],
                )
            )
        except (RuntimeError, ValueError) as err:
            msg = "Odometry failed somehow!"
            raise RuntimeError(msg) from err
        estimated_pose = gtsam.Pose2(
            x=estimated_vector[0],
            y=estimated_vector[1],
            theta=estimated_vector[2],
        )
        noise_model = gtsam.noiseModel.Gaussian.Covariance(covariance)
        factor = gtsam.BetweenFactorPose2(
            key1=previous_index,
            key2=current_index,
            relativePose=estimated_pose,
            noiseModel=noise_model,
        )
        previous_pose = self._pose_estimates.atPose2(j=previous_index)
        current_pose = previous_pose.compose(estimated_pose)
        return current_pose, factor

    def _optimize(self) -> None:
        """Optimize the graph to get the latest pose estimates."""
        self._pose_estimates: gtsam.Values = gtsam.LevenbergMarquardtOptimizer(
            graph=self._graph,
            initialValues=self._pose_estimates,
        ).optimize()

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
        image_gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
        features, _, descriptors = self._image_parser.parse_image(image_gray)
        self._features.append(features)
        self._descriptors.append(descriptors)
        self._bag_of_words.insert_to_database(descriptors)
        current_index = len(self._features) - 1
        self._session_map[current_index] = self._current_session
        return current_index
