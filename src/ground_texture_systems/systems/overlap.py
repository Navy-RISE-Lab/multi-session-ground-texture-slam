"""SLAM with estimated overlap filtering."""

import gtsam
import numpy as np
import shapely
from numpy.typing import NDArray

from ground_texture_systems.systems import base


class Overlap(base.Base):
    """SLAM with field of view overlap filtering."""

    def __init__(self, common_options: base.CommonParameters) -> None:
        super().__init__(common_options=common_options)
        # These are all used to estimate fields of view.
        camera_matrix = common_options.image_parser_options.camera_matrix
        camera_pose = common_options.image_parser_options.camera_pose
        image_pose_camera = np.array(
            object=[
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        image_pose_robot = gtsam.Pose3(mat=camera_pose @ image_pose_camera)
        camera_k_gtsam = gtsam.Cal3_S2(
            fx=camera_matrix[0, 0],
            fy=camera_matrix[1, 1],
            s=camera_matrix[0, 1],
            u0=camera_matrix[0, 2],
            v0=camera_matrix[1, 2],
        )
        # This will be used to convert from pixels to the robot's frame, which
        # does not depend on the pose estimate.
        self._camera_model = gtsam.PinholeCameraCal3_S2(
            pose=image_pose_robot,
            K=camera_k_gtsam,
        )
        self._points_robot: list[NDArray] = []

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
        self._pose_estimates.insert_pose2(j=current_index, pose2=odom_pose)
        self._optimize()
        self._evaluate_loop(current_index=current_index)
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
            overlap = self._overlap(
                current_index=current_index,
                candidate_index=candidate_index,
            )
            if overlap <= 0.0:
                continue
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

    def _overlap(self, current_index: int, candidate_index: int) -> float:
        # Look up the current best pose of each frame.
        current_pose = self._pose_estimates.atPose2(j=current_index)
        candidate_pose = self._pose_estimates.atPose2(j=candidate_index)
        # Use these to transform the corner points into a world frame.
        current_points_world = current_pose.transformFrom(
            self._points_robot[current_index],
        )
        candidate_points_world = candidate_pose.transformFrom(
            self._points_robot[candidate_index],
        )
        # Make polygons out of these.
        current_polygon = shapely.geometry.Polygon(
            shell=current_points_world.T,
        )
        candidate_polygon = shapely.geometry.Polygon(
            shell=candidate_points_world.T,
        )
        intersection = current_polygon.intersection(other=candidate_polygon)
        return intersection.area

    def _parse_image(self, image: NDArray) -> int:
        # Create the field of view in pixels and then convert it to the
        # robot's frame of reference.
        points_pixel = np.array(
            object=[
                [0.0, 0.0],
                [image.shape[1], 0.0],
                [image.shape[1], image.shape[0]],
                [0.0, image.shape[0]],
            ],
        ).transpose()
        points_robot = np.ones_like(a=points_pixel)
        # Backproject doesn't work on an array of points, so calculate
        # serially.
        for point_index in range(points_robot.shape[1]):
            point_pixel = points_pixel[:, point_index]
            point_robot = self._camera_model.backproject(
                p=point_pixel,
                depth=self._camera_model.pose().z(),
            )
            points_robot[:, point_index] = point_robot[0:2]
        self._points_robot.append(points_robot)
        return super()._parse_image(image=image)
