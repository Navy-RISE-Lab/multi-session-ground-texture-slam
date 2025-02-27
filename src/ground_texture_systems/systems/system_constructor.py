"""Create a system from parameter specification."""

from pathlib import Path
from typing import Any

import numpy as np

from ground_texture_systems import components
from ground_texture_systems.systems.base import Base, CommonParameters
from ground_texture_systems.systems.jih import JIH
from ground_texture_systems.systems.kld import KLD
from ground_texture_systems.systems.kld_data import KLDData
from ground_texture_systems.systems.kld_gray import KLDGray
from ground_texture_systems.systems.odometry import Odometry
from ground_texture_systems.systems.original import Original
from ground_texture_systems.systems.overlap import Overlap
from ground_texture_systems.systems.single import Single


def construct_system(system: str, parameters: dict[str, Any]) -> Base:
    """Construct a system from CLI and YAML parameters.

    This is designed to abstract away the construction, so the scripts don't
    have to process all the dict values and command line options.

    Parameters
    ----------
    system
        The string name of the requested system. Must be one of: odometry.
    parameters
        The dict with all the system parameters from the param YAML file.

    Returns
    -------
        The constructed system.

    Raises
    ------
    ValueError
        Raised if the provided system is not recognized.
    """
    # Map all the values to the matching dataclass members.
    settings = CommonParameters(
        bag_of_words_options=components.BagOfWordsOptions(
            vocab_file=Path(parameters["bag_of_words"]["vocabulary_file"]),
        ),
        image_parser_options=components.ImageParserOptions(
            camera_pose=np.genfromtxt(
                fname=parameters["camera"]["pose"],
                delimiter=" ",
            ),
            camera_matrix=np.genfromtxt(
                fname=parameters["camera"]["intrinsic_matrix"],
                delimiter=" ",
            ),
            edge_threshold=parameters["image_parser"]["edge_threshold"],
            fast_threshold=parameters["image_parser"]["fast_threshold"],
            features=parameters["image_parser"]["features"],
            first_level=parameters["image_parser"]["first_level"],
            levels=parameters["image_parser"]["levels"],
            patch_size=parameters["image_parser"]["patch_size"],
            scale_factor=parameters["image_parser"]["scale_factor"],
            use_harris=parameters["image_parser"]["use_harris"],
            wta_k=parameters["image_parser"]["WTA_K"],
        ),
        keypoint_matcher_options=components.KeypointMatcherOptions(
            match_threshold=parameters["keypoint_matcher"]["match_threshold"],
            wta_k=parameters["image_parser"]["WTA_K"],
        ),
        transform_estimator_options=components.TransformEstimatorOptions(
            sigma=parameters["transform_estimator"]["sigma"],
            weight=parameters["transform_estimator"]["weight"],
            type=components.TransformRobustType.str_map(
                parameters["transform_estimator"]["type"],
            ),
        ),
        sliding_window=parameters["sliding_window"],
        threshold_bag_of_words=parameters["thresholds"]["bag_of_words"],
        threshold_keypoints=parameters["thresholds"]["keypoints"],
        threshold_covariance=parameters["thresholds"]["covariance"],
    )
    match system:
        case "jih":
            system_object = JIH(common_options=settings)
        case "kld":
            system_object = KLD(common_options=settings)
        case "kld_data":
            system_object = KLDData(common_options=settings)
        case "kld_gray":
            system_object = KLDGray(common_options=settings)
        case "odometry":
            system_object = Odometry(common_options=settings)
        case "original":
            system_object = Original(common_options=settings)
        case "overlap":
            system_object = Overlap(common_options=settings)
        case "single":
            system_object = Single(common_options=settings)
        case _:
            msg = (
                f"System {system} is not recognized! Must be one of: "
                "odometry, original"
            )
            raise ValueError(msg)
    return system_object
