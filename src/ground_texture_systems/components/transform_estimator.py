"""Construct a TransformEstimator component."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import ground_texture_slam


class TransformRobustType(Enum):
    """The robust estimator for the TransformEstimator."""

    CAUCHY = ground_texture_slam.TransformEstimator.Type.CAUCHY
    GEMAN_MCCLURE = ground_texture_slam.TransformEstimator.Type.GEMAN_MCCLURE
    HUBER = ground_texture_slam.TransformEstimator.Type.HUBER

    @staticmethod
    def str_map(robust_type: str) -> TransformRobustType:
        """Map string robust types to the Enum value.

        Parameters
        ----------
        robust_type
            A string representing the selected type. Should be "cauchy",
            "geman_mcclure", or "huber".

        Returns
        -------
            The correct Enum type.

        Raises
        ------
        ValueError
            Raised if the provided type is not recognized.
        """
        match robust_type.lower().strip():
            case "cauchy":
                return TransformRobustType.CAUCHY
            case "geman_mcclure":
                return TransformRobustType.GEMAN_MCCLURE
            case "huber":
                return TransformRobustType.HUBER
            case _:
                msg = (
                    f"{type} is not recognized. Use cauchy, geman_mcclure, "
                    "or huber!"
                )
                raise ValueError(msg)


@dataclass
class TransformEstimatorOptions:
    """Options for the TransformEstimator component."""

    sigma: float = 1.0
    weight: float = 1.345
    type: TransformRobustType = TransformRobustType.HUBER


def create_transform_estimator(
    options: TransformEstimatorOptions,
) -> ground_texture_slam.TransformEstimator:
    """Create a TransformEstimator object.

    Parameters
    ----------
    options
        Options for this component.

    Returns
    -------
        The constructed object.
    """
    transform_estimator_options = (
        ground_texture_slam.TransformEstimator.Options()
    )
    transform_estimator_options.measurement_sigma = options.sigma
    transform_estimator_options.weight = options.weight
    transform_estimator_options.type = options.type.value
    return ground_texture_slam.TransformEstimator(
        transform_estimator_options,
    )
