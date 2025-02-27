"""Construct an ImageParser component."""

from dataclasses import dataclass

import ground_texture_slam
from numpy.typing import NDArray


@dataclass
class ImageParserOptions:
    """Options for the ImageParser component.

    Values are the same as described in
    [OpenCV's documentation](https://docs.opencv.org/4.10.0/db/d95/classcv_1_1ORB.html#a72958366b9f52a0acd1cadc44d27268a).
    See that link for details.

    After being set, they are validated for correct bounds.
    """

    camera_pose: NDArray
    camera_matrix: NDArray
    edge_threshold: int = 31
    fast_threshold: int = 20
    features: int = 500
    first_level: int = 0
    levels: int = 8
    patch_size: int = 31
    scale_factor: float = 1.2
    use_harris: bool = True
    wta_k: int = 2

    def __post_init__(self) -> None:
        """Validate that all provided attributes are correctly bounded.

        Raises
        ------
        ValueError
            Raised if any attribute is given an incorrect value.
        """
        # Most of the parameters should be positive or at least not negative.
        if self.edge_threshold <= 0:
            msg = "Edge threshold should be positive!"
            raise ValueError(msg)
        if self.fast_threshold <= 0:
            msg = "Fast threshold should be positive!"
            raise ValueError(msg)
        if self.features < 0:
            msg = "The number of features should be zero or greater!"
            raise ValueError(msg)
        if self.first_level < 0:
            msg = "First level should be zero or greater!"
            raise ValueError(msg)
        if self.levels <= 0:
            msg = "Levels should be positive!"
            raise ValueError(msg)
        if self.patch_size <= 0:
            msg = "Patch size should be positive!"
            raise ValueError(msg)
        if self.scale_factor <= 1.0:
            msg = "Scale factor should be greater than one!"
            raise ValueError(msg)
        # The first level cannot be greater than the total levels - 1 (since
        # the count starts at zero).
        if self.first_level > self.levels - 1:
            msg = (
                f"The first level must be between 0 and {self.levels}, "
                "or change the value of levels!"
            )
            raise ValueError(msg)
        # WTA_K can only be 2, 3, or 4.
        if self.wta_k not in [2, 3, 4]:
            msg = "WTA_K must be 2, 3, or 4!"
            raise ValueError(msg)


def create_image_parser(
    options: ImageParserOptions,
) -> ground_texture_slam.ImageParser:
    """Create an ImageParser object.

    Parameters
    ----------
    options
        Options for this component.

    Returns
    -------
        The constructed object.
    """
    image_parser_options = ground_texture_slam.ImageParser.Options()
    image_parser_options.camera_pose = options.camera_pose
    image_parser_options.camera_intrinsic_matrix = options.camera_matrix
    image_parser_options.features = options.features
    image_parser_options.scale_factor = options.scale_factor
    image_parser_options.levels = options.levels
    image_parser_options.edge_threshold = options.edge_threshold
    image_parser_options.first_level = options.first_level
    image_parser_options.WTA_K = options.wta_k
    image_parser_options.use_harris_score = options.use_harris
    image_parser_options.patch_size = options.patch_size
    image_parser_options.fast_threshold = options.fast_threshold
    return ground_texture_slam.ImageParser(image_parser_options)
