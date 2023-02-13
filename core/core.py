from typing import Dict, Any, Tuple
import abc
from abc import ABC
from abc import abstractmethod
import numpy as np


class Detector(ABC):
    """Abstract class for detector."""

    # Set this in SOME subclasses
    metadata = {}

    # Set this in ALL subclasses
    device = None
    dim_feature = None  # dimension of the feature
    max_feature = None  # maximum number of features in an image
    thresh_confid = None  # threshold of the feature confidence

    @abc.abstractmethod
    def detect(self, image) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Detect keypoints and descriptors from a single image.
        Args:
            image (np.ndarray): image to be detected.
        Returns:
            xys (np.ndarray): keypoints' coordinates and size.
            desc (np.ndarray): descriptors.
            scores (np.ndarray): scores of keypoints.
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self


class DetectorWrapper(Detector):
    def __init__(self, detector):
        self.detector = detector

        self._dim_feature = None
        self._max_feature = None
        self._thresh_confid = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.env, name)

    @property
    def dim_feature(self):
        if self._dim_feature is None:
            return self.detector.dim_feature
        return self._dim_feature

    @dim_feature.setter
    def dim_feature(self, value):
        self._dim_feature = value

    @property
    def max_feature(self):
        if self._max_feature is None:
            return self.detector.max_feature
        return self._max_feature

    @max_feature.setter
    def max_feature(self, value):
        self._max_feature = value

    @property
    def thresh_confid(self):
        if self._thresh_confid is None:
            return self.detector.thresh_confid
        return self._thresh_confid

    @thresh_confid.setter
    def thresh_confid(self, value):
        self._thresh_confid = value

    def detect(self, image):
        return self.detector.detect(image)

    @property
    def unwrapped(self):
        return self.detector.unwrapped
