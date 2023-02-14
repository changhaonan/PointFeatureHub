from .sift import SIFTDetector
from .super_point import SuperPointDetector
from .r2d2 import R2D2Detector
from .orb import ORB2Detector

from .bf import BFMatcher
from .super_glue import SuperGlueMatcher
from .loftr import LoFTRMatcher

detector_map = {
    "sift": SIFTDetector,
    "super_point": SuperPointDetector,
    "r2d2": R2D2Detector,
    "orb": ORB2Detector,
}

matcher_map = {
    "bf": BFMatcher,
    "super_glue": SuperGlueMatcher,
    "loftr": LoFTRMatcher,
}