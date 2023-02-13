from .sift import SIFTDetector
from .super_point import SuperPointDetector
from .r2d2 import R2D2Detector
from .orb import ORB2Detector

detector_map = {
    "sift": SIFTDetector,
    "super_point": SuperPointDetector,
    "r2d2": R2D2Detector,
    "orb": ORB2Detector,
}