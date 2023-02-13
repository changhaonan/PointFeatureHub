import cv2
import os
from core.core import Detector, DetectorWrapper
import numpy as np


class SaveImageWrapper(DetectorWrapper):
    """Save detected image to a file"""

    def __init__(
        self,
        detector,
        save_dir: str,
        prefix: str = "image",
        suffix: str = "png",
        padding_zeros: int = 4,
        verbose: bool = False,
    ):
        super(SaveImageWrapper, self).__init__(detector)
        self.save_dir = save_dir
        self.prefix = prefix
        self.suffix = suffix
        self.verbose = verbose
        self.padding_zeros = padding_zeros

        # clean up the save_dir
        if os.path.exists(self.save_dir):
            os.system("rm -rf {}".format(self.save_dir))
        os.makedirs(self.save_dir)
        # counter for image
        self.counter = 0

    def detect(self, image):
        # detect keypoints/descriptors for a single image
        xys, desc, scores, vis_image = self.detector.detect(image)
        # save image
        self.save_image(image)
        return xys, desc, scores, vis_image

    def save_image(self, image):
        # save image
        filename = os.path.join(
            self.save_dir,
            "{}_{}.{}".format(
                self.prefix, str(self.counter).zfill(self.padding_zeros), self.suffix
            ),
        )
        cv2.imwrite(filename, image)
        if self.verbose:
            print("Save image to {}".format(filename))
        self.counter += 1


class DrawKeyPointsWrapper(DetectorWrapper):
    """Draw keypoints on image and visualize it"""

    def __init__(self, detector, window_name: str = "image", vis_height=500, show=True):
        super(DrawKeyPointsWrapper, self).__init__(detector)
        self.window_name = window_name
        self.vis_height = vis_height
        self.show = show

    def detect(self, image):
        # detect keypoints/descriptors for a single image
        xys, desc, scores, vis_image = self.detector.detect(image)
        # visualize image
        vis_image = self.vis_image(image, xys, scores)
        return xys, desc, scores, vis_image

    def vis_image(self, image, xys, scores):
        vis_image = image.copy()
        # resize image height to 500
        scale = self.vis_height / vis_image.shape[0]
        vis_image = cv2.resize(vis_image, None, fx=scale, fy=scale)
        # draw keypoints with colormap using scores
        vis_image = cv2.drawKeypoints(
            vis_image,
            [cv2.KeyPoint(x * scale, y * scale, 1) for x, y in xys],
            vis_image,
            flags=0,
        )
        # visualize image
        if self.show:
            cv2.imshow(self.window_name, vis_image)
            cv2.waitKey(0)
        return vis_image
