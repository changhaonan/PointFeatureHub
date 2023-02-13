import hydra
import os
import cv2
import glob
from method import detector_map, matcher_map
from core.wrapper import (
    DrawKeyPointsDetectorWrapper,
    SaveImageDetectorWrapper,
    DrawKeyPointsMatcherWrapper,
    SaveImageMatcherWrapper,
)


@hydra.main(config_path="cfg", config_name="config")
def launch_detector_hydra(cfg):
    def create_detector_thunk(**kwargs):
        detector = detector_map[cfg.detector](cfg, cfg.device, **kwargs)
        if cfg.draw_keypoints:
            window_name = f"{cfg.task}:{cfg.detector}"
            detector = DrawKeyPointsDetectorWrapper(detector, window_name=window_name)
            if cfg.save_image:
                detector = SaveImageDetectorWrapper(
                    detector,
                    cfg.save_dir,
                    prefix=cfg.prefix,
                    suffix=cfg.suffix,
                    padding_zeros=cfg.padding_zeros,
                    verbose=cfg.verbose,
                )
        return detector

    def create_matcher_thunk(**kwargs):
        matcher = matcher_map[cfg.matcher](cfg, cfg.device, **kwargs)
        if cfg.draw_matches:
            window_name = f"{cfg.task}:{cfg.detector}+{cfg.matcher}"
            matcher = DrawKeyPointsMatcherWrapper(matcher, window_name=window_name)
            if cfg.save_image:
                matcher = SaveImageMatcherWrapper(
                    matcher,
                    cfg.save_dir,
                    prefix=cfg.prefix,
                    suffix=cfg.suffix,
                    padding_zeros=cfg.padding_zeros,
                    verbose=cfg.verbose,
                )
        return matcher

    if cfg.task == "detect":
        detector = create_detector_thunk()
        # go over train list
        for image_file in glob.glob(os.path.join(cfg.data_dir, cfg.train_dir, "*.png")):
            image = cv2.imread(image_file)
            xys, desc, scores, vis_image = detector.detect(image)
    elif cfg.task == "match":
        detector = create_detector_thunk()
        matcher = create_matcher_thunk()
        # go over train list
        image_prev = None
        xys_prev = None
        desc_prev = None
        scores_prev = None
        for image_file in glob.glob(os.path.join(cfg.data_dir, cfg.train_dir, "*.png")):
            image = cv2.imread(image_file)
            xys, desc, scores, vis_image = detector.detect(image)
            if image_prev is not None:
                matcher.match(
                    image_prev,
                    image,
                    xys_prev,
                    xys,
                    desc_prev,
                    desc,
                    scores_prev,
                    scores,
                )
            image_prev = image
            xys_prev = xys
            desc_prev = desc
            scores_prev = scores


if __name__ == "__main__":
    launch_detector_hydra()
