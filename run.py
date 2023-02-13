import hydra
import os
import cv2
import glob
from method import detector_map
from core.wrapper import DrawKeyPointsWrapper, SaveImageWrapper


@hydra.main(config_path="cfg", config_name="config")
def launch_detector_hydra(cfg):
    def create_detector_thunk(**kwargs):
        detector = detector_map[cfg.detector](cfg, **kwargs)
        if cfg.draw_keypoints:
            detector = DrawKeyPointsWrapper(detector)
            if cfg.save_image:
                detector = SaveImageWrapper(
                    detector,
                    cfg.save_dir,
                    prefix=cfg.prefix,
                    suffix=cfg.suffix,
                    padding_zeros=cfg.padding_zeros,
                    verbose=cfg.verbose,
                )
        return detector

    if cfg.task == "detect":
        detector = create_detector_thunk()
        # go over train list
        for image_file in glob.glob(os.path.join(cfg.data_dir, cfg.train_dir, "*.png")):
            image = cv2.imread(image_file)
            xys, desc, scores, vis_image = detector.detect(image)


if __name__ == "__main__":
    launch_detector_hydra()
