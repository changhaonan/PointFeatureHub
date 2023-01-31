import os
import pickle
from tqdm import tqdm
import random
from joblib import Parallel, delayed
import torch
import pickle
import os
import numpy as np
import argparse
import json
import cv2


def load_npy(npy_path):
    """Load numpy file."""
    return np.load(npy_path)


def write_xml(kp, des, score, FName):
    cv_file = cv2.FileStorage(FName, cv2.FILE_STORAGE_WRITE)
    cv_file.write("r2d2_keypoints", kp)
    cv_file.write("r2d2_descriptors", des)
    cv_file.write("r2d2_score", score)
    cv_file.release() 
    return True

def r2d2_inference(img_dir):
    feature_dir = os.path.join(img_dir, "features")
    r2d2_dir = os.path.join(feature_dir, "r2d2_results")
    sub_dirs = [x[0] for x in os.walk(r2d2_dir)][1:]
    for _, dir_name in enumerate(tqdm(sub_dirs)):
        print(dir_name)
        desc = os.path.join(dir_name, "descriptors.npy")
        kp = os.path.join(dir_name, "keypoints.npy")
        score = os.path.join(dir_name, "scores.npy")
        kp = load_npy(kp)
        desc = load_npy(desc)
        xml_name = dir_name.split("/")[-1].replace("png", "xml").replace("masked", "r2d2")
        xml_name = os.path.join(r2d2_dir, xml_name)
        try:
            write_xml(kp, desc, score, xml_name)
        except:
            continue

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="R2D2 Inference")
    parser.add_argument(
        "-i", "--image_dir",
        dest = "image_dir",
        type = str,
        default = ""
    )
    args = parser.parse_args()
    img_dir = args.image_dir
    r2d2_inference(img_dir)