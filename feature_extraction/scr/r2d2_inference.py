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


def write_xml_r2d2(kp1, des1, FName):

    pt = []
    size = []
    angle = []
    response = []
    octave = []
    class_id = []
    for kp in kp1:
        pt.append(list(kp.pt))
        size.append(kp.size)
        angle.append(kp.angle)
        response.append(kp.response)
        octave.append(kp.octave)
        class_id.append(kp.class_id)
    pt = np.asarray(pt, dtype=np.float32)
    size = np.asarray(size, dtype=np.float32)
    angle = np.asarray(angle, dtype=np.float32)
    response = np.asarray(response, dtype=np.float32)
    octave = np.asarray(octave, dtype=np.float32)
    class_id = np.asarray(class_id, dtype=np.float32)

    cv_file = cv2.FileStorage(FName, cv2.FILE_STORAGE_WRITE)
    cv_file.write("points", pt)
    cv_file.write("size", size)
    cv_file.write("angle", angle)
    cv_file.write("response", response)
    cv_file.write("octave", octave)
    cv_file.write("class_id", class_id)
    cv_file.write("descriptions", np.asarray(des1, dtype=np.float32))
    cv_file.release()

def orb_inference(img_dir):
    r2d2_dir = os.path.join(img_dir, "r2d2_results")
    feature_dir = os.path.join(img_dir, "features")
    sub_dirs = [x[0] for x in os.walk(r2d2_dir)][1:]
    for dir_name in sub_dirs:
        desc = os.path.join(dir_name, "descriptors.npy")
        kp = os.path.join(dir_name, "keypoints.npy")
        kp = load_npy(kp)
        desc = load_npy(desc)
        pkl_name = dir_name.split("/")[-1].replace("png", "pkl")
        # json_name = dir_name.split("/")[-1].replace("png", "json").replace("color", "kp")
        xml_name = dir_name.split("/")[-1].replace("png", "xml").replace("color", "feature")
        r2d2_dict = {"points":kp[:, :], "point_descs":desc}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SuperPoint Inference")
    parser.add_argument(
        "-i", "--image_dir",
        dest = "image_dir",
        type = str,
        default = ""
    )
    args = parser.parse_args()
    img_dir = args.image_dir
    orb_inference(img_dir)