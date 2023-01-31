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

def write_xml_orb(kp1, des1, FName):

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

def orb_detect(img):
    orb = cv2.ORB_create()
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des
    
def orb_inference(img_dir):
    img_list = os.listdir(img_dir)
    img_list.sort()
    orb_dir = img_dir.replace("masked_rgb", "features")
    orb_dir = os.path.join(orb_dir, "orb_results")
    if not os.path.exists(orb_dir):
        os.mkdir(orb_dir)
    for i in tqdm(range(len(img_list))):
        img_name = img_list[i]
        if img_name.endswith('png'):
            FName = os.path.join(orb_dir, img_name.replace("masked", "orb").replace("png", "xml"))
            img = cv2.imread(os.path.join(img_dir, img_name))
            kp, des = orb_detect(img)
            write_xml_orb(kp, des, FName)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ORB Inference")
    parser.add_argument(
        "-i", "--image_dir",
        dest = "image_dir",
        type = str,
        default = ""
    )
    args = parser.parse_args()
    img_dir = args.image_dir
    orb_inference(img_dir)