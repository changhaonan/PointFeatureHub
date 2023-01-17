import os
import argparse
import cv2

def mask_image(rgb_img, seg_img):
    seg_img = cv2.convertScaleAbs(seg_img)
    masked = cv2.bitwise_and(rgb_img, rgb_img, mask = seg_img)
    return masked

def get_masked_image(rgb_dir, seg_dir, masked_rgb_dir):
    
    # get image list
    files = os.listdir(rgb_dir)
    for file in files:
        if file.split('.')[-1] != 'png':
                files.remove(file)
    
    # get rgb images and seg images
    for file in files:
        rgb = os.path.join(rgb_dir, file)
        seg = os.path.join(seg_dir, file.replace("color", "seg"))
        rgb_img = cv2.imread(rgb)
        seg_img = cv2.imread(seg, -1)
        masked_img = mask_image(rgb_img, seg_img)
        masked_name = os.path.join(masked_rgb_dir, file.replace("color", "masked"))
        cv2.imwrite(masked_name, masked_img)


def main(rgb_dir):

    seg_dir = rgb_dir.replace("rgb", "seg")
    masked_rgb_dir = rgb_dir.replace("rgb", "masked_rgb")
    get_masked_image(rgb_dir, seg_dir, masked_rgb_dir)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--img_dir", help="rgb images directory")

    args = parser.parse_args()
    dir_path = args.img_dir
    main(dir_path)
    print("Masked RGB images created ...")

