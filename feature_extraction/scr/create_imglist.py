### Doc
# This function will create an image list [xxxx0.jpg, xxxx1.jpg ...]
# for executing R2D2
###


import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--img_dir", help="rgb images directory")

args = parser.parse_args()
dir_path = args.img_dir
filelist = os.listdir(dir_path)
for filename in filelist[:]:
    if not (filename.endswith(".png")):
        filelist.remove(filename)

with open(os.path.join(dir_path, "img_list.txt"), 'w') as fp:
    for name in filelist:
        abs_name = os.path.join(dir_path, name)
        fp.write("%s\n" % abs_name)

print("created image list...")