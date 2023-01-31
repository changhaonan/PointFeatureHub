#! /bin/bash

while getopts d: flag
do
    case "${flag}" in
        d) img_dir=${OPTARG};;
    esac
done

### variables
scr_dir="./scr"
rgb_dir=$img_dir"/rgb"
seg_dir=$img_dir"/seg"
cam_dir=$img_dir"/cam-00"
feature_dir=$img_dir"/features"
masked_rgb_dir=$img_dir"/masked_rgb"
###

### Create working directories
echo "Image dir is: $img_dir";
mkdir -p "$rgb_dir";
echo "Created rgb img dir: $rgb_dir ...";
mkdir -p "$seg_dir";
echo "Created seg img dir: $seg_dir ...";
mkdir -p "$masked_rgb_dir";
echo "Created masked rgb image dir: $masked_rgb_dir ...";
mkdir -p "$feature_dir";
echo "Created features dir: $feature_dir ...";
cp "$cam_dir/"*.color.png "$rgb_dir";
cp "$cam_dir/"*.seg.png "$seg_dir";
echo "RGB & Seg images copied ..."

### Conda env
eval "$(conda shell.bash hook)"
### 

### Generate masked RGB images
conda activate r2d2;
echo "Conda env activated: $CONDA_DEFAULT_ENV ...";
echo "Creating masked RGB images ..."
python "$scr_dir/create_masked_rgb.py" -d "$rgb_dir";
###

### Orb feature
echo "Executing Orb inference ..."
python "$scr_dir/orb_inference.py" -i "$masked_rgb_dir"
echo "Orb inference executed ..."
###


