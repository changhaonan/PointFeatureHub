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
superpoint_results_dir=$feature_dir"/superpoint_results"
superpoint_ws="../external/SuperGluePretrainedNetwork"
###

### Create working directories
echo "Image dir is: $img_dir";
mkdir -p "$rgb_dir";
echo "Created rgb img dir: $rgb_dir ...";
mkdir -p "$seg_dir";
echo "Created feature dir: $feature_dir ...";
mkdir -p "$feature_dir";
echo "Created seg img dir: $seg_dir ...";
mkdir -p "$superpoint_results_dir";
echo "Created superpoint results dir: $superpoint_results_dir ...";
mkdir -p "$masked_rgb_dir";
echo "Created masked rgb image dir: $masked_rgb_dir ...";
cp "$cam_dir/"*.color.png "$rgb_dir";
cp "$cam_dir/"*.seg.png "$seg_dir";
echo "RGB & Seg images copied ..."

### Conda env
eval "$(conda shell.bash hook)"
### 

### Generate masked RGB images
conda activate py36-superglue;
echo "Conda env activated: $CONDA_DEFAULT_ENV ...";
echo "Creating masked RGB images ..."
python "$scr_dir/create_masked_rgb.py" -d "$rgb_dir";
###

### Superpoint
conda activate py36-superglue;
echo "Conda env activated: $CONDA_DEFAULT_ENV ...";
echo "Executing Superpoint ..."
echo "$superpoint_ws"
python $superpoint_ws"/superpoint_inference.py" --input "$masked_rgb_dir" --output_dir "$superpoint_results_dir"
echo "Superpoint model executed ..."
conda deactivate;


