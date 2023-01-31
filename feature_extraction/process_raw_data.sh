#! /bin/bash

while getopts d: flag
do
    case "${flag}" in
        d) img_dir=${OPTARG};;
    esac
done

### variables
rgb_dir=$img_dir"/rgb"
seg_dir=$img_dir"/seg"
cam_dir=$img_dir"/cam-00"
feature_dir=$img_dir"/features"
masked_rgb_dir=$img_dir"/masked_rgb"
main_dir=$(pwd)
r2d2_results_dir=$feature_dir"/r2d2_results"
r2d2_src="/home/lance/ws_Yuqiu/r2d2/extract.py"
r2d2_model="/home/lance/ws_Yuqiu/r2d2/models/r2d2_WASF_N16.pt"
img_list=$rgb_dir"/img_list.txt"
###

### Create working directories
echo "Image dir is: $img_dir";
mkdir -p "$rgb_dir";
echo "Created rgb img dir: $rgb_dir ...";
mkdir -p "$seg_dir";
echo "Created seg img dir: $seg_dir ...";
mkdir -p "$r2d2_results_dir";
echo "Created r2d2 results dir: $r2d2_results_dir ...";
mkdir -p "$masked_rgb_dir";
echo "Created masked rgb image dir: $masked_rgb_dir ...";
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
python ./create_masked_rgb.py -d "$rgb_dir";
###

# ### R2D2
# conda activate r2d2;
# echo "Conda env activated: $CONDA_DEFAULT_ENV ...";
# python ./create_imglist.py -d "$rgb_dir";
# echo "Executing R2D2 ..."
# python "$r2d2_src" --model "$r2d2_model" --images "$img_list" --top-k 5000
# echo "R2D2 model executed ..."
# mv "$rgb_dir/"*.r2d2 "$r2d2_results_dir";
# conda deactivate;
# for i in "$r2d2_results_dir/"*.r2d2
# do
# mkdir -p "${i/.r2d2//}"
# unzip "$i" -d "${i/.r2d2//}"
# echo "Unzipped to ${i/.r2d2//}"
# done
# rm "$r2d2_results_dir/"*.r2d2
# echo "Unzip r2d2 results done ..."
# ###

# ### SuperPoint Masked RGB Image Case Study
# conda activate airobj;
# echo "Conda env activated: $CONDA_DEFAULT_ENV ...";
# cd "/home/lance/ws_Yuqiu/AirObject"
# airobj_scr="/home/lance/ws_Yuqiu/AirObject/tracking/superpoint_inference.py"
# airobj_cfg="/home/lance/ws_Yuqiu/AirObject/config/tracking/superpoint_inference.yaml"
# echo "Executing SuperPoint ..."
# for thr in 0.2 0.15 0.1 0.05
# do
# python "$airobj_scr" -c "$airobj_cfg" -i "$img_dir" -t $thr -g 1 -m 1;
# echo "SuperPoint model - thresh $thr executed ..."
# mv "$img_dir""/sp_0" "$img_dir/sp_$thr"
# done
# conda deactivate;
# ###

### Draw SuperPoint results
cd "$main_dir";
conda activate airobj;
echo "Conda env activated: $CONDA_DEFAULT_ENV ...";
python ./draw_sp.py -d "$masked_rgb_dir"
echo "Draw superpoint results ..."
conda deactivate;
###



# ### SuperPoint
# conda activate airobj;
# echo "Conda env activated: $CONDA_DEFAULT_ENV ...";
# cd "/home/lance/ws_Yuqiu/AirObject"
# airobj_scr="/home/lance/ws_Yuqiu/AirObject/tracking/superpoint_inference.py"
# airobj_cfg="/home/lance/ws_Yuqiu/AirObject/config/tracking/superpoint_inference.yaml"
# echo "Executing SuperPoint ..."
# python "$airobj_scr" -c "$airobj_cfg" -i "$img_dir" -g 1;
# echo "SuperPoint model executed ..."
# conda deactivate;
# ###

# ### Combine SuperPoint & R2D2
# mkdir -p "$feature_dir"
# cd "$main_dir";
# conda activate r2d2;
# echo "Conda env activated: $CONDA_DEFAULT_ENV ...";
# python combine.py -d "$img_dir";
# echo "Data process done ..."
# ###
