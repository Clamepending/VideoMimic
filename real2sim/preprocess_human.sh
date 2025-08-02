#!/bin/bash

# Usage: ./preprocess_human.sh <name> [<vis_flag; 1 or 0>] 
# Ex) vis: bash preprocess_human.sh fourleghuman_tutorial1_subset 1 
# Ex) no vis: bash preprocess_human.sh fourleghuman_tutorial1_subset 0 


# Input arguments
NAME=$1

# Determine if --vis should be included
VIS_FLAG=""
if [ "$2" == "1" ]; then
    VIS_FLAG="--vis"
fi

# Extract directory and camera name
BASE_DIR="demo_data"
VIDEO_DIR="$BASE_DIR/input_images/$NAME"
CAM_DIR="$VIDEO_DIR/cam01"
MASKS_DIR="$BASE_DIR/input_masks/$NAME/cam01"

echo "Running Grounding-SAM-2..."
# CMD="python stage0_preprocessing/sam2_segmentation.py --video-dir \"$CAM_DIR\" --output-dir \"$MASKS_DIR\" $VIS_FLAG"
CMD="python stage0_preprocessing/sam2_segmentation.py --video-dir \"$CAM_DIR\" --output-dir \"$MASKS_DIR\" --vis"
echo "$CMD"
eval "$CMD"

echo -e "\nRunning ViTPose..."
CMD="python stage0_preprocessing/vitpose_2d_poses.py --video-dir \"$CAM_DIR\" --bbox-dir \"$MASKS_DIR/json_data\" --output-dir \"$BASE_DIR/input_2d_poses/$NAME/cam01\" $VIS_FLAG"
echo "$CMD"
eval "$CMD"

echo -e "\nRunning VIMO..."
CMD="python stage0_preprocessing/vimo_3d_mesh.py --img-dir \"$CAM_DIR\" --mask-dir \"$MASKS_DIR\" --out-dir \"$BASE_DIR/input_3d_meshes/$NAME/cam01\""
echo "$CMD"
eval "$CMD"

echo -e "\nRunning BSTRO..."
CMD="python stage0_preprocessing/bstro_contact_detection.py --video-dir \"$CAM_DIR\" --bbox-dir \"$MASKS_DIR/json_data\" --output-dir \"$BASE_DIR/input_contacts/$NAME/cam01\"  --feet-contact-ratio-thr 0.2 --contact-thr 0.95"
echo "$CMD"
eval "$CMD"

# Step 1.5: Run Wilor for hand detection
echo "Step 1.5: Running Wilor for hand detection..."
echo "python stage0_preprocessing/wilor_hand_poses.py \
    --img_dir \"$BASE_DIR/input_images/$NAME/cam01\" \
    --output_dir \"$BASE_DIR/mano/$NAME/cam01\" \
    --pose2d_dir \"$BASE_DIR/input_2d_poses/$NAME/cam01\" \
    --batch_size 2 \
    --person_ids 1 \
    --hand_bbox_thr 0.8 \
    --vis"
python stage0_preprocessing/wilor_hand_poses.py \
    --img_dir "$BASE_DIR/input_images/$NAME/cam01" \
    --output_dir "$BASE_DIR/mano/$NAME/cam01" \
    --pose2d_dir "$BASE_DIR/input_2d_poses/$NAME/cam01" \
    --batch_size 2 \
    --person_ids 1 \
    --hand_bbox_thr 0.8 \
    --vis

# Step 1.75: Combine SMPL and MANO to create SMPLX
echo "Step 1.75: Running smpl_to_smplx_conversion.py..."
echo "python stage0_preprocessing/smpl_to_smplx_conversion.py \
    --smpl-dir \"$BASE_DIR/input_3d_meshes/$NAME/cam01\" \
    --mano-dir \"$BASE_DIR/mano/$NAME/cam01\" \
    --output-dir \"$BASE_DIR/input_smplx/$NAME/cam01\" \
    $VIS_FLAG"
python stage0_preprocessing/smpl_to_smplx_conversion.py \
    --smpl-dir "$BASE_DIR/input_3d_meshes/$NAME/cam01" \
    --mano-dir "$BASE_DIR/mano/$NAME/cam01" \
    --output-dir "$BASE_DIR/input_smplx/$NAME/cam01" \
    $VIS_FLAG


echo -e "\nDone!"
