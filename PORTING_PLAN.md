# Plan for Porting Hand Support (SMPLX) to VideoMimic

This document outlines the steps to port the hand support functionality from the `videomimic_hand_test` repository to the `VideoMimic` repository.

## 1. New Files to be Added

We will add the following new files to the `VideoMimic` repository:

-   `VideoMimic/process_video_smplx.sh`: A new pipeline script that orchestrates the entire process with hand support. This will be a modified version of `real2sim/process_video.sh` that includes steps for hand detection and SMPLX model processing.
-   `VideoMimic/real2sim/stage1_reconstruction/get_mano_wilor.py`: This script is responsible for running hand detection (MANO) on the input video frames. It will be copied from `videomimic_hand_test/get_mano_wilor.py`.
-   `VideoMimic/real2sim/stage2_optimization/mano_smpl_to_smplx.py`: This script combines the SMPL body pose and MANO hand poses into a full SMPLX model. It will be copied from `videomimic_hand_test/mano_smpl_to_smplx.py`.

## 2. Existing Files to be Modified

The following files in `VideoMimic` will be modified to support SMPLX:

-   `VideoMimic/real2sim/stage2_optimization/megahunter_optimization.py`: This is the core optimization script. It will be updated to:
    -   Accept an `--smplx-dir` argument instead of `--smpl-dir`.
    -   Load SMPLX data instead of SMPL data.
    -   Incorporate hand pose optimization into the main optimization loop.
    -   The logic will be based on `videomimic_hand_test/align_world_env_jax_apr18_SMPLX.py`.

-   `VideoMimic/real2sim/stage3_postprocessing/postprocessing_pipeline.py`: The post-processing script will be updated to handle the output from the SMPLX-enabled optimization. This includes saving the SMPLX parameters correctly.

## 3. High-Level Steps

1.  **Create `VideoMimic/PORTING_PLAN.md`**: The plan you are reading now.
2.  **Create `VideoMimic/process_video_smplx.sh`**: This script will be created by duplicating `VideoMimic/real2sim/process_video.sh` and adding the following steps:
    -   Run `get_mano_wilor.py` after preprocessing.
    -   Run `mano_smpl_to_smplx.py` to generate SMPLX data.
    -   Call `megahunter_optimization.py` with the new `--smplx-dir` argument.
3.  **Copy Scripts**:
    -   Copy `videomimic_hand_test/get_mano_wilor.py` to `VideoMimic/real2sim/stage1_reconstruction/`.
    -   Copy `videomimic_hand_test/mano_smpl_to_smplx.py` to `VideoMimic/real2sim/stage2_optimization/`.
4.  **Modify Python Scripts**:
    -   Update `megahunter_optimization.py` to handle SMPLX models and hand pose optimization.
    -   Update `postprocessing_pipeline.py` to correctly process and save the SMPLX output.

This plan ensures a systematic porting of the hand support feature, maintaining the modular structure of the `VideoMimic` pipeline. 