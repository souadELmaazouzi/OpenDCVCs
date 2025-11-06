# Prepare
Before start training please make sure that you have installed the enviroment and downloaded the datasets.

## Training set preprocessing
To improve efficiency, we preprocess the i frames offline and during training, we directly load them.

Intra models should be downloaded from the official DCVC repo.

```
python preprocessing.py --i_frame_model_path /checkpoints/intra_psnr_TMM_q1.pth.tar --quality_index 1

python preprocessing.py --i_frame_model_path /checkpoints/intra_psnr_TMM_q2.pth.tar --quality_index 2 

python preprocessing.py --i_frame_model_path /checkpoints/intra_psnr_TMM_q3.pth.tar --quality_index 3 

python preprocessing.py --i_frame_model_path /checkpoints/intra_psnr_TMM_q4.pth.tar --quality_index 4
```

After preprocssing, the training set folder is:

    /Vimeo90k/vimeo_septuplet/
        sequences/
            00001/
            00002/
            00003/
            ...
        reference_sequences_DCVC_TCM/
            1/
              00001/
              00002/
              00003/
            2/
            3/
            4/

## Test dataset

In DCVC-TCM, the spatial resolution of video needs to be cropped into integral times of 64.

The dataset format can be seen in dataset_config_example.json. 

We crop the yuv video and transform to RGB colorspace by using "crop.sh" and "yuv2rgb.sh" under the DCVC/scripts folder.


At last, the folder structure of the dataset is like:

    /HEVC_B/png_sequences/
        BQTerrace_1920x1024_60/
            - im00001.png
            - im00002.png
            - im00003.png
            - ...
        BasketballDrive_1920x1024_50/
            - im00001.png
            - im00002.png
            - im00003.png
            - ...
        ...
    ...
And it is similar for other test sets.



# Train DCVC-TCM
The training consist of two stages --Pretraining and Fintuning.

For pretraining, we have four steps, each step has its own loss function and is responsible for a different part. During training, we use UVG datasets for evaluation.

For step 1, use quality 0 as example:
We run:
```
python train_dcvc_tcm_transform.py \
  --vimeo_dir /Vimeo90k/vimeo_septuplet/sequences \
  --precomputed_dir /Vimeo90k/vimeo_septuplet/reference_sequences_DCVC_TCM \
  --septuplet_list /Vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --uvg_dir /dataset/UVG/png_sequences \
  --i_frame_model_path ./checkpoints/intra_psnr_TMM_q1.pth.tar \
  --lambda_value 256 \
  --quality_index 1 \
  --stage 1 \
  --epochs 50 \
  --model_type psnr \
  --batch_size 4 \
  --lr_scheduler plateau \
  --lr_patience 3 \
  --cuda_device 0 \
  --use_ema \
  --evaluate_both \
  --warmup
```

Then for step 2:
```
python train_dcvc_tcm_transform.py \
  --vimeo_dir /Vimeo90k/vimeo_septuplet/sequences \
  --precomputed_dir /Vimeo90k/vimeo_septuplet/reference_sequences_DCVC_TCM \
  --septuplet_list /Vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --uvg_dir /dataset/UVG/png_sequences \
  --i_frame_model_path ./checkpoints/intra_psnr_TMM_q1.pth.tar \
  --lambda_value 256 \
  --quality_index 1 \
  --stage 2 \
  --epochs 50 \
  --model_type psnr \
  --batch_size 4 \
  --lr_scheduler plateau \
  --lr_patience 3 \
  --cuda_device 0 \
  --use_ema \
  --evaluate_both
  --previous_stage_checkpoint /best_ckpt_transform/xxxxx.pth \
  --not_skip_test \
  --warmup
```
And it is similarly for steps 3 and 4. During the first few epochs of steps 1,2. It is recommended to add "--warmup" to ensure stable training.

Then for the finetune stage, we run:
```
python train_dcvc_tcm_transform.py \
  --vimeo_dir /Vimeo90k/vimeo_septuplet/sequences \
  --precomputed_dir /Vimeo90k/vimeo_septuplet/reference_sequences_DCVC_TCM \
  --septuplet_list /Vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --uvg_dir /UVG/png_sequences \
  --i_frame_model_path ./checkpoints/intra_psnr_TMM_q2.pth.tar \
  --lambda_value 256 \
  --quality_index 1 \
  --stage 4 \
  --epochs 50 \
  --model_type psnr \
  --batch_size 4 \
  --lr_scheduler plateau \
  --lr_patience 3 \
  --cuda_device 0 \
  --use_ema \
  --evaluate_both \
  --finetune_checkpoint /best_ckpt_transform/model_dcvc_lambda_512.0_quality_2_stage_4_global_best.pth \
  --finetune \
  --not_skip_test
```
For other quality levels, only need to change the corresponding hyperparameters.

An example of resume training is:
```
python train_dcvc_tcm_transform.py \
  --vimeo_dir /Vimeo90k/vimeo_septuplet/sequences \
  --precomputed_dir /Vimeo90k/vimeo_septuplet/reference_sequences_DCVC_TCM \
  --septuplet_list /Vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --uvg_dir /UVG/png_sequences \
  --i_frame_model_path ./checkpoints/intra_psnr_TMM_q1.pth.tar \
  --lambda_value 256 \
  --quality_index 1 \
  --stage 4 \
  --epochs 50 \
  --model_type psnr \
  --batch_size 4 \
  --lr_scheduler plateau \
  --lr_patience 3 \
  --cuda_device 0 \
  --use_ema \
  --evaluate_both \
  --resume /results/checkpoints_transform/model_dcvc_lambda_256.0_quality_1_stage_1_latest.pth
```


# Test DCVC-TCM
After training for all the lambda values, we could test the models.
Example of test the models:
```bash
python test_video_transform.py --i_frame_model_name IntraNoAR --i_frame_model_path ./checkpoints/intra_psnr_TMM_q1.pth.tar --model_path /model_dcvc_lambda_256.0_quality_1_stage_4_global_best.pth --test_config recommended_test_full_results_IP32.json --cuda 1 -w 1 --write_stream 0 --output_path results_IP32_transform_Q1.json

python test_video_transform.py --i_frame_model_name IntraNoAR --i_frame_model_path ./checkpoints/intra_psnr_TMM_q2.pth.tar --model_path /model_dcvc_lambda_512.0_quality_2_stage_4_global_best.pth --test_config recommended_test_full_results_IP32.json --cuda 1 -w 1 --write_stream 0 --output_path results_IP32_transform_Q2.json

python test_video_transform.py --i_frame_model_name IntraNoAR --i_frame_model_path ./checkpoints/intra_psnr_TMM_q3.pth.tar --model_path /model_dcvc_lambda_1024.0_quality_3_stage_4_global_best.pth --test_config recommended_test_full_results_IP32.json --cuda 1 -w 1 --write_stream 0 --output_path results_IP32_transform_Q3.json

python test_video_transform.py --i_frame_model_name IntraNoAR --i_frame_model_path ./checkpoints/intra_psnr_TMM_q4.pth.tar --model_path /model_dcvc_lambda_2048.0_quality_4_stage_4_global_best.pth --test_config recommended_test_full_results_IP32.json --cuda 1 -w 1 --write_stream 0 --output_path results_IP32_transform_Q4.json

```

Besides the transform function version, we also have a CompressAI lowerbound version. 
The training and test scripts are "train_dcvc_tcm_lower.py" and "test_video_lower.py". The hyperparameters are the same as the transform version.  Please note that we do not test real entropy encoding.



test_video.py is for testing original DCVC:
```
python test_video.py --i_frame_model_name IntraNoAR --i_frame_model_path ./checkpoints/intra_psnr_TMM_q1.pth.tar --model_path ./checkpoints/inter_psnr_TMM_q1.pth --test_config recommended_test_full_results_IP32.json --cuda 1 -w 1 --write_stream 0 --output_path results_IP32_comp_Q1.json
```

