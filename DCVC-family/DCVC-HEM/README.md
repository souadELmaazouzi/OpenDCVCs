# Prepare
Before start training please make sure that you have installed the enviroment and downloaded the datasets.

## Training set preprocessing
To improve efficiency, we preprocess the i frames offline and during training, we directly load them.

Intra models should be downloaded from the official DCVC repo.

```
python preprocessing.py --i_frame_model_path /checkpoints/acmmm2022_image_psnr.pth.tar --quality_index 0
python preprocessing.py --i_frame_model_path /checkpoints/acmmm2022_image_psnr.pth.tar --quality_index 1
python preprocessing.py --i_frame_model_path /checkpoints/acmmm2022_image_psnr.pth.tar --quality_index 2
python preprocessing.py --i_frame_model_path /checkpoints/acmmm2022_image_psnr.pth.tar --quality_index 3
```

After preprocssing, the training set folder is:

    /Vimeo90k/vimeo_septuplet/
        sequences/
            00001/
            00002/
            00003/
            ...
        reference_sequences_DCVC_HEM/
            0/
              00001/
              00002/
              00003/
            1/
            2/
            3/

## Test dataset

In DCVC-HEM, the spatial resolution of video needs to be cropped into integral times of 64.

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



# Train DCVC-HEM
The training consist of two stages --Pretraining and Fintuning.

For pretraining, we have four steps, each step has its own loss function and is responsible for a different part. During training, we use UVG datasets for evaluation.

For step 1:
We run:
```
python train_dcvc_hem_transform.py \
  --vimeo_dir /Vimeo90k/vimeo_septuplet/sequences \
  --precomputed_dir /Vimeo90k/vimeo_septuplet/reference_sequences_DCVC_HEM \
  --septuplet_list /Vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --uvg_dir /UVG/png_sequences \
  --i_frame_model_path ./checkpoints/acmmm2022_image_psnr.pth.tar \
  --stage 1 \
  --epochs 50 \
  --model_type psnr \
  --batch_size 4 \
  --lr_scheduler plateau \
  --lr_patience 3 \
  --cuda_device 0 \
  --use_ema \
  --evaluate_both \
  --single_lambda \
  --warmup
```

Then for step 2:
```
python train_dcvc_hem_transform.py \
  --vimeo_dir /Vimeo90k/vimeo_septuplet/sequences \
  --precomputed_dir /Vimeo90k/vimeo_septuplet/reference_sequences_DCVC_HEM \
  --septuplet_list /Vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --uvg_dir /UVG/png_sequences \
  --i_frame_model_path ./checkpoints/acmmm2022_image_psnr.pth.tar \
  --stage 2 \
  --epochs 50 \
  --model_type psnr \
  --batch_size 4 \
  --lr_scheduler plateau \
  --lr_patience 3 \
  --cuda_device 0 \
  --use_ema \
  --evaluate_both \
  --single_lambda
  --previous_stage_checkpoint /best_ckpt_transform/xxxxx.pth \
  --not_skip_test \
  --warmup
```
And it is similarly for steps 3 and 4. During the first few epochs of steps 1,2, it is recommended to add "--warmup" to ensure stable training.

Then for the finetune stage, we run:
```
python train_dcvc_hem_transform.py \
  --vimeo_dir /Vimeo90k/vimeo_septuplet/sequences \
  --precomputed_dir /Vimeo90k/vimeo_septuplet/reference_sequences_DCVC_HEM \
  --septuplet_list /Vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --uvg_dir /UVG/png_sequences \
  --i_frame_model_path ./checkpoints/acmmm2022_image_psnr.pth.tar \
  --stage 4 \
  --epochs 50 \
  --model_type psnr \
  --batch_size 4 \
  --lr_scheduler plateau \
  --lr_patience 3 \
  --cuda_device 0 \
  --use_ema \
  --evaluate_both \
  --single_lambda
  --finetune_checkpoint /best_ckpt_transform/model_dcvc_lambda_512.0_quality_2_stage_4_global_best.pth \
  --finetune \
  --not_skip_test
```
For other quality levels, only need to change the corresponding hyperparameters.

An example of resume training is:
```
python train_dcvc_hem_transform.py \
  --vimeo_dir /Vimeo90k/vimeo_septuplet/sequences \
  --precomputed_dir /Vimeo90k/vimeo_septuplet/reference_sequences_DCVC_HEM \
  --septuplet_list /Vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --uvg_dir /UVG/png_sequences \
  --i_frame_model_path ./checkpoints/acmmm2022_image_psnr.pth.tar \
  --stage 1 \
  --epochs 50 \
  --model_type psnr \
  --batch_size 4 \
  --lr_scheduler plateau \
  --lr_patience 3 \
  --cuda_device 0 \
  --use_ema \
  --evaluate_both \
  --single_lambda
  --resume /results/checkpoints_transform/model_dcvc_lambda_256.0_quality_1_stage_1_latest.pth
```


# Test DCVC-HEM
After training for all the lambda values, we could test the models.
Example of test the models:
```bash
python test_video_transform.py --i_frame_model_path ./checkpoints/acmmm2022_image_psnr.pth.tar --model_path /checkpoints/xxxx.pth.tar --rate_num 4 --test_config ./dataset_config_example.json --cuda 1 -w 1 --write_stream 0 --output_path test_result.json --force_intra_period 32 --force_frame_num 96


```

Besides the transform function version, we also have a CompressAI lowerbound version. 
The training and test scripts are "train_dcvc_hem.py" and "test_video_lower.py". The hyperparameters are the same as the transform version.



test_video.py is for testing original DCVC-HEM:
```
python test_video.py --i_frame_model_path ./checkpoints/acmmm2022_image_psnr.pth.tar --model_path ./checkpoints/acmmm2022_video_psnr.pth.tar --rate_num 4 --test_config ./dataset_config_example.json --cuda 1 -w 1 --write_stream 0 --output_path test_result.json --force_intra_period 32 --force_frame_num 96

```

