# Prepare
Before start training please make sure that you have installed the enviroment and downloaded the datasets.

## Training set preprocessing
To improve efficiency, we preprocess the i frames offline and during training, we directly load them.

* Download CompressAI intra models
    ```
    cd ./checkpoints
    python download_compressai_models.py
    cd ..
    ```

```
python preprocessing.py --septuplet_list /xxx/Vimeo90k/vimeo_septuplet/sep_trainlist.txt --i_frame_model_path /xxx/checkpoints/cheng2020-anchor-3-e49be189.pth.tar --quality_index 0

python preprocessing.py --septuplet_list /xxx/Vimeo90k/vimeo_septuplet/sep_trainlist.txt --i_frame_model_path /xxx/checkpoints/cheng2020-anchor-4-98b0b468.pth.tar --quality_index 1

python preprocessing.py --septuplet_list /xxx/Vimeo90k/vimeo_septuplet/sep_trainlist.txt --i_frame_model_path /xxx/checkpoints/cheng2020-anchor-5-23852949.pth.tar --quality_index 2

python preprocessing.py --septuplet_list /xxx/Vimeo90k/vimeo_septuplet/sep_trainlist.txt --i_frame_model_path /xxx/checkpoints/cheng2020-anchor-6-4c052b1a.pth.tar --quality_index 3
```

After preprocssing, the training set folder is:

    /Vimeo90k/vimeo_septuplet/
        sequences/
            00001/
            00002/
            00003/
            ...
        reference_sequences/
            0/
              00001/
              00002/
              00003/
            1/
            2/
            3/

## Test dataset

In DCVC, the spatial resolution of video needs to be cropped into integral times of 64.

The dataset format can be seen in dataset_config_example.json. 

We crop the yuv video and transform to RGB colorspace by using "crop.sh" and "yuv2rgb.sh" under the scripts folder.


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



# Train DCVC
The training consist of two stages --Pretraining and Fintuning.

For pretraining, we have four steps, each step has its own loss function and is responsible for a different part. During training, we use UVG datasets for evaluation.  DCVC pretraining only consists of 2 frames (IP)

For step 1, use quality 0 as example:
We run:
```
python train_dcvc_transform.py \
  --vimeo_dir /Vimeo90k/vimeo_septuplet/sequences \
  --precomputed_dir /Vimeo90k/vimeo_septuplet/reference_sequences \
  --septuplet_list /Vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --uvg_dir /dataset/UVG/png_sequences \
  --i_frame_model_path ./checkpoints/cheng2020-anchor-3-e49be189.pth.tar \
  --lambda_value 256 \
  --quality_index 0 \
  --stage 1 \
  --epochs 50 \
  --model_type psnr \
  --batch_size 4 \
  --lr_scheduler plateau \
  --lr_patience 3 \
  --cuda_device 0 \
  --use_ema \
  --evaluate_both
```

Then for step 2:
```
python train_dcvc_transform.py \
  --vimeo_dir /Vimeo90k/vimeo_septuplet/sequences \
  --precomputed_dir /Vimeo90k/vimeo_septuplet/reference_sequences \
  --septuplet_list /Vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --uvg_dir /dataset/UVG/png_sequences \
  --i_frame_model_path ./checkpoints/cheng2020-anchor-3-e49be189.pth.tar \
  --lambda_value 256 \
  --quality_index 0 \
  --stage 2 \
  --epochs 50 \
  --model_type psnr \
  --batch_size 4 \
  --lr_scheduler plateau \
  --lr_patience 3 \
  --cuda_device 0 \
  --use_ema \
  --evaluate_both \
  --previous_stage_checkpoint /best_ckpt_transform/model_dcvc_lambda_256.0_quality_0_stage_1_global_best.pth \
  --not_skip_test
```
And it is similarly for step 3 and 4.

Then for the finetune stage, we run:
```
python train_dcvc_transform.py \
  --vimeo_dir /Vimeo90k/vimeo_septuplet/sequences \
  --precomputed_dir /Vimeo90k/vimeo_septuplet/reference_sequences \
  --septuplet_list /Vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --uvg_dir /dataset/UVG/png_sequences \
  --i_frame_model_path ./checkpoints/cheng2020-anchor-3-e49be189.pth.tar \
  --lambda_value 256 \
  --quality_index 0 \
  --stage 2 \
  --epochs 50 \
  --model_type psnr \
  --batch_size 4 \
  --lr_scheduler plateau \
  --lr_patience 3 \
  --cuda_device 0 \
  --use_ema \
  --evaluate_both \
  --not_skip_test \
  --finetune_checkpoint /best_ckpt_transform/model_dcvc_lambda_256.0_quality_0_stage_4_global_best.pth \
  --finetune
```
For other quality levels, only need to change the corresponding hyperparameters.

An example of resume training is:
```
python train_dcvc_transform.py \
  --vimeo_dir /Vimeo90k/vimeo_septuplet/sequences \
  --precomputed_dir /Vimeo90k/vimeo_septuplet/reference_sequences \
  --septuplet_list /Vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --uvg_dir /dataset/UVG/png_sequences \
  --i_frame_model_path ./checkpoints/cheng2020-anchor-3-e49be189.pth.tar \
  --lambda_value 256 \
  --quality_index 0 \
  --stage 1 \
  --epochs 50 \
  --model_type psnr \
  --batch_size 4 \
  --lr_scheduler plateau \
  --lr_patience 3 \
  --cuda_device 0 \
  --use_ema \
  --evaluate_both \
  --resume /xxxx/model_dcvc_lambda_256.0_quality_0_stage_1_latest.pth
```


# Test DCVC
After training for all the lambda values, we could test the models.
Example of test the models:
```bash
# Quality 0
python test_video_transform.py \
    --i_frame_model_name cheng2020-anchor \
    --i_frame_model_path checkpoints/cheng2020-anchor-3-e49be189.pth.tar \
    --test_config dataset_config_example.json \
    --cuda true \
    --cuda_device 0 \
    --worker 1 \
    --output_json_result_path DCVC_transform_results_IP32_q0.json \
    --model_type psnr \
    --model_path /results/model_dcvc_lambda_256.0_quality_0_stage_4_best.pth

# Quality 1
python test_video_transform.py \
    --i_frame_model_name cheng2020-anchor \
    --i_frame_model_path checkpoints/cheng2020-anchor-4-98b0b468.pth.tar \
    --test_config dataset_config_example.json \
    --cuda true \
    --cuda_device 0 \
    --worker 1 \
    --output_json_result_path DCVC_transform_results_IP32_q1.json \
    --model_type psnr \
    --model_path /results/model_dcvc_lambda_512.0_quality_1_stage_4_best.pth

```

Besides the transform function version, we also have a CompressAI lowerbound version. 
The training and test scripts are "train_dcvc_lowerbound.py" and "test_video_lower.py". The hyperparameters are the same as the transform version. Please note that we do not test real entropy encoding.



test_video.py is for testing original DCVC:
```
python test_video.py \
    --i_frame_model_name cheng2020-anchor \
    --i_frame_model_path checkpoints/cheng2020-anchor-3-e49be189.pth.tar \
    --test_config dataset_config_example.json \
    --cuda true \
    --cuda_device 0 \
    --worker 1 \
    --output_json_result_path DCVC_results_IP32_comp_Q0.json \
    --model_type psnr \
    --model_path checkpoints/model_dcvc_quality_0_psnr.pth
```

