# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import concurrent.futures
import json
import multiprocessing
import time

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from src.models.video_model_ori import DMC
from src.models.image_model import IntraNoAR
from src.utils.common import str2bool, interpolate_log, create_folder, generate_log_json, dump_json
from src.utils.stream_helper import get_padding_size, get_state_dict
from src.utils.png_reader import PNGReader
from tqdm import tqdm
from pytorch_msssim import ms_ssim
import psutil
import gc


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script with performance metrics")

    parser.add_argument('--i_frame_model_path', type=str)
    parser.add_argument('--i_frame_q_scales', type=float, nargs="+")
    parser.add_argument("--force_intra", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument('--model_path',  type=str)
    parser.add_argument('--p_frame_y_q_scales', type=float, nargs="+")
    parser.add_argument('--p_frame_mv_y_q_scales', type=float, nargs="+")
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--force_root_path', type=str, default=None, required=False)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--cuda_device", default=None,
                        help="the cuda device used, e.g., 0; 0,1; 1,2,3; etc.")
    parser.add_argument('--write_stream', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--decoded_frame_path', type=str, default='decoded_frames')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    return args


def read_image_to_torch(path):
    input_image = Image.open(path).convert('RGB')
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0)/255
    return input_image


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def save_torch_image(img, save_path):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


def count_parameters(model):
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_info(device):
    """Get GPU memory usage information"""
    device_obj = torch.device(device) if isinstance(device, str) else device
    if device_obj.type == 'cuda':
        try:
            # Set the device as current for memory queries
            current_device = torch.cuda.current_device()
            if device_obj.index != current_device:
                torch.cuda.set_device(device_obj)
            
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'cached_mb': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                'max_cached_mb': torch.cuda.max_memory_reserved() / 1024**2
            }
        except Exception as e:
            print(f"Warning: Could not get CUDA memory info: {e}")
            return {
                'allocated_mb': 0,
                'cached_mb': 0,
                'max_allocated_mb': 0,
                'max_cached_mb': 0
            }
    else:
        # For CPU, use system memory
        memory_info = psutil.virtual_memory()
        return {
            'allocated_mb': memory_info.used / 1024**2,
            'cached_mb': 0,
            'max_allocated_mb': memory_info.used / 1024**2,
            'max_cached_mb': 0
        }


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-serializable types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # Handle torch tensors and similar
        return obj.item()
    else:
        return obj


def run_test(video_net, i_frame_net, args, device):
    frame_num = args['frame_num']
    gop_size = args['gop_size']
    write_stream = 'write_stream' in args and args['write_stream']
    save_decoded_frame = 'save_decoded_frame' in args and args['save_decoded_frame']
    verbose = args['verbose'] if 'verbose' in args else 0

    if args['src_type'] == 'png':
        src_reader = PNGReader(args['img_path'], args['src_width'], args['src_height'])

    frame_types = []
    psnrs = []
    msssims = []
    bits = []
    frame_pixel_num = 0
    frame_encode_times = []

    # Performance metrics
    performance_metrics = {}
    
    # Count parameters
    if i_frame_net is not None:
        i_frame_params = count_parameters(i_frame_net)
        performance_metrics['i_frame_model_params'] = i_frame_params
    else:
        i_frame_params = 0
        performance_metrics['i_frame_model_params'] = 0
        
    if video_net is not None:
        video_params = count_parameters(video_net)
        performance_metrics['video_model_params'] = video_params
    else:
        video_params = 0
        performance_metrics['video_model_params'] = 0
        
    performance_metrics['total_params'] = i_frame_params + video_params

    start_time = time.time()
    p_frame_number = 0
    overall_p_decoding_time = 0
    encoding_start_time = time.time()
    
    with torch.no_grad():
        for frame_idx in range(frame_num):
            frame_start_time = time.time()
            rgb = src_reader.read_one_frame(src_format="rgb")
            x = np_image_to_tensor(rgb)
            x = x.to(device)
            pic_height = x.shape[2]
            pic_width = x.shape[3]

            if frame_pixel_num == 0:
                frame_pixel_num = x.shape[2] * x.shape[3]
            else:
                assert frame_pixel_num == x.shape[2] * x.shape[3]

            # pad if necessary
            padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width)
            x_padded = torch.nn.functional.pad(
                x,
                (padding_l, padding_r, padding_t, padding_b),
                mode="constant",
                value=0,
            )

            bin_path = os.path.join(args['bin_folder'], f"{frame_idx}.bin") \
                if write_stream else None

            if frame_idx % gop_size == 0:
                result = i_frame_net.encode_decode(x_padded, args['i_frame_q_scale'], bin_path,
                                                   pic_height=pic_height, pic_width=pic_width)
                dpb = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                recon_frame = result["x_hat"]
                frame_types.append(0)
                bits.append(result["bit"])
            else:
                result = video_net.encode_decode(x_padded, dpb, bin_path,
                                                 pic_height=pic_height, pic_width=pic_width,
                                                 mv_y_q_scale=args['p_frame_mv_y_q_scale'],
                                                 y_q_scale=args['p_frame_y_q_scale'])
                dpb = result["dpb"]
                recon_frame = dpb["ref_frame"]
                frame_types.append(1)
                bits.append(result['bit'])
                p_frame_number += 1
                overall_p_decoding_time += result['decoding_time']

            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
            psnr = PSNR(x_hat, x)
            msssim = ms_ssim(x_hat, x, data_range=1).item()
            psnrs.append(psnr)
            msssims.append(msssim)
            
            frame_end_time = time.time()
            frame_encode_time = frame_end_time - frame_start_time
            frame_encode_times.append(frame_encode_time)

            if verbose >= 2:
                print(f"frame {frame_idx}, {frame_encode_time:.3f} seconds,",
                      f"bits: {bits[-1]:.3f}, PSNR: {psnrs[-1]:.4f}, MS-SSIM: {msssims[-1]:.4f} ")

            if save_decoded_frame:
                save_path = os.path.join(args['decoded_frame_folder'], f'{frame_idx}.png')
                save_torch_image(x_hat, save_path)

    test_time = time.time() - start_time
    encoding_time = time.time() - encoding_start_time
    
    # Calculate timing metrics
    performance_metrics['total_encoding_time_sec'] = encoding_time
    performance_metrics['avg_frame_encode_time_sec'] = np.mean(frame_encode_times)
    performance_metrics['fps'] = frame_num / encoding_time
    
    # Get memory usage information
    memory_info = get_gpu_memory_info(device)
    performance_metrics.update({f'memory_{k}': v for k, v in memory_info.items()})
    
    if verbose >= 1 and p_frame_number > 0:
        print(f"decoding {p_frame_number} P frames, "
              f"average {overall_p_decoding_time/p_frame_number * 1000:.0f} ms.")

    # Separate timing for I and P frames
    i_frame_times = []
    p_frame_times = []
    
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            i_frame_times.append(frame_encode_times[idx])
        else:
            p_frame_times.append(frame_encode_times[idx])

    # Add timing details for I and P frames
    if i_frame_times:
        performance_metrics['avg_i_frame_encode_time_sec'] = np.mean(i_frame_times)
    else:
        performance_metrics['avg_i_frame_encode_time_sec'] = 0
        
    if p_frame_times:
        performance_metrics['avg_p_frame_encode_time_sec'] = np.mean(p_frame_times)
    else:
        performance_metrics['avg_p_frame_encode_time_sec'] = 0

    log_result = generate_log_json(frame_num, frame_types, bits, psnrs, msssims,
                                   frame_pixel_num, test_time)
    
    # Add performance metrics to log result
    log_result['performance_metrics'] = performance_metrics
    
    return log_result


def encode_one(args, device):
    sequence_start_time = time.time()
    
    # Convert device string to torch device object
    device_obj = torch.device(device) if isinstance(device, str) else device
    
    # Reset GPU memory stats
    if device_obj.type == 'cuda':
        try:
            # Set the device as current and reset stats
            torch.cuda.set_device(device_obj)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Could not reset CUDA memory stats: {e}")
    
    i_state_dict = get_state_dict(args['i_frame_model_path'])
    i_frame_net = IntraNoAR()
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    if args['force_intra']:
        video_net = None
    else:
        p_state_dict = get_state_dict(args['model_path'])
        video_net = DMC()
        video_net.load_state_dict(p_state_dict)
        video_net = video_net.to(device)
        video_net.eval()

    if args['write_stream']:
        if video_net is not None:
            video_net.update(force=True)
        i_frame_net.update(force=True)

    sub_dir_name = args['video_path']
    gop_size = args['gop']
    frame_num = args['frame_num']

    bin_folder = os.path.join(args['stream_path'], sub_dir_name, str(args['rate_idx']))
    if args['write_stream']:
        create_folder(bin_folder, True)

    if args['save_decoded_frame']:
        decoded_frame_folder = os.path.join(args['decoded_frame_path'], sub_dir_name,
                                            str(args['rate_idx']))
        create_folder(decoded_frame_folder)
    else:
        decoded_frame_folder = None

    args['img_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    args['gop_size'] = gop_size
    args['frame_num'] = frame_num
    args['bin_folder'] = bin_folder
    args['decoded_frame_folder'] = decoded_frame_folder

    result = run_test(video_net, i_frame_net, args, device=device)

    result['ds_name'] = args['ds_name']
    result['video_path'] = args['video_path']
    result['rate_idx'] = args['rate_idx']
    result['sequence_total_time_sec'] = time.time() - sequence_start_time

    return result


def worker(use_cuda, args):
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    gpu_num = 0
    if use_cuda:
        gpu_num = torch.cuda.device_count()

    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    result = encode_one(args, device)
    return result


def calculate_summary_stats(results, rate_num):
    """Calculate summary statistics across all sequences and rate points"""
    summary = {}
    
    # Group results by rate point
    for rate_idx in range(rate_num):
        rate_results = [r for r in results if r['rate_idx'] == rate_idx]
        
        if rate_results:
            rate_summary = {
                'total_sequences': len(rate_results),
                'total_frames': sum(r['i_frame_num'] + r['p_frame_num'] for r in rate_results),
                'avg_performance_metrics': {}
            }
            
            # Calculate average performance metrics
            perf_metrics = rate_results[0]['performance_metrics']
            for key in perf_metrics.keys():
                values = [r['performance_metrics'][key] for r in rate_results if key in r['performance_metrics']]
                if values:
                    rate_summary['avg_performance_metrics'][key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            summary[f'rate_{rate_idx:03d}'] = rate_summary
    
    return summary


def main():
    begin_time = time.time()

    torch.backends.cudnn.enabled = True
    args = parse_args()

    if args.cuda_device is not None and args.cuda_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f:
        config = json.load(f)

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num)
    objs = []

    count_frames = 0
    count_sequences = 0

    rate_num = args.rate_num
    i_frame_q_scales = IntraNoAR.get_q_scales_from_ckpt(args.i_frame_model_path)
    print("q_scales in intra ckpt: ", end='')
    for q in i_frame_q_scales:
        print(f"{q:.3f}, ", end='')
    print()
    if args.i_frame_q_scales is not None:
        assert len(args.i_frame_q_scales) == rate_num
        i_frame_q_scales = args.i_frame_q_scales
        print(f"testing {rate_num} rate points with pre-defined intra y q_scales: ", end='')
    elif len(i_frame_q_scales) == rate_num:
        print(f"testing {rate_num} rate points with intra y q_scales in ckpt: ", end='')
    else:
        max_q_scale = i_frame_q_scales[0]
        min_q_scale = i_frame_q_scales[-1]
        i_frame_q_scales = interpolate_log(min_q_scale, max_q_scale, rate_num)
        print(f"testing {rate_num} rates, using intra y q_scales: ", end='')

    for q in i_frame_q_scales:
        print(f"{q:.3f}, ", end='')
    print()

    if not args.force_intra:
        p_frame_y_q_scales, p_frame_mv_y_q_scales = DMC.get_q_scales_from_ckpt(args.model_path)
        print("y_q_scales in inter ckpt: ", end='')
        for q in p_frame_y_q_scales:
            print(f"{q:.3f}, ", end='')
        print()
        print("mv_y_q_scales in inter ckpt: ", end='')
        for q in p_frame_mv_y_q_scales:
            print(f"{q:.3f}, ", end='')
        print()
        if args.p_frame_y_q_scales is not None:
            assert len(args.p_frame_y_q_scales) == rate_num
            assert len(args.p_frame_mv_y_q_scales) == rate_num
            p_frame_y_q_scales = args.p_frame_y_q_scales
            p_frame_mv_y_q_scales = args.p_frame_mv_y_q_scales
            print(f"testing {rate_num} rate points with pre-defined inter q_scales")
        elif len(p_frame_y_q_scales) == rate_num:
            print(f"testing {rate_num} rate points with inter q_scales in ckpt")
        else:
            max_y_q_scale = p_frame_y_q_scales[0]
            min_y_q_scale = p_frame_y_q_scales[-1]
            p_frame_y_q_scales = interpolate_log(min_y_q_scale, max_y_q_scale, rate_num)

            max_mv_y_q_scale = p_frame_mv_y_q_scales[0]
            min_mv_y_q_scale = p_frame_mv_y_q_scales[-1]
            p_frame_mv_y_q_scales = interpolate_log(min_mv_y_q_scale, max_mv_y_q_scale, rate_num)
        print("y_q_scales for testing: ", end='')
        for q in p_frame_y_q_scales:
            print(f"{q:.3f}, ", end='')
        print()
        print("mv_y_q_scales for testing: ", end='')
        for q in p_frame_mv_y_q_scales:
            print(f"{q:.3f}, ", end='')
        print()

    root_path = args.force_root_path if args.force_root_path is not None else config['root_path']
    config = config['test_classes']
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for seq_name in config[ds_name]['sequences']:
            count_sequences += 1
            for rate_idx in range(rate_num):
                cur_args = {}
                cur_args['rate_idx'] = rate_idx
                cur_args['i_frame_model_path'] = args.i_frame_model_path
                cur_args['i_frame_q_scale'] = i_frame_q_scales[rate_idx]
                if not args.force_intra:
                    cur_args['model_path'] = args.model_path
                    cur_args['p_frame_y_q_scale'] = p_frame_y_q_scales[rate_idx]
                    cur_args['p_frame_mv_y_q_scale'] = p_frame_mv_y_q_scales[rate_idx]
                cur_args['force_intra'] = args.force_intra
                cur_args['video_path'] = seq_name
                cur_args['src_type'] = config[ds_name]['src_type']
                cur_args['src_height'] = config[ds_name]['sequences'][seq_name]['height']
                cur_args['src_width'] = config[ds_name]['sequences'][seq_name]['width']
                cur_args['gop'] = config[ds_name]['sequences'][seq_name]['gop']
                if args.force_intra:
                    cur_args['gop'] = 1
                if args.force_intra_period > 0:
                    cur_args['gop'] = args.force_intra_period
                cur_args['frame_num'] = config[ds_name]['sequences'][seq_name]['frames']
                if args.force_frame_num > 0:
                    cur_args['frame_num'] = args.force_frame_num
                cur_args['dataset_path'] = os.path.join(root_path, config[ds_name]['base_path'])
                cur_args['write_stream'] = args.write_stream
                cur_args['stream_path'] = args.stream_path
                cur_args['save_decoded_frame'] = args.save_decoded_frame
                cur_args['decoded_frame_path'] = f'{args.decoded_frame_path}_DMC_{rate_idx}'
                cur_args['ds_name'] = ds_name
                cur_args['verbose'] = args.verbose

                count_frames += cur_args['frame_num']

                obj = threadpool_executor.submit(
                    worker,
                    args.cuda,
                    cur_args)
                objs.append(obj)

    results = []
    for obj in tqdm(objs):
        result = obj.result()
        results.append(result)

    log_result = {}
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            log_result[ds_name][seq] = {}
            for rate in range(rate_num):
                for res in results:
                    if res['rate_idx'] == rate and ds_name == res['ds_name'] \
                            and seq == res['video_path']:
                        # Convert to JSON-serializable before storing
                        log_result[ds_name][seq][f"{rate:03d}"] = convert_to_json_serializable(res)

    # Add summary statistics
    summary_stats = calculate_summary_stats(results, rate_num)
    log_result['summary_statistics'] = convert_to_json_serializable(summary_stats)

    # Add metadata about the test run
    log_result['test_metadata'] = convert_to_json_serializable({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_test_time_min': (time.time() - begin_time) / 60,
        'worker_count': worker_num,
        'cuda_enabled': args.cuda,
        'cuda_device': args.cuda_device,
        'rate_num': rate_num,
        'i_frame_model_path': args.i_frame_model_path,
        'model_path': args.model_path if not args.force_intra else None
    })

    # Convert all data to JSON-serializable format before saving
    log_result = convert_to_json_serializable(log_result)

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        create_folder(out_json_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)

    total_minutes = (time.time() - begin_time) / 60
    print('Test finished')
    print(f'Tested {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')
    
    # Print summary of performance metrics
    print('\nPerformance Summary:')
    for rate_key, stats in summary_stats.items():
        print(f'\n{rate_key}:')
        perf = stats['avg_performance_metrics']
        if 'total_params' in perf:
            print(f"  Parameters: {perf['total_params']['mean']:.0f}")
        if 'fps' in perf:
            print(f"  Average FPS: {perf['fps']['mean']:.2f}")
        if 'memory_max_allocated_mb' in perf:
            print(f"  Peak Memory (MB): {perf['memory_max_allocated_mb']['mean']:.1f}")


if __name__ == "__main__":
    main()