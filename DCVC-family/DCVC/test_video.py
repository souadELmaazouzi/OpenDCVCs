# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
import os
import concurrent.futures
import multiprocessing
import torch
import json
import numpy as np
from PIL import Image
from src.models.DCVC_net_ori import DCVC_net
from src.zoo.image import model_architectures as architectures
import time
from tqdm import tqdm
import warnings
from pytorch_msssim import ms_ssim
import psutil
import gc

# For MACs measurement
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    print("Warning: ptflops not available. MACs measurement will be skipped.")
    PTFLOPS_AVAILABLE = False

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script with performance metrics")

    parser.add_argument('--i_frame_model_name', type=str, default="cheng2020-anchor")
    parser.add_argument('--i_frame_model_path', type=str, nargs="+")
    parser.add_argument('--model_path',  type=str, nargs="+")
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--cuda_device", default=None,
                        help="the cuda device used, e.g., 0; 0,1; 1,2,3; etc.")
    parser.add_argument('--write_stream', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("--write_recon_frame", type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--recon_bin_path', type=str, default="recon_bin_path")
    parser.add_argument('--output_json_result_path', type=str, required=True)
    parser.add_argument("--model_type",  type=str,  default="psnr", help="psnr, msssim")
    parser.add_argument("--measure_macs", type=str2bool, nargs='?', const=True, default=True,
                        help="Whether to measure MACs (requires ptflops)")

    args = parser.parse_args()
    return args


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


def read_frame_to_torch(path):
    input_image = Image.open(path).convert('RGB')
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0)/255
    return input_image


def write_torch_frame(frame, path):
    frame_result = frame.clone()
    frame_result = frame_result.cpu().detach().numpy().transpose(1, 2, 0)*255
    frame_result = np.clip(np.rint(frame_result), 0, 255)
    frame_result = Image.fromarray(frame_result.astype('uint8'), 'RGB')
    frame_result.save(path)


def count_parameters(model):
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())


def measure_model_macs(model, input_shape, device, is_video_model=False):
    """Measure MACs (FLOPs) for a model"""
    if not PTFLOPS_AVAILABLE:
        return 0, 0
    
    try:
        if is_video_model:
            # For video model, create a wrapper that takes two inputs
            class VideoModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    # For video model, use the same input as both reference and current frame
                    return self.model(x, x)['recon_image']
            
            wrapped_model = VideoModelWrapper(model).to(device)
        else:
            # For I-frame model
            class IFrameModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    return self.model(x)['x_hat']
            
            wrapped_model = IFrameModelWrapper(model).to(device)
        
        macs, params = get_model_complexity_info(
            wrapped_model, 
            input_shape, 
            print_per_layer_stat=False, 
            verbose=False
        )
        
        # Convert string format to numbers (e.g., "1.23 GMac" -> 1.23e9)
        if isinstance(macs, str):
            if 'GMac' in macs:
                macs = float(macs.replace(' GMac', '')) * 1e9
            elif 'MMac' in macs:
                macs = float(macs.replace(' MMac', '')) * 1e6
            elif 'KMac' in macs:
                macs = float(macs.replace(' KMac', '')) * 1e3
            else:
                macs = float(macs.replace(' Mac', ''))
        
        return macs, params
    except Exception as e:
        print(f"Warning: MACs measurement failed: {e}")
        return 0, 0


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


def encode_one(args_dict, device):
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
    
    i_frame_load_checkpoint = torch.load(args_dict['i_frame_model_path'],
                                         map_location=torch.device('cpu'))
    i_frame_net = architectures[args_dict['i_frame_model_name']].from_state_dict(
        i_frame_load_checkpoint).eval()

    video_net = DCVC_net()
    load_checkpoint = torch.load(args_dict['model_path'], map_location=torch.device('cpu'))
    video_net.load_dict(load_checkpoint)

    video_net = video_net.to(device)
    video_net.eval()
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()
    
    # Model performance metrics
    performance_metrics = {}
    
    # Count parameters
    i_frame_params = count_parameters(i_frame_net)
    video_params = count_parameters(video_net)
    performance_metrics['i_frame_model_params'] = i_frame_params
    performance_metrics['video_model_params'] = video_params
    performance_metrics['total_params'] = i_frame_params + video_params
    
    if args_dict['write_stream']:
        video_net.update(force=True)
        i_frame_net.update(force=True)

    sub_dir_name = args_dict['video_path']
    ref_frame = None
    frame_types = []
    qualitys = []
    bits = []
    bits_mv_y = []
    bits_mv_z = []
    bits_y = []
    bits_z = []
    frame_encode_times = []

    gop_size = args_dict['gop']
    frame_pixel_num = 0
    frame_num = args_dict['frame_num']

    recon_bin_folder = os.path.join(args_dict['recon_bin_path'], sub_dir_name,
                                    os.path.basename(args_dict['model_path'])[:-4])
    if not os.path.exists(recon_bin_folder):
        os.makedirs(recon_bin_folder)

    # Figure out the naming convention
    pngs = os.listdir(os.path.join(args_dict['dataset_path'], sub_dir_name))
    if 'im1.png' in pngs:
        padding = 1
    elif 'im00001.png' in pngs:
        padding = 5
    else:
        raise ValueError('unknown image naming convention; please specify')

    # Measure MACs on first frame
    first_frame_path = os.path.join(args_dict['dataset_path'], sub_dir_name, f"im{str(1).zfill(padding)}.png")
    sample_frame = read_frame_to_torch(first_frame_path).to(device)
    frame_pixel_num = sample_frame.shape[2] * sample_frame.shape[3]
    
    if args_dict.get('measure_macs', True) and PTFLOPS_AVAILABLE:
        # Measure MACs for I-frame model
        input_shape = (sample_frame.shape[1], sample_frame.shape[2], sample_frame.shape[3])
        i_frame_macs, _ = measure_model_macs(i_frame_net, input_shape, device_obj, is_video_model=False)
        
        # For video model, measure with two inputs (ref + current frame)
        video_macs, _ = measure_model_macs(video_net, input_shape, device_obj, is_video_model=True)
        
        performance_metrics['i_frame_model_macs'] = i_frame_macs
        performance_metrics['video_model_macs'] = video_macs
    else:
        performance_metrics['i_frame_model_macs'] = 0
        performance_metrics['video_model_macs'] = 0

    encoding_start_time = time.time()
    
    with torch.no_grad():
        for frame_idx in range(frame_num):
            frame_start_time = time.time()
            
            ori_frame = read_frame_to_torch(
                os.path.join(args_dict['dataset_path'],
                             sub_dir_name,
                             f"im{str(frame_idx+1).zfill(padding)}.png"))
            ori_frame = ori_frame.to(device)

            if frame_idx == 0:
                assert(frame_pixel_num == ori_frame.shape[2]*ori_frame.shape[3])

            if args_dict['write_stream']:
                bin_path = os.path.join(recon_bin_folder, f"{frame_idx}.bin")
                if frame_idx % gop_size == 0:
                    result = i_frame_net.encode_decode(ori_frame, bin_path)
                    ref_frame = result["x_hat"]
                    bpp = result["bpp"]
                    frame_types.append(0)
                    bits.append(bpp*frame_pixel_num)
                    bits_mv_y.append(0)
                    bits_mv_z.append(0)
                    bits_y.append(0)
                    bits_z.append(0)
                else:
                    result = video_net.encode_decode(ref_frame, ori_frame, bin_path)
                    ref_frame = result['recon_image']
                    bpp = result['bpp']
                    frame_types.append(1)
                    bits.append(bpp*frame_pixel_num)
                    bits_mv_y.append(result['bpp_mv_y']*frame_pixel_num)
                    bits_mv_z.append(result['bpp_mv_z']*frame_pixel_num)
                    bits_y.append(result['bpp_y']*frame_pixel_num)
                    bits_z.append(result['bpp_z']*frame_pixel_num)
            else:
                if frame_idx % gop_size == 0:
                    result = i_frame_net(ori_frame)
                    bit = sum((torch.log(likelihoods).sum() / (-math.log(2)))
                              for likelihoods in result["likelihoods"].values())
                    ref_frame = result["x_hat"]
                    frame_types.append(0)
                    bits.append(bit.item())
                    bits_mv_y.append(0)
                    bits_mv_z.append(0)
                    bits_y.append(0)
                    bits_z.append(0)
                else:
                    result = video_net(ref_frame, ori_frame)
                    ref_frame = result['recon_image']
                    bpp = result['bpp']
                    frame_types.append(1)
                    bits.append(bpp.item()*frame_pixel_num)
                    bits_mv_y.append(result['bpp_mv_y'].item()*frame_pixel_num)
                    bits_mv_z.append(result['bpp_mv_z'].item()*frame_pixel_num)
                    bits_y.append(result['bpp_y'].item()*frame_pixel_num)
                    bits_z.append(result['bpp_z'].item()*frame_pixel_num)

            ref_frame = ref_frame.clamp_(0, 1)
            if args_dict['write_recon_frame']:
                write_torch_frame(ref_frame.squeeze(), os.path.join(recon_bin_folder, f"recon_frame_{frame_idx}.png"))
            if args_dict['model_type'] == 'psnr':
                qualitys.append(PSNR(ref_frame, ori_frame))
            else:
                qualitys.append(
                    ms_ssim(ref_frame, ori_frame, data_range=1.0).item())
            
            frame_encode_time = time.time() - frame_start_time
            frame_encode_times.append(frame_encode_time)

    encoding_end_time = time.time()
    
    # Calculate timing metrics
    performance_metrics['total_encoding_time_sec'] = encoding_end_time - encoding_start_time
    performance_metrics['avg_frame_encode_time_sec'] = np.mean(frame_encode_times)
    performance_metrics['fps'] = frame_num / (encoding_end_time - encoding_start_time)
    
    # Get memory usage information
    memory_info = get_gpu_memory_info(device)
    performance_metrics.update({f'memory_{k}': v for k, v in memory_info.items()})

    cur_all_i_frame_bit = 0
    cur_all_i_frame_quality = 0
    cur_all_p_frame_bit = 0
    cur_all_p_frame_bit_mv_y = 0
    cur_all_p_frame_bit_mv_z = 0
    cur_all_p_frame_bit_y = 0
    cur_all_p_frame_bit_z = 0
    cur_all_p_frame_quality = 0
    cur_i_frame_num = 0
    cur_p_frame_num = 0
    
    i_frame_times = []
    p_frame_times = []
    
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            cur_all_i_frame_bit += bits[idx]
            cur_all_i_frame_quality += qualitys[idx]
            cur_i_frame_num += 1
            i_frame_times.append(frame_encode_times[idx])
        else:
            cur_all_p_frame_bit += bits[idx]
            cur_all_p_frame_bit_mv_y += bits_mv_y[idx]
            cur_all_p_frame_bit_mv_z += bits_mv_z[idx]
            cur_all_p_frame_bit_y += bits_y[idx]
            cur_all_p_frame_bit_z += bits_z[idx]
            cur_all_p_frame_quality += qualitys[idx]
            cur_p_frame_num += 1
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

    log_result = {}
    log_result['name'] = f"{os.path.basename(args_dict['model_path'])}_{sub_dir_name}"
    log_result['ds_name'] = args_dict['ds_name']
    log_result['video_path'] = args_dict['video_path']
    log_result['frame_pixel_num'] = frame_pixel_num
    log_result['i_frame_num'] = cur_i_frame_num
    log_result['p_frame_num'] = cur_p_frame_num
    log_result['ave_i_frame_bpp'] = cur_all_i_frame_bit / cur_i_frame_num / frame_pixel_num
    log_result['ave_i_frame_quality'] = cur_all_i_frame_quality / cur_i_frame_num
    
    if cur_p_frame_num > 0:
        total_p_pixel_num = cur_p_frame_num * frame_pixel_num
        log_result['ave_p_frame_bpp'] = cur_all_p_frame_bit / total_p_pixel_num
        log_result['ave_p_frame_bpp_mv_y'] = cur_all_p_frame_bit_mv_y / total_p_pixel_num
        log_result['ave_p_frame_bpp_mv_z'] = cur_all_p_frame_bit_mv_z / total_p_pixel_num
        log_result['ave_p_frame_bpp_y'] = cur_all_p_frame_bit_y / total_p_pixel_num
        log_result['ave_p_frame_bpp_z'] = cur_all_p_frame_bit_z / total_p_pixel_num
        log_result['ave_p_frame_quality'] = cur_all_p_frame_quality / cur_p_frame_num
    else:
        log_result['ave_p_frame_bpp'] = 0
        log_result['ave_p_frame_quality'] = 0
        log_result['ave_p_frame_bpp_mv_y'] = 0
        log_result['ave_p_frame_bpp_mv_z'] = 0
        log_result['ave_p_frame_bpp_y'] = 0
        log_result['ave_p_frame_bpp_z'] = 0
        
    log_result['ave_all_frame_bpp'] = (cur_all_i_frame_bit + cur_all_p_frame_bit) / \
        (frame_num * frame_pixel_num)
    log_result['ave_all_frame_quality'] = (cur_all_i_frame_quality + cur_all_p_frame_quality) / frame_num
    
    # Add performance metrics to log result
    log_result['performance_metrics'] = performance_metrics
    log_result['sequence_total_time_sec'] = time.time() - sequence_start_time
    
    return log_result


def worker(use_cuda, args):
    if args['write_stream']:
        torch.backends.cudnn.benchmark = False
        if 'use_deterministic_algorithms' in dir(torch):
            torch.use_deterministic_algorithms(True)
        else:
            torch.set_deterministic(True)
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
    result['model_idx'] = args['model_idx']
    return result


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


def filter_dict(result):
    keys = ['i_frame_num', 'p_frame_num', 'ave_i_frame_bpp', 'ave_i_frame_quality', 'ave_p_frame_bpp',
            'ave_p_frame_bpp_mv_y', 'ave_p_frame_bpp_mv_z', 'ave_p_frame_bpp_y',
            'ave_p_frame_bpp_z', 'ave_p_frame_quality', 'ave_all_frame_bpp', 'ave_all_frame_quality',
            'performance_metrics', 'sequence_total_time_sec']
    res = {k: v for k, v in result.items() if k in keys}
    return convert_to_json_serializable(res)


def calculate_summary_stats(results, args):
    """Calculate summary statistics across all sequences and models"""
    summary = {}
    
    # Group results by model
    models = {}
    for result in results:
        model_name = os.path.basename(args.model_path[result['model_idx']])
        if model_name not in models:
            models[model_name] = []
        models[model_name].append(result)
    
    for model_name, model_results in models.items():
        model_summary = {
            'total_sequences': len(model_results),
            'total_frames': sum(r['i_frame_num'] + r['p_frame_num'] for r in model_results),
            'avg_performance_metrics': {}
        }
        
        # Calculate average performance metrics
        if model_results:
            perf_metrics = model_results[0]['performance_metrics']
            for key in perf_metrics.keys():
                values = [r['performance_metrics'][key] for r in model_results if key in r['performance_metrics']]
                if values:
                    model_summary['avg_performance_metrics'][key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        summary[model_name] = model_summary
    
    return summary


def main():
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
    begin_time = time.time()
    
    for ds_name in config:
        for seq_name in config[ds_name]['sequences']:
            count_sequences += 1
            for model_idx in range(len(args.model_path)):
                cur_dict = {}
                cur_dict['model_idx'] = model_idx
                cur_dict['i_frame_model_path'] = args.i_frame_model_path[model_idx]
                cur_dict['i_frame_model_name'] = args.i_frame_model_name
                cur_dict['model_path'] = args.model_path[model_idx]
                cur_dict['video_path'] = seq_name
                cur_dict['gop'] = config[ds_name]['sequences'][seq_name]['gop']
                cur_dict['frame_num'] = config[ds_name]['sequences'][seq_name]['frames']
                cur_dict['dataset_path'] = config[ds_name]['base_path']
                cur_dict['write_stream'] = args.write_stream
                cur_dict['write_recon_frame'] = args.write_recon_frame
                cur_dict['recon_bin_path'] = args.recon_bin_path
                cur_dict['model_type'] = args.model_type
                cur_dict['ds_name'] = ds_name
                cur_dict['measure_macs'] = args.measure_macs

                count_frames += cur_dict['frame_num']

                obj = threadpool_executor.submit(
                    worker,
                    args.cuda,
                    cur_dict)
                objs.append(obj)

    results = []
    for obj in tqdm(objs):
        result = obj.result()
        results.append(result)

    log_result = {}

    for ds_name in config:
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            log_result[ds_name][seq] = {}
            for model_idx in range(len(args.model_path)):
                ckpt = os.path.basename(args.model_path[model_idx])
                for res in results:
                    if res['name'].startswith(ckpt) and ds_name == res['ds_name'] \
                            and seq == res['video_path']:
                        log_result[ds_name][seq][ckpt] = filter_dict(res)

    # Add summary statistics
    summary_stats = calculate_summary_stats(results, args)
    log_result['summary_statistics'] = convert_to_json_serializable(summary_stats)

    # Add metadata about the test run
    log_result['test_metadata'] = convert_to_json_serializable({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_test_time_min': (time.time() - begin_time) / 60,
        'worker_count': worker_num,
        'cuda_enabled': args.cuda,
        'cuda_device': args.cuda_device,
        'ptflops_available': PTFLOPS_AVAILABLE,
        'model_paths': args.model_path,
        'i_frame_model_paths': args.i_frame_model_path
    })

    # Convert all data to JSON-serializable format before saving
    log_result = convert_to_json_serializable(log_result)

    with open(args.output_json_result_path, 'w') as fp:
        json.dump(log_result, fp, indent=2)

    total_minutes = (time.time() - begin_time) / 60

    count_models = len(args.model_path)
    count_frames = count_frames // count_models
    print('Test finished')
    print(f'Tested {count_models} models on {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')
    
    # Print summary of performance metrics
    print('\nPerformance Summary:')
    for model_name, stats in summary_stats.items():
        print(f'\nModel: {model_name}')
        perf = stats['avg_performance_metrics']
        if 'total_params' in perf:
            print(f"  Parameters: {perf['total_params']['mean']:.0f}")
        if 'i_frame_model_macs' in perf and 'video_model_macs' in perf:
            total_macs = perf['i_frame_model_macs']['mean'] + perf['video_model_macs']['mean']
            print(f"  Total MACs: {total_macs:.2e}")
        if 'fps' in perf:
            print(f"  Average FPS: {perf['fps']['mean']:.2f}")
        if 'memory_max_allocated_mb' in perf:
            print(f"  Peak Memory (MB): {perf['memory_max_allocated_mb']['mean']:.1f}")


if __name__ == "__main__":
    main()