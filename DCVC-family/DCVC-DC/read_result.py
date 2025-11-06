import json
import pandas as pd
from collections import defaultdict
import os

def read_video_metrics(json_file_path):
    """
    Universal JSON file reader for different formats of video coding data
    
    Args:
        json_file_path (str): Path to JSON file
    
    Returns:
        dict: Dictionary containing analysis results
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Dictionary to store results
    results = {
        'by_dataset': defaultdict(lambda: defaultdict(dict)),
        'summary': defaultdict(dict),
        'metadata': {}
    }
    
    # Special sections to skip
    skip_sections = ['summary_statistics', 'test_metadata']
    
    # Extract test metadata
    if 'test_metadata' in data:
        results['metadata'] = data['test_metadata']
    
    # Iterate through datasets
    for dataset_name, videos in data.items():
        # Skip statistics and metadata sections
        if dataset_name in skip_sections:
            print(f"Skipping non-video data section: {dataset_name}")
            continue
            
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Store data for all quality levels of this dataset
        dataset_metrics = defaultdict(list)
        
        # Check if videos is a dictionary
        if not isinstance(videos, dict):
            print(f"  Warning: Abnormal data format for {dataset_name}, skipping")
            continue
        
        # Iterate through videos
        for video_name, quality_levels in videos.items():
            print(f"  Video: {video_name}")
            
            # Check if quality_levels is a dictionary
            if not isinstance(quality_levels, dict):
                print(f"    Warning: Abnormal data format for {video_name}, skipping")
                continue
            
            # Iterate through quality levels
            for quality_idx, metrics in quality_levels.items():
                # Check if metrics is a dictionary and contains required fields
                if not isinstance(metrics, dict):
                    print(f"    Warning: Abnormal data format for quality level {quality_idx}, skipping")
                    continue
                
                # Check if necessary fields exist
                if 'ave_all_frame_bpp' not in metrics or 'ave_all_frame_psnr' not in metrics:
                    print(f"    Warning: Quality level {quality_idx} missing required fields, skipping")
                    continue
                    
                bpp = metrics.get('ave_all_frame_bpp', 0)
                psnr = metrics.get('ave_all_frame_psnr', 0)
                msssim = metrics.get('ave_all_frame_msssim', 0)  # May be 0 or non-existent
                
                # Store by quality level
                dataset_metrics[quality_idx].append({
                    'video': video_name,
                    'bpp': bpp,
                    'psnr': psnr,
                    'msssim': msssim
                })
                
                # Store in detailed results
                if quality_idx not in results['by_dataset'][dataset_name]:
                    results['by_dataset'][dataset_name][quality_idx] = {
                        'videos': [],
                        'avg_bpp': 0,
                        'avg_psnr': 0,
                        'avg_msssim': 0
                    }
                
                results['by_dataset'][dataset_name][quality_idx]['videos'].append({
                    'name': video_name,
                    'bpp': bpp,
                    'psnr': psnr,
                    'msssim': msssim
                })
                
                print(f"    Quality level {quality_idx}: BPP={bpp:.6f}, PSNR={psnr:.6f}, MS-SSIM={msssim:.6f}")
        
        # Calculate average for each quality level
        for quality_idx, video_data in dataset_metrics.items():
            if len(video_data) > 0:
                avg_bpp = sum(item['bpp'] for item in video_data) / len(video_data)
                avg_psnr = sum(item['psnr'] for item in video_data) / len(video_data)
                avg_msssim = sum(item['msssim'] for item in video_data) / len(video_data)
                
                results['by_dataset'][dataset_name][quality_idx]['avg_bpp'] = avg_bpp
                results['by_dataset'][dataset_name][quality_idx]['avg_psnr'] = avg_psnr
                results['by_dataset'][dataset_name][quality_idx]['avg_msssim'] = avg_msssim
                
                # Store in summary
                results['summary'][dataset_name][quality_idx] = {
                    'avg_bpp': avg_bpp,
                    'avg_psnr': avg_psnr,
                    'avg_msssim': avg_msssim,
                    'video_count': len(video_data)
                }
    
    return results

def print_summary(results):
    """Print results summary"""
    print("\n" + "="*90)
    print("Dataset Quality Level Average Summary")
    print("="*90)
    
    # Print metadata information
    if results['metadata']:
        print(f"\nTest metadata:")
        print(f"  Timestamp: {results['metadata'].get('timestamp', 'N/A')}")
        print(f"  Total test time: {results['metadata'].get('total_test_time_min', 'N/A')} minutes")
        print(f"  Quality level count: {results['metadata'].get('rate_num', 'N/A')}")
        if 'i_frame_model_path' in results['metadata']:
            print(f"  I-frame model: {os.path.basename(results['metadata']['i_frame_model_path'])}")
        if 'p_frame_model_path' in results['metadata']:
            print(f"  P-frame model: {os.path.basename(results['metadata']['p_frame_model_path'])}")
        elif 'model_path' in results['metadata']:
            print(f"  Video model: {os.path.basename(results['metadata']['model_path'])}")
    
    for dataset_name, quality_data in results['summary'].items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 70)
        
        # Sort by quality level
        sorted_qualities = sorted(quality_data.keys())
        
        for quality_idx in sorted_qualities:
            data = quality_data[quality_idx]
            print(f"Quality level {quality_idx}:")
            print(f"  Average BPP:     {data['avg_bpp']:.6f}")
            print(f"  Average PSNR:    {data['avg_psnr']:.6f} dB")
            if data['avg_msssim'] > 0:
                print(f"  Average MS-SSIM: {data['avg_msssim']:.6f}")
            print(f"  Video count:     {data['video_count']}")

def export_to_csv(results, output_file='video_metrics_summary.csv'):
    """Export results to CSV file"""
    rows = []
    
    for dataset_name, quality_data in results['summary'].items():
        for quality_idx, data in quality_data.items():
            row = {
                'Dataset': dataset_name,
                'Quality_Level': quality_idx,
                'Avg_BPP': data['avg_bpp'],
                'Avg_PSNR': data['avg_psnr'],
                'Video_Count': data['video_count']
            }
            if data['avg_msssim'] > 0:
                row['Avg_MS_SSIM'] = data['avg_msssim']
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['Dataset', 'Quality_Level'])
    df.to_csv(output_file, index=False)
    print(f"\nResults exported to: {output_file}")
    return df

def create_comparison_table(results):
    """Create comparison table"""
    import pandas as pd
    
    # Create BPP and PSNR comparison tables
    bpp_data = {}
    psnr_data = {}
    msssim_data = {}
    
    for dataset_name, quality_data in results['summary'].items():
        bpp_row = {}
        psnr_row = {}
        msssim_row = {}
        
        for quality_idx, data in quality_data.items():
            bpp_row[f'Q{quality_idx}'] = data['avg_bpp']
            psnr_row[f'Q{quality_idx}'] = data['avg_psnr']
            if data['avg_msssim'] > 0:
                msssim_row[f'Q{quality_idx}'] = data['avg_msssim']
        
        bpp_data[dataset_name] = bpp_row
        psnr_data[dataset_name] = psnr_row
        if msssim_row:
            msssim_data[dataset_name] = msssim_row
    
    bpp_df = pd.DataFrame(bpp_data).T
    psnr_df = pd.DataFrame(psnr_data).T
    
    print("\n" + "="*90)
    print("BPP Comparison Table (bits per pixel)")
    print("="*90)
    print(bpp_df.round(6))
    
    print("\n" + "="*90)
    print("PSNR Comparison Table (dB)")
    print("="*90)
    print(psnr_df.round(6))
    
    if msssim_data:
        msssim_df = pd.DataFrame(msssim_data).T
        print("\n" + "="*90)
        print("MS-SSIM Comparison Table")
        print("="*90)
        print(msssim_df.round(6))
        return bpp_df, psnr_df, msssim_df
    
    return bpp_df, psnr_df, None

def analyze_rate_distortion(results):
    """Analyze rate-distortion performance"""
    print("\n" + "="*90)
    print("Rate-Distortion Performance Analysis")
    print("="*90)
    
    for dataset_name, quality_data in results['summary'].items():
        print(f"\n{dataset_name} Dataset:")
        
        # Sort by quality level
        sorted_qualities = sorted(quality_data.keys())
        
        bpp_values = []
        psnr_values = []
        
        for quality_idx in sorted_qualities:
            data = quality_data[quality_idx]
            bpp_values.append(data['avg_bpp'])
            psnr_values.append(data['avg_psnr'])
            
        # Calculate efficiency metrics
        if len(bpp_values) > 1:
            for i in range(1, len(bpp_values)):
                bpp_increase = bpp_values[i] - bpp_values[i-1]
                psnr_increase = psnr_values[i] - psnr_values[i-1]
                if bpp_increase > 0:
                    efficiency = psnr_increase / bpp_increase
                    print(f"  Q{sorted_qualities[i-1]} -> Q{sorted_qualities[i]}: "
                          f"PSNR gain {psnr_increase:.3f} dB, "
                          f"BPP increase {bpp_increase:.6f}, "
                          f"Efficiency {efficiency:.2f} dB/bpp")

def simple_extract_metrics(json_file_path):
    """Simplified version: Quick extraction of average metrics for all datasets"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    results = []
    skip_sections = ['summary_statistics', 'test_metadata']
    
    for dataset_name, videos in data.items():
        # Skip statistics and metadata sections
        if dataset_name in skip_sections:
            continue
            
        if not isinstance(videos, dict):
            continue
            
        for video_name, quality_levels in videos.items():
            # Check data format
            if not isinstance(quality_levels, dict):
                continue
                
            for quality_idx, metrics in quality_levels.items():
                # Check if metrics is a dictionary and contains required fields
                if (not isinstance(metrics, dict) or 
                    'ave_all_frame_bpp' not in metrics or 
                    'ave_all_frame_psnr' not in metrics):
                    continue
                    
                row = {
                    'Dataset': dataset_name,
                    'Video': video_name,
                    'Quality_Level': quality_idx,
                    'BPP': metrics.get('ave_all_frame_bpp', 0),
                    'PSNR': metrics.get('ave_all_frame_psnr', 0)
                }
                
                # Add MS-SSIM if exists and non-zero
                msssim = metrics.get('ave_all_frame_msssim', 0)
                if msssim > 0:
                    row['MS_SSIM'] = msssim
                    
                results.append(row)
    
    return pd.DataFrame(results)

def compare_multiple_files(file_list, labels=None):
    """Compare results from multiple JSON files"""
    if labels is None:
        labels = [f"File_{i+1}" for i in range(len(file_list))]
    
    all_results = {}
    
    for i, file_path in enumerate(file_list):
        print(f"\n{'='*50}")
        print(f"Processing file: {file_path}")
        print(f"Label: {labels[i]}")
        print('='*50)
        
        try:
            results = read_video_metrics(file_path)
            all_results[labels[i]] = results
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    # Create comparison table
    comparison_data = []
    for label, results in all_results.items():
        for dataset_name, quality_data in results['summary'].items():
            for quality_idx, data in quality_data.items():
                comparison_data.append({
                    'File': label,
                    'Dataset': dataset_name,
                    'Quality_Level': quality_idx,
                    'Avg_BPP': data['avg_bpp'],
                    'Avg_PSNR': data['avg_psnr'],
                    'Video_Count': data['video_count']
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('multi_file_comparison.csv', index=False)
    
    print(f"\nMulti-file comparison results saved to: multi_file_comparison.csv")
    return comparison_df

# Main function
def main():
    results = read_video_metrics('/home/zhan5096/project/OpenDCVC/DCVC-family/DCVC-DC/output_result.json')
    print_summary(results)
    export_to_csv(results)