import json
import pandas as pd
from collections import defaultdict

def read_video_metrics(json_file_path):
    """
    Read JSON file and extract average bpp and psnr for each dataset at various quality levels
    
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
        'summary': defaultdict(dict)
    }
    
    # Special sections to skip
    skip_sections = ['summary_statistics', 'test_metadata']
    
    # Iterate through datasets
    for dataset_name, videos in data.items():
        # Skip statistics and metadata sections
        if dataset_name in skip_sections:
            print(f"\nSkipping non-video data section: {dataset_name}")
            continue
            
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Store data for all quality levels of this dataset
        dataset_metrics = defaultdict(list)
        
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
                    
                bpp = metrics.get('ave_all_frame_bpp', 0)
                psnr = metrics.get('ave_all_frame_psnr', 0)
                
                # Store by quality level
                dataset_metrics[quality_idx].append({
                    'video': video_name,
                    'bpp': bpp,
                    'psnr': psnr
                })
                
                # Store in detailed results
                if quality_idx not in results['by_dataset'][dataset_name]:
                    results['by_dataset'][dataset_name][quality_idx] = {
                        'videos': [],
                        'avg_bpp': 0,
                        'avg_psnr': 0
                    }
                
                results['by_dataset'][dataset_name][quality_idx]['videos'].append({
                    'name': video_name,
                    'bpp': bpp,
                    'psnr': psnr
                })
                
                print(f"    Quality level {quality_idx}: BPP={bpp:.6f}, PSNR={psnr:.6f}")
        
        # Calculate average for each quality level
        for quality_idx, video_data in dataset_metrics.items():
            avg_bpp = sum(item['bpp'] for item in video_data) / len(video_data)
            avg_psnr = sum(item['psnr'] for item in video_data) / len(video_data)
            
            results['by_dataset'][dataset_name][quality_idx]['avg_bpp'] = avg_bpp
            results['by_dataset'][dataset_name][quality_idx]['avg_psnr'] = avg_psnr
            
            # Store in summary
            results['summary'][dataset_name][quality_idx] = {
                'avg_bpp': avg_bpp,
                'avg_psnr': avg_psnr,
                'video_count': len(video_data)
            }
    
    return results

def print_summary(results):
    """Print results summary"""
    print("\n" + "="*80)
    print("Dataset Quality Level Average Summary")
    print("="*80)
    
    for dataset_name, quality_data in results['summary'].items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 50)
        
        # Sort by quality level
        sorted_qualities = sorted(quality_data.keys())
        
        for quality_idx in sorted_qualities:
            data = quality_data[quality_idx]
            print(f"Quality level {quality_idx}:")
            print(f"  Average BPP:  {data['avg_bpp']:.6f}")
            print(f"  Average PSNR: {data['avg_psnr']:.6f}")
            print(f"  Video count:  {data['video_count']}")

def export_to_csv(results, output_file='video_metrics_summary.csv'):
    """Export results to CSV file"""
    rows = []
    
    for dataset_name, quality_data in results['summary'].items():
        for quality_idx, data in quality_data.items():
            rows.append({
                'Dataset': dataset_name,
                'Quality_Level': quality_idx,
                'Avg_BPP': data['avg_bpp'],
                'Avg_PSNR': data['avg_psnr'],
                'Video_Count': data['video_count']
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['Dataset', 'Quality_Level'])
    df.to_csv(output_file, index=False)
    print(f"\nResults exported to: {output_file}")

def create_comparison_table(results):
    """Create comparison table"""
    import pandas as pd
    
    # Create BPP comparison table
    bpp_data = {}
    psnr_data = {}
    
    for dataset_name, quality_data in results['summary'].items():
        bpp_row = {}
        psnr_row = {}
        
        for quality_idx, data in quality_data.items():
            bpp_row[f'Q{quality_idx}'] = data['avg_bpp']
            psnr_row[f'Q{quality_idx}'] = data['avg_psnr']
        
        bpp_data[dataset_name] = bpp_row
        psnr_data[dataset_name] = psnr_row
    
    bpp_df = pd.DataFrame(bpp_data).T
    psnr_df = pd.DataFrame(psnr_data).T
    
    print("\n" + "="*80)
    print("BPP Comparison Table")
    print("="*80)
    print(bpp_df.round(6))
    
    print("\n" + "="*80)
    print("PSNR Comparison Table")
    print("="*80)
    print(psnr_df.round(6))
    
    return bpp_df, psnr_df

# Main function example
def main():
    # Usage example
    json_file_path = "your_video_data.json"  # Replace with your JSON file path
    
    try:
        print("Starting to read and analyze JSON file...")
        results = read_video_metrics(json_file_path)
        
        # Print summary
        print_summary(results)
        
        # Create comparison tables
        bpp_df, psnr_df = create_comparison_table(results)
        
        # Export to CSV
        export_to_csv(results)
        
        # Optional: Export comparison tables
        bpp_df.to_csv('bpp_comparison.csv')
        psnr_df.to_csv('psnr_comparison.csv')
        
        print("\nAnalysis completed!")
        
    except FileNotFoundError:
        print(f"Error: File not found {json_file_path}")
    except json.JSONDecodeError:
        print("Error: Incorrect JSON file format")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

# Simplified version - if you only need basic functionality
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
            
        for video_name, quality_levels in videos.items():
            # Check data format
            if not isinstance(quality_levels, dict):
                continue
                
            for quality_idx, metrics in quality_levels.items():
                # Check if metrics is a dictionary and contains required fields
                if not isinstance(metrics, dict) or 'ave_all_frame_bpp' not in metrics:
                    continue
                    
                results.append({
                    'Dataset': dataset_name,
                    'Video': video_name,
                    'Quality_Level': quality_idx,
                    'BPP': metrics.get('ave_all_frame_bpp', 0),
                    'PSNR': metrics.get('ave_all_frame_psnr', 0)
                })
    
    return pd.DataFrame(results)

# Basic usage
results = read_video_metrics('/home/zhan5096/project/OpenDCVC/DCVC-family/DCVC-HEM/test_result.json')
print_summary(results)
export_to_csv(results)