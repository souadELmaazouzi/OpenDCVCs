# OpenDCVCs: PyTorch Implementation of Deep Contextual Video Compression Series


## Overview

**OpenDCVCs** is an open-source PyTorch implementation and benchmarking suite for the DCVC series of learned video compression models. It provides comprehensive, training-ready code for four advanced codecs:

- **DCVC**
- **DCVC with Temporal Context Modeling (DCVC-TCM)**
- **DCVC with Hybrid Entropy Modeling (DCVC-HEM)**
- **DCVC with Diverse Contexts (DCVC-DC)**


## Features

- Full training & evaluation pipelines for all DCVC model variants
- Extensive instrucion
- Open, extensible codebase

## Supported Algorithms

| Model      | Description                                                                                                   |
|------------|--------------------------------------------------------------------------------------------------------------|
| DCVC       | Feature-domain conditional coding with contextual entropy modeling                                            |
| DCVC-TCM   | Multi-scale temporal context mining and refilling for richer temporal modeling                                |
| DCVC-HEM   | Hybrid spatial-temporal entropy modeling and multi-granularity quantization                                   |
| DCVC-DC    | Hierarchical quality, offset diversity, and quadtree-based entropy coding for robust, diverse context mining  |

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/OpenDCVCs.git
cd OpenDCVCs
pip install -r requirements.txt
````

*Requires Python 3.8+, PyTorch 1.12+, and CUDA for GPU training/evaluation.*

## Getting Started

### Training dataset
We use Vimeo90k septuplet dataset for training, which consists of 91,701 7-frame sequences with fixed resolution 448 x 256. Available at http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip

### Test dataset
For testing, we evaluate our models on benchmark datasets widely used in the video compression literature:

- [HEVC Class B (1080p)](https://hevc.hhi.fraunhofer.de/)
- [UVG (1080p)](https://ultravideo.fi/dataset.html)
- [MCL-JCV (1080p)](https://mcl.usc.edu/mcl-jcv-dataset/)

### Training and Evaluation Example
See each method subfolder

## Benchmark Results


**Example Results Table:**

| Method        | HEVC-B BD-Rate | UVG BD-Rate | MCL-JCV BD-Rate | Model Params | Inference Time | GPU Mem |
| ------------- | -------------- | ----------- | --------------- | ------------ | -------------- | ------- |
| DCVC-official | 0%             | 0%          | 0%              | 7.94 M       | 0.2615 s       | 21.79GB |
| OpenDCVC      | -10.60%        | -6.35%      | 10.40%          | 7.94 M       | 0.2620 s       | 21.80GB |
| OpenDCVC-TCM  | -42.35%        | -46.70%     | -27.11%         | 10.70 M      | 0.3070 s       | 5.67GB  |
| OpenDCVC-HEM  | -56.39%        | -59.75%     | -46.94%         | 17.52 M      | 0.3458 s       | 4.74GB  |
| OpenDCVC-DC   | -61.56%        | -65.49%     | -52.74%         | 19.77 M      | 0.5255 s       | 7.78GB  |

*Tested on Nvidia L40S GPU, AMD EPYC 9554 CPU, 384GB RAM, 1080p videos.*


<!-- 
## Citation

If you use OpenDCVCs in your research, please cite:

```bibtex
@inproceedings{zhang2025opendcvcs,
  title={OpenDCVCs: A PyTorch Open Source Implementation and Performance Evaluation of the DCVC series Video Codecs},
  author={Zhang, Yichi and Zhu, Fengqing},
  booktitle={Proceedings of ACM Multimedia},
  year={2025}
}
``` -->

## License

This project is licensed under the MIT License.

## Contact & Contributions

We welcome contributions and suggestions!

* Submit issues or pull requests via GitLab
* Contact: [Yichi Zhang](mailto:zhan5096@purdue.edu)



# Acknowledgement
The implementation is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI), [PyTorchVideoCompression](https://github.com/ZhihaoHu/PyTorchVideoCompression) and the official [DCVC](https://github.com/microsoft/DCVC). Some model weights of intra coding come from [CompressAI](https://github.com/InterDigitalInc/CompressAI).# OpenDCVCs
