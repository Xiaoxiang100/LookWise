# SvfEye: A Semantic-Visual Fusion Framework with Multi-Scale Visual Context for Multimodal Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2603.00171-b31b1b.svg)](https://arxiv.org/abs/2603.00171)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

SvfEye is a training-free framework for adaptive visual-semantic fusion in multimodal large language models. It decides **when** extra local visual evidence is needed with token confidence, then decides **where** to look with semantic-guided localization. This avoids blind cropping, reduces attention drift, and improves fine-grained visual reasoning without additional model training.

## Highlights

- **Training-free and plug-and-play.** SvfEye can be attached to open-source MLLMs without parameter updates.
- **Adaptive local inspection.** A confidence-based decision module skips unnecessary crops for easy samples.
- **Semantic-guided localization.** Query targets are extracted from the question and used to guide attention-based region selection.
- **Efficient inference.** SvfEye achieves an approximately **4.0x** speedup over the search-based ZoomEye pipeline on high-resolution reasoning while remaining competitive in accuracy.

## Main Results

Results are reported on AOKVQA, POPE, V*-Bench, HR-Bench 4K, and HR-Bench 8K. Higher is better.

| Model | Method | Training-free | AOKVQA | POPE | V*-Bench | HR-Bench 4K | HR-Bench 8K |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| LLaVA-v1.5-7B | Baseline | yes | 71.00 | 86.98 | 48.68 | 36.13 | 32.13 |
| LLaVA-v1.5-7B | DC2 | yes | - | - | 57.60 | - | 39.50 |
| LLaVA-v1.5-7B | VisCrop | yes | - | - | 62.30 | 46.25 | 35.75 |
| LLaVA-v1.5-7B | MLLMs-Know | yes | 72.31 | 87.25 | 56.02 | 44.38 | 37.25 |
| LLaVA-v1.5-7B | ZoomEye | yes | 70.56 | **88.94** | **83.25** | **49.88** | **48.63** |
| LLaVA-v1.5-7B | **SvfEye** | yes | **72.90** | 87.37 | 62.80 | 47.38 | 42.00 |
| LLaVA-v1.5-7B | Delta vs. baseline | - | +1.90 | +0.39 | +14.12 | +11.25 | +9.87 |
| Qwen2.5-VL-3B | Baseline | yes | 71.44 | 87.20 | 75.90 | 67.50 | 58.88 |
| Qwen2.5-VL-3B | Pixel Reasoner | no | - | - | 84.82 | - | 66.00 |
| Qwen2.5-VL-3B | MLLMs-Know | yes | 71.62 | **89.12** | 75.90 | 66.36 | 64.88 |
| Qwen2.5-VL-3B | ZoomEye | yes | 71.26 | 88.93 | **89.01** | 70.13 | 68.38 |
| Qwen2.5-VL-3B | **SvfEye** | yes | **73.10** | **89.12** | 86.38 | **73.25** | **70.00** |
| Qwen2.5-VL-3B | Delta vs. baseline | - | +1.66 | +1.92 | +10.48 | +5.75 | +11.12 |

## Installation

```bash
git clone https://github.com/Xiaoxiang100/SvfEye.git
cd SvfEye
pip install -r requirements.txt
```

## Data And Models

Set the model and annotation roots before running evaluation:

```bash
export MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
export ANNO_PATH=/path/to/svfeye_data
```

The annotation directory is expected to contain benchmark folders such as:

```text
svfeye_data/
  hr-bench_4k/annotation_hr-bench_4k.json
  hr-bench_8k/annotation_hr-bench_8k.json
```

## Evaluation

Run SvfEye on HR-Bench 4K:

```bash
bash perform_svfeye_4k.sh
```

Run the direct-answer baseline on HR-Bench 4K:

```bash
bash perform_svfeye_4k.sh direct
```

Run SvfEye on HR-Bench 8K:

```bash
bash perform_svfeye_8k.sh
```

Use a custom confidence threshold:

```bash
CONF_THRESHOLD=0.96 bash perform_svfeye_4k.sh
```

Evaluate a merged HR-Bench answer file:

```bash
python svfeye/eval/eval_results_hr-bench.py --answers-file /path/to/result.jsonl
```

## Project Structure

```text
SvfEye/
  svfeye/
    svfeye.py                  # Core SvfEye inference pipeline
    svfeye_model_qwenvl.py     # Qwen2.5-VL wrapper
    utils.py                   # Utility functions
    qwen2_5_methods.py         # Qwen-specific helpers
    eval/
      perform_svfeye.py        # Evaluation inference entry point
      eval_results_hr-bench.py # HR-Bench scorer
    ic_examples/               # In-context examples
    llava/                     # LLaVA-derived components
  perform_svfeye_4k.sh         # HR-Bench 4K runner
  perform_svfeye_8k.sh         # HR-Bench 8K runner
  requirements.txt
```

## Citation

If SvfEye helps your research, please cite:

```bibtex
@article{shen2026svfeye,
  title={SvfEye: A Semantic-Visual Fusion Framework with Multi-Scale Visual Context for Multimodal Reasoning},
  author={Shen, Yuxiang and Huang, Hailong and Gao, Zhenkun and Li, Xueheng and Zhou, Man and Xie, Chengjun and Che, Haoxuan and He, Xuanhua and Zhang, Jie},
  journal={arXiv preprint arXiv:2603.00171},
  year={2026}
}
```

## Contact

For questions, suggestions, or reproduction issues, please open an issue in this repository.
