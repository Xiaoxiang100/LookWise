import argparse
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from lookwise.pipeline import get_direct_response, get_response_with_attention
from lookwise.qwen_vl_model import QwenVLMethod

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_chunk(items, num_chunks, chunk_idx):
    chunks = [[] for _ in range(num_chunks)]
    for i in range(num_chunks):
        chunks[i] = items[i::num_chunks]
    return chunks[chunk_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Qwen2.5-VL model path")
    parser.add_argument("--annotation_path", type=str, default="lookwise/lookwise_data", help="Path to benchmark annotations")
    parser.add_argument("--benchmark", type=str, choices=["vstar", "hr-bench_4k", "hr-bench_8k", "mme-realworld", "aokvqa", "docvqa", "pope", "textvqa"], default="aokvqa")
    parser.add_argument("--direct-answer", action="store_true")
    parser.add_argument("--conf-threshold", type=float, default=1.00)
    args = parser.parse_args()

    model_path = args.model_path
    annotation_path = args.annotation_path
    benchmark = args.benchmark

    if args.answers_file is None:
        answers_dir = f"lookwise/eval/answers/{benchmark}"
        answers_dir = os.path.join(answers_dir, os.path.basename(args.model_path))
        os.makedirs(answers_dir, exist_ok=True)
        answer_tag = "lookwise" if not args.direct_answer else "direct_answer"
        args.answers_file = os.path.join(answers_dir, f"{answer_tag}.jsonl")
        print(f"Output file: {args.answers_file}")

    print("Initializing model...")
    kwargs = {"load_in_8bit": True} if "32b" in model_path.lower() else {}
    method = QwenVLMethod(
        model_path=model_path,
        device="cuda:0",
        torch_dtype=torch.bfloat16,
        patch_scale=1.2,
        **kwargs,
    )

    ic_examples_path = f"lookwise/ic_examples/{benchmark}.json"
    data_path = os.path.join(annotation_path, f"{benchmark}/annotation_{benchmark}.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Annotation file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    annotations = get_chunk(annotations, args.num_chunks, args.chunk_idx)

    results_file = open(args.answers_file, "w", encoding="utf-8")

    total_start_time = time.time()
    time_stats = {
        "total_samples": len(annotations),
        "total_time": 0.0,
        "avg_time_per_sample": 0.0,
        "sample_times": [],
    }
    pipeline_called_count = 0
    skip_crop_after_filter = 0
    crop_count = 0
    no_cropping_count = 0
    option_list_crop_count = 0
    option_list_no_crop_count = 0

    print(f"Start inference on {len(annotations)} samples...")
    with open(ic_examples_path, "r", encoding="utf-8") as f:
        ic_examples = json.load(f)

    for idx, annotation in enumerate(tqdm(annotations), 1):
        sample_start_time = time.time()
        try:
            if not args.direct_answer:
                pipeline_called_count += 1
                response, image_list, attention_maps, entered_crop = get_response_with_attention(
                    method=method,
                    annotation=annotation,
                    ic_examples=ic_examples,
                    image_folder=os.path.join(annotation_path, f"{benchmark}"),
                    conf_threshold=args.conf_threshold,
                )
                print(response)

                answer_type = annotation.get("answer_type", "free_form")
                if answer_type == "option_list" and entered_crop is not None:
                    if entered_crop is True:
                        option_list_crop_count += 1
                    elif entered_crop is False:
                        option_list_no_crop_count += 1

                if len(image_list) == 1:
                    no_cropping_count += 1

                if isinstance(response, dict) and response.get("skip_crop"):
                    skip_crop_after_filter += 1
                else:
                    crop_count += 1
            else:
                response = get_direct_response(
                    method=method,
                    annotation=annotation,
                    image_folder=os.path.join(annotation_path, f"{benchmark}"),
                )

            sample_time = time.time() - sample_start_time
            annotation["output"] = response

            results_file.write(json.dumps(annotation) + "\n")
            results_file.flush()

            time_stats["sample_times"].append(
                {
                    "idx": idx,
                    "time": round(sample_time, 3),
                    "image": annotation.get("input_image", ""),
                }
            )

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\n[OOM Error] Sample {idx} failed. Image: {annotation.get('input_image', 'N/A')}")
            annotation["output"] = "[OOM_ERROR] Out of memory"
            annotation["error"] = "OOM"
            results_file.write(json.dumps(annotation) + "\n")
            results_file.flush()
            continue

        except Exception as exc:
            torch.cuda.empty_cache()
            print(f"\n[Error] Sample {idx} failed. Error: {str(exc)}")
            annotation["output"] = f"[ERROR] {type(exc).__name__}: {str(exc)}"
            annotation["error"] = type(exc).__name__
            results_file.write(json.dumps(annotation) + "\n")
            results_file.flush()
            continue

    results_file.close()

    total_time = time.time() - total_start_time
    time_stats["total_time"] = round(total_time, 3)
    time_stats["avg_time_per_sample"] = round(total_time / len(annotations), 3) if len(annotations) > 0 else 0.0

    print("\n" + "=" * 60)
    print("Inference Statistics Summary:")
    print("=" * 60)
    print(f"Total Samples: {time_stats['total_samples']}")
    print(f"Total Time   : {time_stats['total_time']:.3f} s ({time_stats['total_time'] / 60:.2f} min)")
    print(f"Avg Time/Item: {time_stats['avg_time_per_sample']:.3f} s")
    print(f"Entered method pipeline   : {pipeline_called_count}")
    print(f"Skipped crop after filter: {skip_crop_after_filter}")
    print(f"Crop pipeline runs       : {crop_count}")
    if time_stats["sample_times"]:
        times = [sample["time"] for sample in time_stats["sample_times"]]
        print(f"Fastest      : {min(times):.3f} s")
        print(f"Slowest      : {max(times):.3f} s")
    print("=" * 60)
    print("no_cropping_count", no_cropping_count)
