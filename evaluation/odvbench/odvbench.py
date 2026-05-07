#!/usr/bin/env python3
"""
ODVBench evaluation for Qwen2.5-VL-SaveMem.

Single file handles both single-GPU and multi-GPU execution.
"""

import argparse
import json
import logging
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

from qwen_vl_utils_savemem import process_vision_info
from qwen2_5_vl_savemem import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


class ODVBenchEvaluator:
    """Evaluator for ODVBench using Qwen2.5-VL-SaveMem."""

    PROMPT_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {question}

Options:
{options}

The best option is:"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._inference_count = 0
        self._setup_logging()
        self._load_model()

    def _setup_logging(self):
        log_handlers = []
        if self.args.log_path:
            Path(self.args.log_path).parent.mkdir(parents=True, exist_ok=True)
            log_handlers.append(logging.FileHandler(self.args.log_path))
        log_handlers.append(logging.StreamHandler())
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=log_handlers,
        )
        self.logger = logging.getLogger(__name__)

    def _load_model(self):
        self.logger.info(f"Loading SaveMem model: {self.args.ckpt_path}")
        torch.manual_seed(1234)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.args.ckpt_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            self.args.ckpt_path,
            min_pixels=16 * 28 * 28,
            max_pixels=self.args.max_pixels,
        )
        # Set tokenizer for pseudo-question semantic scoring in SaveMem
        if hasattr(self.model.model.savemem, 'set_tokenizer'):
            self.model.model.savemem.set_tokenizer(self.processor.tokenizer)
        self.logger.info("SaveMem model loaded successfully")

    @staticmethod
    def _format_candidates(candidates):
        """Format a list of candidate strings into lettered options."""
        formatted = []
        for i, opt in enumerate(candidates):
            letter = chr(65 + i)
            formatted.append(f"{letter}. {opt}")
        return '\n'.join(formatted)

    @staticmethod
    def _answer_to_letter(answer, candidates):
        """Convert the ground-truth answer string to a letter (A/B/C/D)."""
        for i, cand in enumerate(candidates):
            if cand.strip() == answer.strip():
                return chr(65 + i)
        return answer.strip()

    def _format_prompt(self, question, candidates):
        options_str = self._format_candidates(candidates)
        return self.PROMPT_TEMPLATE.format(question=question, options=options_str)

    def _extract_answer(self, response):
        import re
        response = response.strip()
        patterns = [
            r"option\s*([A-D])",
            r"([A-D])\s*is\s*the\s*best",
            r"answer\s*is\s*([A-D])",
            r"([A-D])\s*\)",
            r"^([A-D])$",
            r"option is\s*([A-D])",
            r"\(([A-D])\)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        match = re.search(r"[A-D]", response)
        return match.group(0) if match else ""

    # ------------------------------------------------------------------ #
    #  Data loading                                                       #
    # ------------------------------------------------------------------ #
    def _load_data(self):
        """Load ODVBench JSON and return list of items."""
        with open(self.args.task_json, 'r') as f:
            data = json.load(f)
        if self.args.start_sample > 0:
            data = data[self.args.start_sample:]
            self.logger.info(f"Starting from sample index {self.args.start_sample}")
        if self.args.max_sample is not None and self.args.max_sample > 0:
            data = data[:self.args.max_sample]
            self.logger.info(f"Limiting evaluation to {self.args.max_sample} samples")
        self.logger.info(f"Loaded {len(data)} questions")
        return data

    # ------------------------------------------------------------------ #
    #  Core inference                                                     #
    # ------------------------------------------------------------------ #
    def _preprocess(self, video_path, prompt, start_time, end_time):
        fps = float(self.args.fps)
        anchor_end = False
        if getattr(self.args, 'time_window_size', None) is not None and self.args.time_window_size > 0:
            start_time = max(0.0, end_time - float(self.args.time_window_size))
            anchor_end = True

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": 16 * 28 * 28,
                    "max_pixels": int(self.args.max_pixels),
                    "max_frames": int(self.args.max_num_frames),
                    "min_frames": 2,
                    "fps": fps,
                    "video_start": start_time,
                    "video_end": end_time,
                    "anchor_end": anchor_end,
                },
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs_cpu = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        del image_inputs, video_inputs
        return inputs_cpu

    def _run_inference(self, inputs_cpu):
        inputs = inputs_cpu.to(self.device, non_blocking=True)
        del inputs_cpu

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                use_savemem=self.args.use_savemem,
                memory_drop_method=self.args.frame_sampling,
                short_frames=self.args.short_frames,
                medium_frames=self.args.medium_frames,
                recency_gate_drop_ratio=self.args.recency_gate_drop_ratio,
                max_memory_tokens=self.args.max_memory_tokens,
                bbox_attention_bias=self.args.bbox_attention_bias,
                pair_sim_threshold=self.args.pair_sim_threshold,
                save_path=self.args.save_path,
            )

        input_len = inputs.input_ids.shape[1]
        del inputs
        output_ids = generated_ids[0][input_len:]
        del generated_ids
        self._inference_count += 1
        if self._inference_count % 10 == 0:
            torch.cuda.empty_cache()

        return self.processor.decode(output_ids, skip_special_tokens=True)

    def _process_video(self, video_path, prompt, start_time, end_time):
        fps = float(self.args.fps)
        anchor_end = False
        if getattr(self.args, 'time_window_size', None) is not None and self.args.time_window_size > 0:
            start_time = max(0.0, end_time - float(self.args.time_window_size))
            anchor_end = True

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": 16 * 28 * 28,
                    "max_pixels": int(self.args.max_pixels),
                    "max_frames": int(self.args.max_num_frames),
                    "min_frames": 2,
                    "fps": fps,
                    "video_start": start_time,
                    "video_end": end_time,
                    "anchor_end": anchor_end,
                },
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.device, non_blocking=True)
        del image_inputs, video_inputs

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                use_savemem=self.args.use_savemem,
                memory_drop_method=self.args.frame_sampling,
                short_frames=self.args.short_frames,
                medium_frames=self.args.medium_frames,
                recency_gate_drop_ratio=self.args.recency_gate_drop_ratio,
                max_memory_tokens=self.args.max_memory_tokens,
                bbox_attention_bias=self.args.bbox_attention_bias,
                pair_sim_threshold=self.args.pair_sim_threshold,
                save_path=self.args.save_path,
            )

        input_len = inputs.input_ids.shape[1]
        del inputs
        output_ids = generated_ids[0][input_len:]
        del generated_ids
        self._inference_count += 1
        if self._inference_count % 10 == 0:
            torch.cuda.empty_cache()

        return self.processor.decode(output_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    #  Result recording                                                   #
    # ------------------------------------------------------------------ #
    def _record_result(self, item, idx, response, f, stats):
        gt_letter = self._answer_to_letter(item['answer'], item['candidates'])
        predicted = self._extract_answer(response)
        correct = predicted == gt_letter

        result = {
            'index': idx,
            'task': item['task'],
            'subtask': item['subtask'],
            'question': item['question'],
            'answer': item['answer'],
            'answer_letter': gt_letter,
            'candidates': item['candidates'],
            'predicted_answer': predicted,
            'response': response,
            'correct': correct,
            'video': item['video'],
        }
        f.write(json.dumps(result) + '\n')
        f.flush()

        correct_mark = "✓" if correct else "✗"
        self.logger.info(
            f"[{idx}] Q: {item['question'][:80]} | "
            f"GT: {gt_letter} | Pred: {predicted} | Response: {response!r} {correct_mark}"
        )

        stats['overall']['total'] += 1
        stats[item['task']]['total'] += 1
        stats[item['subtask']]['total'] += 1
        if correct:
            stats['overall']['correct'] += 1
            stats[item['task']]['correct'] += 1
            stats[item['subtask']]['correct'] += 1

    # ------------------------------------------------------------------ #
    #  Evaluation loops                                                   #
    # ------------------------------------------------------------------ #
    def evaluate(self):
        data = self._load_data()
        stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        Path(self.args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

        with open(self.args.output_jsonl, 'w') as f:
            if getattr(self.args, 'parallel_inference', False):
                self._evaluate_parallel(data, stats, f)
                self._log_results(stats)
                self._save_summary(stats)
                return

            for idx, item in enumerate(tqdm(data, desc="Evaluating")):
                video_path = os.path.join(self.args.video_dir, item['video'])
                if not os.path.exists(video_path):
                    self.logger.warning(f"Video not found: {video_path}")
                    continue

                prompt = self._format_prompt(item['question'], item['candidates'])
                start_time = item.get('start', 0.0)
                end_time = item['end']

                try:
                    response = self._process_video(
                        video_path, prompt, start_time, end_time,
                    )
                    self._record_result(item, idx, response, f, stats)
                except Exception as e:
                    self.logger.error(f"Error processing item {idx}: {e}")
                    continue

        self._log_results(stats)
        self._save_summary(stats)

    def _evaluate_parallel(self, data, stats, f):
        valid_items = []
        for idx, item in enumerate(data):
            video_path = os.path.join(self.args.video_dir, item['video'])
            if os.path.exists(video_path):
                valid_items.append((idx, item, video_path))
            else:
                self.logger.warning(f"Video not found: {video_path}")

        if not valid_items:
            return

        def _submit(executor, item, video_path):
            prompt = self._format_prompt(item['question'], item['candidates'])
            start_time = item.get('start', 0.0)
            end_time = item['end']
            return executor.submit(self._preprocess, video_path, prompt, start_time, end_time)

        with ThreadPoolExecutor(max_workers=1) as executor:
            prefetch_future = _submit(executor, valid_items[0][1], valid_items[0][2])

            for i, (idx, item, _) in enumerate(tqdm(valid_items, desc="Evaluating [parallel]")):
                inputs_cpu = prefetch_future.result()

                if i + 1 < len(valid_items):
                    prefetch_future = _submit(executor, valid_items[i + 1][1], valid_items[i + 1][2])

                try:
                    response = self._run_inference(inputs_cpu)
                    self._record_result(item, idx, response, f, stats)
                except Exception as e:
                    self.logger.error(f"Error processing item {idx}: {e}")

    # ------------------------------------------------------------------ #
    #  Reporting                                                          #
    # ------------------------------------------------------------------ #
    def _log_results(self, stats):
        self.logger.info("=" * 50)
        self.logger.info("EVALUATION RESULTS (ODVBench)")
        self.logger.info("=" * 50)
        for key, counts in stats.items():
            if counts['total'] > 0:
                acc = counts['correct'] / counts['total'] * 100
                self.logger.info(f"{key}: {counts['correct']}/{counts['total']} = {acc:.2f}%")

    def _save_summary(self, stats):
        summary = {
            'model': self.args.ckpt_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': {},
            'memory_args': {
                'use_savemem': self.args.use_savemem,
                'frame_sampling': self.args.frame_sampling,
                'short_frames': self.args.short_frames,
                'medium_frames': self.args.medium_frames,
            },
        }
        for key, counts in stats.items():
            if counts['total'] > 0:
                summary['results'][key] = {
                    'total': counts['total'],
                    'correct': counts['correct'],
                    'accuracy': counts['correct'] / counts['total'],
                }
        summary_path = self.args.output_jsonl.replace('.jsonl', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Summary saved to: {summary_path}")


# ================================================================== #
#  CLI                                                                 #
# ================================================================== #
def build_argparser():
    parser = argparse.ArgumentParser(description='ODVBench Evaluation (Qwen2.5-VL-SaveMem)')
    parser.add_argument('--ckpt_path', required=True, help='Model checkpoint path')
    parser.add_argument('--task_json', type=str, default='/workspace/Proj-Apr-2026/mem/odv/ODVbench.json',
                        help='ODVBench JSON file')
    parser.add_argument('--video_dir', required=True, help='Root video directory (videos are relative to this)')
    parser.add_argument('--result_dir', type=str, default=None, help='Result directory (optional)')
    parser.add_argument('--run_name', type=str, default='odvbench', help='Run name for result directory')
    parser.add_argument('--output_jsonl', type=str, default=None, help='Output JSONL (auto-set if omitted)')
    parser.add_argument('--log_path', type=str, default=None, help='Log path (auto-set if omitted)')

    # Memory flags
    parser.add_argument('--use_savemem', action='store_true', help='Enable memory during generation')
    parser.add_argument('--frame_sampling', choices=['uniform', 'tail'], default='uniform',
                        help='Frame sampling method')
    parser.add_argument('--short_frames', type=int, default=8, help='Number of short-term frames to keep fully')
    parser.add_argument('--medium_frames', type=int, default=16, help='Mid-term queue upper bound')
    parser.add_argument('--recency_gate_drop_ratio', type=float, default=None,
                        help='Recency gate drop ratio (0.0=always short-only, 0.9=normal gate, None=always full-memory)')
    parser.add_argument('--max_memory_tokens', type=int, default=None,
                        help='Max memory tokens for SaveMem (default: 2048 in SaveMem)')
    parser.add_argument('--bbox_attention_bias', type=float, default=None,
                        help='Positive attention bias for visual tokens inside query-referenced bbox regions')
    parser.add_argument('--pair_sim_threshold', type=float, default=None,
                        help='Fixed similarity threshold to bypass Otsu')

    # Video flags
    parser.add_argument('--fps', type=float, default=1.0, help='Frames per second for sampling')
    parser.add_argument('--max_num_frames', '--max-num-frames', dest='max_num_frames', type=int, default=256,
                        help='Maximum frames to sample')
    parser.add_argument('--start_sample', type=int, default=0,
                        help='Index of the first sample to start from (0-based)')
    parser.add_argument('--max_pixels', '--max-pixels', dest='max_pixels', type=int, default=256 * 28 * 28,
                        help='Maximum pixels per frame')
    parser.add_argument('--time_window_size', type=float, default=None,
                        help='Seconds to look back from end timestamp; None = use start/end from data')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Optional jsonl to append memory/drop stats per sample')
    parser.add_argument('--max_sample', type=int, default=None,
                        help='Maximum number of samples to evaluate; None = use all')

    # Parallel inference flag
    parser.add_argument('--parallel_inference', action='store_true',
                        help='Overlap CPU preprocessing with GPU inference via a prefetch thread.')

    # Multi-GPU flags
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU mode')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs for multi-GPU mode')
    parser.add_argument('--dry_run', action='store_true', help='Show GPU splits without running (multi-GPU)')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return parser


def _default_paths(args):
    ts = time.strftime('%Y%m%d_%H%M%S')
    result_dir = Path(args.result_dir) if args.result_dir else Path(f"eval_results/odvbench/{args.run_name}_{ts}")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "log").mkdir(exist_ok=True)
    (result_dir / "output").mkdir(exist_ok=True)
    output_jsonl = args.output_jsonl or str(result_dir / "output" / f"results_{ts}.jsonl")
    log_path = args.log_path or str(result_dir / "log" / f"eval_{ts}.log")
    return result_dir, output_jsonl, log_path


def run_single(args):
    if args.output_jsonl and args.log_path:
        result_dir = Path(args.result_dir) if args.result_dir else Path(args.output_jsonl).parent.parent
        output_jsonl = args.output_jsonl
        log_path = args.log_path
    else:
        result_dir, output_jsonl, log_path = _default_paths(args)
        args.output_jsonl = output_jsonl
        args.log_path = log_path
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    evaluator = ODVBenchEvaluator(args)
    evaluator.evaluate()
    return result_dir


def _split_data(data, num_gpus):
    if num_gpus <= 1:
        return [data]
    chunk = len(data) // num_gpus
    splits = []
    for i in range(num_gpus):
        start = i * chunk
        end = start + chunk if i < num_gpus - 1 else len(data)
        splits.append(data[start:end])
    return splits


def run_multi(args):
    with open(args.task_json, 'r') as f:
        data = json.load(f)
    if args.start_sample > 0:
        data = data[args.start_sample:]
        print(f"[Multi-GPU] Starting from sample index {args.start_sample}")
    if args.max_sample is not None and args.max_sample > 0:
        data = data[:args.max_sample]
        print(f"[Multi-GPU] Limiting evaluation to {args.max_sample} samples")

    splits = _split_data(data, args.num_gpus)
    result_dir, _, _ = _default_paths(args)
    temp_dir = result_dir / "tmp"
    temp_dir.mkdir(exist_ok=True)

    def launch_worker(gpu_id, split_data):
        if len(split_data) == 0:
            return True
        split_json = temp_dir / f"split_{gpu_id}.json"
        with open(split_json, 'w') as f:
            json.dump(split_data, f)

        output_jsonl = result_dir / "output" / f"gpu_{gpu_id}.jsonl"
        log_path = result_dir / "log" / f"gpu_{gpu_id}.log"

        cmd = [
            "python", "evaluation/odvbench/odvbench.py",
            "--ckpt_path", args.ckpt_path,
            "--task_json", str(split_json),
            "--video_dir", args.video_dir,
            "--output_jsonl", str(output_jsonl),
            "--log_path", str(log_path),
            "--worker",
        ]

        if args.use_savemem:
            cmd.append("--use_savemem")
        cmd += ["--frame_sampling", args.frame_sampling]
        cmd += ["--short_frames", str(args.short_frames), "--medium_frames", str(args.medium_frames)]
        if args.recency_gate_drop_ratio is not None:
            cmd += ["--recency_gate_drop_ratio", str(args.recency_gate_drop_ratio)]
        if args.max_memory_tokens is not None:
            cmd += ["--max_memory_tokens", str(args.max_memory_tokens)]
        if args.bbox_attention_bias is not None:
            cmd += ["--bbox_attention_bias", str(args.bbox_attention_bias)]
        if args.pair_sim_threshold is not None:
            cmd += ["--pair_sim_threshold", str(args.pair_sim_threshold)]
        if getattr(args, 'parallel_inference', False):
            cmd.append("--parallel_inference")
        cmd += [
            "--fps", str(args.fps),
            "--max-num-frames", str(args.max_num_frames),
            "--max-pixels", str(args.max_pixels),
        ]
        if args.time_window_size is not None:
            cmd += ["--time_window_size", str(args.time_window_size)]
        if args.save_path:
            cmd += ["--save_path", str(args.save_path)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {gpu_id}] start {len(split_data)} samples")
        res = subprocess.run(cmd, env=env)
        return res.returncode == 0

    if args.dry_run:
        for gid, split in enumerate(splits):
            print(f"GPU {gid}: {len(split)} samples")
        return result_dir

    with ThreadPoolExecutor(max_workers=args.num_gpus) as pool:
        results = list(pool.map(lambda p: launch_worker(*p), enumerate(splits)))

    if not all(results):
        raise RuntimeError("Some GPU workers failed")

    merged_file = result_dir / "merged_results.jsonl"
    with open(merged_file, "w") as fout:
        for jsonl_file in sorted((result_dir / "output").glob("gpu_*.jsonl")):
            with open(jsonl_file) as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
    print(f"Merged results -> {merged_file}")

    # ── Aggregate stats from merged results ──────────────────────────
    stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    with open(merged_file) as fin:
        for line in fin:
            if not line.strip():
                continue
            result = json.loads(line)
            stats['overall']['total'] += 1
            task = result.get('task', 'unknown')
            subtask = result.get('subtask', 'unknown')
            stats[task]['total'] += 1
            stats[subtask]['total'] += 1
            if result.get('correct'):
                stats['overall']['correct'] += 1
                stats[task]['correct'] += 1
                stats[subtask]['correct'] += 1

    print("=" * 50)
    print("MERGED EVALUATION RESULTS (ODVBench)")
    print("=" * 50)
    for key, counts in stats.items():
        if counts['total'] > 0:
            acc = counts['correct'] / counts['total'] * 100
            print(f"{key}: {counts['correct']}/{counts['total']} = {acc:.2f}%")

    # Save merged summary
    summary = {
        'model': args.ckpt_path,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': {},
    }
    for key, counts in stats.items():
        if counts['total'] > 0:
            summary['results'][key] = {
                'total': counts['total'],
                'correct': counts['correct'],
                'accuracy': counts['correct'] / counts['total'],
            }
    summary_path = result_dir / "merged_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Merged summary saved to: {summary_path}")

    return result_dir


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if args.multi_gpu and not args.worker:
        result_dir = run_multi(args)
    else:
        result_dir = run_single(args)
    print(f"Finished. Results in: {result_dir}")


if __name__ == '__main__':
    main()
