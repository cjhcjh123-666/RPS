#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机抽取 COCO val2017 数据集的若干图片，复制到 demo 目录，并调用 repo 自带的 demo.py 批量推理。

相比旧版本，当前脚本具有以下改进：
1. 默认路径指向当前仓库 (RPS)，不会再引用 /regnet_sam_l；
2. 通过命令行参数自定义模型、配置、输出目录等信息；
3. 提供可选的目录清理、随机种子设置、输出日志保存等功能；
4. 单次运行仅初始化一次目录结构，确保与现有文件夹不冲突。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    default_repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="随机抽样 COCO val 图片并调用 demo.py 进行推理，可同时保存原图与可视化结果。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo,
        help="RPS 仓库根目录（包含 demo、configs、seg 等子目录）")

    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="推理使用的权重文件路径，默认为 repo 根目录下的 best_coco_panoptic_PQ_epoch_57.pth。")

    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="推理配置文件路径，默认使用 configs/reg_sam/regnet_dis.py。")

    parser.add_argument(
        "--coco-val-dir",
        type=Path,
        default=Path("/data/zhangyafei/RMP-SAM/data/coco/val2017"),
        help="COCO val2017 图片目录。")

    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="随机抽取的图片数量。")

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。")

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="所有输出（可视化结果、日志、选取列表）保存的根目录。默认使用 repo_root/outputs/coco_visualization。")

    parser.add_argument(
        "--demo-copy-dir",
        type=Path,
        default=None,
        help="原图复制到的目录，默认使用 repo_root/demo/coco_val_random。")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="指定已有图片目录，直接使用该目录下的图片，无需再次随机抽样。")

    parser.add_argument(
        "--list-file",
        type=Path,
        default=None,
        help="记录随机抽样结果的文本文件路径，默认位于 output_root 下。")

    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="可视化时的置信度阈值，会通过命令行传递给 demo.py。")

    parser.add_argument(
        "--palette",
        type=str,
        default="coco",
        help="可视化配色方案，将透传给 demo.py。")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="并行执行 demo.py 的线程数；1 表示串行。")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="用于推理的 GPU 编号，逗号分隔，例如 '0,1,2,3'。")
    parser.add_argument(
        "--procs-per-gpu",
        type=int,
        default=1,
        help="每块 GPU 同时运行的 demo.py 进程数。")
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="移除检测框与类别文本，可视化中仅保留 mask。")
    parser.add_argument(
        "--no-mask-boundary",
        action="store_true",
        help="移除 mask 外侧的白色边缘线。")
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=None,
        help="覆盖 mask 的透明度 (0~1)。")

    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="不清理 demo_copy_dir 和输出目录的历史文件。")

    parser.add_argument(
        "--log-json",
        action="store_true",
        help="将推理结果统计信息保存为 JSON 文件，存放在 output_root。")

    parser.add_argument(
        "--show-env",
        action="store_true",
        help="打印关键路径和环境信息，便于排查问题。")

    return parser.parse_args()


def ensure_path(path: Path, must_exist: bool = True) -> Path:
    """简单的路径存在性检查，便于提前给出明确错误信息。"""
    path = path.expanduser().resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    return path


def ensure_directories(
    repo_root: Path,
    output_root: Path,
    demo_copy_dir: Path,
    clean: bool,
    create_demo_dir: bool = True,
) -> None:
    """创建并（可选）清理输出目录。"""
    # 创建基础目录
    output_root.mkdir(parents=True, exist_ok=True)
    if create_demo_dir:
        demo_copy_dir.mkdir(parents=True, exist_ok=True)

    # 可视化结果目录统一放在 output_root 下
    visual_dir = output_root / "visualized_results"
    visual_dir.mkdir(parents=True, exist_ok=True)

    if not clean:
        return

    def _safe_clear(target: Path) -> None:
        if not target.exists():
            return
        for item in target.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    if create_demo_dir:
        _safe_clear(demo_copy_dir)
    _safe_clear(visual_dir)


def collect_images(image_dir: Path, num_images: int, seed: int) -> List[Path]:
    """随机选取若干图片路径。"""
    candidates: List[Path] = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not candidates:
        raise RuntimeError(f"目录 {image_dir} 中未找到可用图片（扩展名 .jpg/.jpeg/.png）。")

    if num_images <= 0:
        raise ValueError("num_images 必须为正整数。")

    rng = random.Random(seed)
    if len(candidates) <= num_images:
        print(f"提示: 目录仅包含 {len(candidates)} 张图片，少于或等于请求数量，全部使用。")
        return candidates.copy()

    return rng.sample(candidates, num_images)


def copy_originals(selected_images: Sequence[Path], demo_copy_dir: Path) -> None:
    """将随机选取的图片复制到 demo_copy_dir。"""
    print("正在复制原图至 demo 目录...")
    for img in selected_images:
        dst = demo_copy_dir / img.name
        shutil.copy2(img, dst)
        print(f"已复制: {img.name}")


def save_selection(selected_images: Sequence[Path], list_file: Path) -> None:
    """将抽样结果保存为文本文件，便于复现。"""
    list_file.parent.mkdir(parents=True, exist_ok=True)
    with list_file.open("w", encoding="utf-8") as f:
        for img in selected_images:
            f.write(str(img) + "\n")
    print(f"随机图片列表已保存: {list_file}")


def run_demo_batch(
    selected_images: Sequence[Path],
    repo_root: Path,
    config_path: Path,
    model_path: Path,
    demo_copy_dir: Path,
    visual_dir: Path,
    score_thr: float,
    palette: str,
    num_workers: int,
    gpu_ids: Sequence[int],
    procs_per_gpu: int,
    no_bbox: bool,
    no_mask_boundary: bool,
    mask_alpha: Optional[float],
) -> dict:
    """调用 demo.py 对图片推理，可并行执行，并记录成功/失败数量与错误信息。"""
    env = os.environ.copy()
    repo_str = str(repo_root)
    original_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_str}{os.pathsep}{original_pythonpath}"
        if original_pythonpath else repo_str
    )

    success = 0
    failures: List[dict] = []

    demo_script = repo_root / "demo" / "demo.py"
    if not demo_script.exists():
        raise FileNotFoundError(f"未找到 demo.py，期望路径: {demo_script}")

    gpu_ids = list(gpu_ids)
    if not gpu_ids:
        gpu_ids = [0]
    procs_per_gpu = max(1, procs_per_gpu)
    max_threads = max(1, num_workers, len(gpu_ids) * procs_per_gpu)

    img_gpu_map = {
        img.name: gpu_ids[idx % len(gpu_ids)]
        for idx, img in enumerate(selected_images)
    }

    def _build_command(img_name: str) -> List[str]:
        assigned_gpu = img_gpu_map[img_name]
        command = [
            sys.executable,
            str(demo_script),
            str(demo_copy_dir / img_name),
            str(config_path),
            "--weights",
            str(model_path),
            "--out-dir",
            str(visual_dir),
            "--no-save-pred",
            "--pred-score-thr",
            str(score_thr),
            "--palette",
            palette,
            "--device",
            f"cuda:{assigned_gpu}",
        ]
        if no_bbox:
            command.append("--no-bbox")
        if no_mask_boundary:
            command.append("--no-mask-boundary")
        if mask_alpha is not None:
            command.extend(["--mask-alpha", f"{mask_alpha}"])
        return command

    def _run_single(idx: int, total: int, img: Path) -> tuple[str, subprocess.CompletedProcess[str]]:
        img_name = img.name
        print(f"推理进度 {idx}/{total}: {img_name}")
        command = _build_command(img_name)
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return img_name, completed

    if max_threads == 1:
        for idx, img in enumerate(selected_images, 1):
            img_name, completed = _run_single(idx, len(selected_images), img)
            if completed.returncode == 0:
                success += 1
            else:
                failure_record = {
                    "image": img_name,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
                failures.append(failure_record)
                print(f"推理失败: {img_name}")
                if completed.stdout:
                    print("stdout:\n", completed.stdout)
                if completed.stderr:
                    print("stderr:\n", completed.stderr)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        counter = threading.Lock()
        progress_state = {"idx": 0}

        def submit_task(img: Path):
            with counter:
                progress_state["idx"] += 1
                current_idx = progress_state["idx"]
            return executor.submit(_run_single, current_idx, len(selected_images), img)

        total = len(selected_images)
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            future_map = {
                submit_task(img): img
                for img in selected_images
            }
            for future in as_completed(future_map):
                img_name, completed = future.result()
                if completed.returncode == 0:
                    success += 1
                else:
                    failure_record = {
                        "image": img_name,
                        "returncode": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }
                    failures.append(failure_record)
                    print(f"推理失败: {img_name}")
                    if completed.stdout:
                        print("stdout:\n", completed.stdout)
                    if completed.stderr:
                        print("stderr:\n", completed.stderr)

    return {"success": success, "failed": len(failures), "failures": failures}


def dump_json_report(report: dict, output_root: Path) -> Path:
    """保存 JSON 报告，文件名带时间戳便于区分。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_root / f"visualize_report_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report_path


def main() -> None:
    args = parse_args()

    repo_root = ensure_path(args.repo_root)
    coco_val_dir = ensure_path(args.coco_val_dir)

    model_path = ensure_path(
        args.model_path or repo_root / "best_coco_panoptic_PQ_epoch_57.pth")
    config_path = ensure_path(
        args.config_path or repo_root / "configs/reg_sam/regnet_dis.py")

    output_root = (args.output_root or repo_root / "outputs" / "coco_visualization")
    output_root = output_root.expanduser().resolve()
    demo_copy_dir = (args.demo_copy_dir or repo_root / "demo" / "coco_val_random")
    demo_copy_dir = demo_copy_dir.expanduser().resolve()
    list_file = (args.list_file or output_root / "selected_images.txt")
    list_file = list_file.expanduser().resolve()

    visual_dir = output_root / "visualized_results"

    input_dir = args.input_dir
    skip_copy = False
    if input_dir is not None:
        input_dir = ensure_path(input_dir)
        selected_images = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not selected_images:
            raise RuntimeError(f"{input_dir} 中未找到可用图片。")
        if args.num_images > 0 and len(selected_images) > args.num_images:
            selected_images = selected_images[:args.num_images]
        if args.demo_copy_dir is None:
            demo_copy_dir = input_dir
        skip_copy = demo_copy_dir.resolve() == input_dir.resolve()
    else:
        selected_images = collect_images(
            image_dir=coco_val_dir,
            num_images=args.num_images,
            seed=args.seed,
        )

    if args.show_env:
        print("======= 环境信息 =======")
        print(f"repo_root      : {repo_root}")
        print(f"model_path     : {model_path}")
        print(f"config_path    : {config_path}")
        print(f"coco_val_dir   : {coco_val_dir}")
        print(f"output_root    : {output_root}")
        print(f"demo_copy_dir  : {demo_copy_dir}")
        print(f"list_file      : {list_file}")
        if input_dir is not None:
            print(f"input_dir      : {input_dir}")
        print("========================\n")

    ensure_directories(
        repo_root=repo_root,
        output_root=output_root,
        demo_copy_dir=demo_copy_dir,
        clean=(not args.no_clean) and (not skip_copy),
        create_demo_dir=not skip_copy,
    )

    print(f"共选择 {len(selected_images)} 张图片。")

    if not skip_copy:
        copy_originals(selected_images, demo_copy_dir)
    save_selection(selected_images, list_file)

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        gpu_ids = [0]

    stats = run_demo_batch(
        selected_images=selected_images,
        repo_root=repo_root,
        config_path=config_path,
        model_path=model_path,
        demo_copy_dir=demo_copy_dir,
        visual_dir=visual_dir,
        score_thr=args.score_thr,
        palette=args.palette,
        num_workers=args.num_workers,
        gpu_ids=gpu_ids,
        procs_per_gpu=args.procs_per_gpu,
        no_bbox=args.no_bbox,
        no_mask_boundary=args.no_mask_boundary,
        mask_alpha=args.mask_alpha,
    )

    print("\n任务完成。")
    print(f"推理成功: {stats['success']} 张")
    print(f"推理失败: {stats['failed']} 张")
    print(f"原图保存在: {demo_copy_dir}")
    print(f"可视化结果保存在: {visual_dir}")
    print(f"随机列表保存在: {list_file}")

    if args.log_json:
        report_path = dump_json_report(stats, output_root)
        print(f"推理报告已保存: {report_path}")


if __name__ == "__main__":
    main()

