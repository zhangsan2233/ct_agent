import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.ctclip import CtClipRuntime


def cache_path_for(
    volume: Path,
    checkpoint: Path,
    cache_dir: Path,
    use_fp16: bool,
    variant: str,
) -> Path:
    volume_stat = volume.stat()
    checkpoint_stat = checkpoint.stat()
    fingerprint = json.dumps(
        {
            "version": 2,
            "variant": variant,
            "volume": str(volume.resolve()),
            "volume_size": volume_stat.st_size,
            "volume_mtime_ns": volume_stat.st_mtime_ns,
            "checkpoint_size": checkpoint_stat.st_size,
            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
            "fp16": use_fp16,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def read_cache(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {key: float(value) for key, value in payload["probabilities"].items()}
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def write_cache(path: Path, probabilities: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"probabilities": probabilities}, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def completed_cases(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("case_id") and not item.get("error"):
            completed.add(str(item["case_id"]))
    return completed


def run_batch(args: argparse.Namespace, runtime: CtClipRuntime) -> None:
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.resume:
        output_path.write_text("", encoding="utf-8")
    completed = completed_cases(output_path) if args.resume else set()

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit > 0:
        rows = rows[: args.limit]

    failures = 0
    cache_dir = Path(args.cache_dir)
    checkpoint = Path(args.checkpoint)
    with output_path.open("a", encoding="utf-8") as output:
        for index, row in enumerate(rows, start=1):
            case_id = str(row["case_id"])
            if case_id in completed:
                print(f"Skipped [{index}/{len(rows)}] {case_id}: output exists", flush=True)
                continue
            volume = Path(row["ct_volume_path"])
            if not volume.is_absolute():
                volume = PROJECT_ROOT / volume
            cache_path = cache_path_for(
                volume, checkpoint, cache_dir, args.fp16, args.variant
            )
            probabilities = read_cache(cache_path)
            cache_hit = probabilities is not None
            started = time.perf_counter()
            try:
                if probabilities is None:
                    probabilities = runtime.predict(str(volume))
                    write_cache(cache_path, probabilities)
                payload = {
                    "case_id": case_id,
                    "ct_volume_path": str(volume),
                    "probabilities": probabilities,
                    "cache_hit": cache_hit,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
                output.write(json.dumps(payload, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"Completed [{index}/{len(rows)}] {case_id} "
                    f"cache={cache_hit} latency_ms={payload['latency_ms']}",
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                payload = {
                    "case_id": case_id,
                    "ct_volume_path": str(volume),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                output.write(json.dumps(payload, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"Failed [{index}/{len(rows)}] {case_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
    print(f"Batch finished: total={len(rows)} failures={failures}", flush=True)
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--volume")
    source.add_argument("--manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variant", choices=("lipro", "zeroshot"), default="zeroshot")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output", default="artifacts/evaluation/ctclip_predictions.jsonl")
    parser.add_argument("--cache-dir", default="artifacts/ct_cache")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    runtime = CtClipRuntime(
        checkpoint=Path(args.checkpoint),
        source_dir=Path(args.source_dir),
        device=args.device,
        use_fp16=args.fp16,
        variant=args.variant,
    )
    if args.volume:
        print(json.dumps(runtime.predict(args.volume), sort_keys=True))
        return
    run_batch(args, runtime)


if __name__ == "__main__":
    main()
