"""Sequential isolated processes for a frozen paired-seed reward matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .train import write_json


def run_matrix(manifest_path, output_dir):
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Use a fresh matrix output directory; failed attempts are retained')
    output_dir.mkdir(parents=True, exist_ok=True)
    module_root = Path(__file__).resolve().parent
    training_files = ('__init__.py', '__main__.py', 'matrix.py', 'train.py', 'legwheel_env.py', 'rewards.py')
    code_hashes = {name: hashlib.sha256((module_root/name).read_bytes()).hexdigest()
                   for name in training_files}
    receipt = dict(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), code_sha256=code_hashes)
    write_json(output_dir/'frozen-inputs.json', receipt)
    start = time.monotonic()
    completed = []
    for seed in manifest['seeds']:
        for arm in manifest['order_by_seed'][str(seed)]:
            if manifest_path.read_bytes() != manifest_bytes or any(
                hashlib.sha256((module_root/name).read_bytes()).hexdigest() != digest
                for name, digest in code_hashes.items()):
                raise ValueError('Frozen matrix input or experiment code changed')
            run_dir = output_dir/f'seed-{seed}'/arm
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            status = dict(phase='training', current_arm=arm, current_seed=seed,
                completed=completed, elapsed_seconds=time.monotonic()-start, **receipt)
            write_json(output_dir/'status.json', status)
            print(json.dumps(dict(event='start', arm=arm, seed=seed, completed=len(completed))), flush=True)
            command = [sys.executable, '-B', '-m', 'experiments.reward_ablation',
                       '--manifest', str(manifest_path), '--arm', arm, '--seed', str(seed),
                       '--output-dir', str(run_dir)]
            log_path = output_dir/f'seed-{seed}-{arm}.log'
            try:
                with log_path.open('w', encoding='utf-8') as stream:
                    process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                             timeout=600, check=False)
                if process.returncode != 0:
                    raise RuntimeError(f'{arm}/seed-{seed} failed; inspect {log_path}')
                result = json.loads((run_dir/'status.json').read_text())
                if result['phase'] != 'completed' or result['steps'] != manifest['steps_per_run']:
                    raise RuntimeError('Run did not complete the equal step budget')
                completed.append(result)
                print(json.dumps(dict(event='completed', arm=arm, seed=seed,
                                      steps=result['steps'], seconds=result['elapsed_seconds'])), flush=True)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                write_json(output_dir/'status.json', dict(status, phase='failed', error=str(exc),
                                                          elapsed_seconds=time.monotonic()-start))
                raise
    result = dict(phase='completed', completed=completed, total_runs=len(completed),
                  elapsed_seconds=time.monotonic()-start, **receipt)
    write_json(output_dir/'status.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    options = parser.parse_args()
    run_matrix(options.manifest, options.output_dir)
