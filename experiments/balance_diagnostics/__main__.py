"""Evaluate fixed native descriptors, or render an existing diagnostic result."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--manifest', type=Path, required=True)
    evaluate.add_argument('--output-dir', type=Path, required=True)
    evaluate.add_argument('--seconds', type=float, default=30.)
    evaluate.add_argument('--reference-seconds', type=float, default=6.)
    render = commands.add_parser('report')
    render.add_argument('results', type=Path)
    render.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from .report import write_report
        if args.command == 'evaluate':
            from .runner import evaluate_manifest
            bundle = evaluate_manifest(args.manifest, args.output_dir, seconds=args.seconds,
                                       reference_seconds=args.reference_seconds)
        else:
            bundle = json.loads(args.results.read_text(encoding='utf-8'))
            from .runner import write_json
            write_json(args.output_dir / 'results.json', bundle)
        print(json.dumps(write_report(bundle, args.output_dir), ensure_ascii=False), flush=True)
    except (ValueError, TypeError, OSError, KeyError, ImportError, RuntimeError) as exc:
        parser.exit(2, 'balance_diagnostics: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
