import argparse
import json
from pathlib import Path
import sys
from . import freeze_inputs,verify_inputs,run_matrix,validate_plan,load_study,summarize,select_snapshot
from .io import load_json,save_json


def main(argv=None):
    parser=argparse.ArgumentParser(description='Offline comparisons, snapshot selection, input checks and explicitly requested jobs.')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['compare','normalize']:
        p=sub.add_parser(name);p.add_argument('input',type=Path);p.add_argument('--output',type=Path)
    p=sub.add_parser('select');p.add_argument('input',type=Path);p.add_argument('--condition',required=True);p.add_argument('--metric',required=True)
    p.add_argument('--partition',default='primary')
    p.add_argument('--require-full-parameters',action='store_true');p.add_argument('--min-step',type=int,default=1);p.add_argument('--output',type=Path)
    p=sub.add_parser('freeze');p.add_argument('paths',nargs='+',type=Path);p.add_argument('--output',type=Path)
    p=sub.add_parser('check');p.add_argument('input',type=Path);p.add_argument('--output',type=Path)
    p=sub.add_parser('run');p.add_argument('input',type=Path);p.add_argument('--output',type=Path);p.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    try:
        if getattr(args,'output',None) and getattr(args,'input',None) and args.output.resolve()==args.input.resolve():raise ValueError('Output must not overwrite input')
        if args.command in ('compare','normalize','select'):
            study=load_study(args.input)
            if args.command=='normalize':summarize(study);result=study
            elif args.command=='compare':
                result=summarize(study);result['source_provenance']=study.get('provenance')
            else:result=select_snapshot(study,condition=args.condition,metric=args.metric,partition=args.partition,require_full_parameters=args.require_full_parameters,min_step=args.min_step)
        elif args.command=='freeze':result=freeze_inputs(args.paths)
        elif args.command=='check':result=verify_inputs(load_json(args.input))
        else:
            plan=validate_plan(load_json(args.input))
            if args.dry_run:
                print(json.dumps(dict(dry_run=True,plan=plan),ensure_ascii=False,allow_nan=False,indent=2));return 0
            if args.output is None:raise ValueError('run needs a new --output directory, or --dry-run')
            result=run_matrix(plan,args.output)
            print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2));return 0 if result['phase']=='completed' else 1
        if args.output:save_json(args.output,result)
        else:print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))
        return 0
    except (ValueError,KeyError,TypeError,OSError) as error:
        print('experiment-tools: '+str(error),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
