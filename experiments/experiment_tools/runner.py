"""Sequential, budgeted subprocess jobs with separate orchestration artifacts.

Commands are argv lists passed with shell=False.  Only whole argv tokens
{python} and {run_dir} are substituted; Python code, braces, environment syntax
and other arguments remain literal.  Each job owns runs/<id>, while runner
records/logs live in jobs/<id>.json and logs/<id>.log.  Timeout handling kills
and waits for the direct child through subprocess.Popen; it does not claim to
terminate detached grandchildren or integrate with a production controller.
"""
from collections.abc import Mapping
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .freeze import _validate_snapshot,verify_inputs
from .io import load_json


def _require(condition,message):
    if not condition:raise ValueError(message)


def _positive(value):
    if type(value) not in (int,float):return False
    try:return math.isfinite(value) and value>0
    except OverflowError:return False


def _text(value,*,empty=False):
    return isinstance(value,str) and (empty or bool(value)) and '\0' not in value


def _job_id(value):
    if not _text(value) or value!=value.strip() or len(value)>128:return False
    if value in ('.','..') or value[-1] in ('.',' ') or any(ord(c)<32 or c in '/\\:<>"|?*' for c in value):return False
    stem=value.split('.',1)[0].upper()
    return stem not in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}


def validate_plan(plan):
    """Validate and copy a JSON job plan without reading inputs or spawning jobs.

    cwd defaults to run_dir; its exact {run_dir} token is also supported.  Other
    cwd strings stay literal.  Relative expected-result paths are rooted at the
    private run_dir, and only their leading {run_dir}/ marker is expanded.
    Result checks address dotted object keys and compare JSON values (booleans
    remain distinct from numbers).  No directory creation occurs here.
    """
    _require(isinstance(plan,Mapping),'Plan must be an object')
    allowed={'schema_version','kind','max_total_seconds','jobs','frozen_inputs'}
    _require(set(plan)<=allowed and {'schema_version','kind','max_total_seconds','jobs'}<=set(plan),
             'Unknown or missing experiment plan fields')
    _require(type(plan['schema_version']) is int and plan['schema_version']==1 and plan['kind']=='experiment_jobs',
             'Need schema_version 1 and kind experiment_jobs')
    _require(_positive(plan['max_total_seconds']),'max_total_seconds must be finite and positive')
    _require(isinstance(plan['jobs'],list) and bool(plan['jobs']),'Plan needs at least one job')
    seen=set()
    for job in plan['jobs']:
        _require(isinstance(job,dict),'Each job must be an object')
        _require(set(job)<={'id','command','timeout_seconds','cwd','expected_result'}
                 and {'id','command','timeout_seconds'}<=set(job),'Unknown or missing job fields')
        identity=job['id'];_require(_job_id(identity),'Unsafe job id: '+str(identity))
        _require(identity.casefold() not in seen,'Duplicate job id: '+identity);seen.add(identity.casefold())
        command=job['command']
        _require(isinstance(command,list) and bool(command) and _text(command[0])
                 and all(_text(value,empty=True) for value in command),'command must be a nonempty argv string list')
        _require(_positive(job['timeout_seconds']),'Job timeout_seconds must be finite and positive')
        if 'cwd' in job:_require(_text(job['cwd']),'cwd must be a nonempty literal path or {run_dir}')
        if 'expected_result' in job:
            result=job['expected_result']
            _require(isinstance(result,dict) and set(result)=={'path','checks'} and _text(result['path'])
                     and isinstance(result['checks'],dict),'expected_result needs path and checks')
            _require(all(_text(key) and all(key.split('.')) for key in result['checks']),
                     'Result check keys must be nonempty dotted object paths')
    if 'frozen_inputs' in plan:_validate_snapshot(plan['frozen_inputs'])
    try:return json.loads(json.dumps(dict(plan),ensure_ascii=False,allow_nan=False))
    except (TypeError,ValueError,RecursionError) as error:raise ValueError('Plan must contain finite JSON values') from error


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path,value):
    """Replace only runner-owned status files, keeping readers off partial JSON."""
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                         prefix='.'+path.name+'.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
        os.replace(temporary,path)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


def _persist(output,state,started):
    state['elapsed_seconds']=time.monotonic()-started
    state['completed_jobs']=sum(job['phase']=='completed' for job in state['jobs'])
    for job in state['jobs']:_write_json(output/'jobs'/(job['id']+'.json'),job)
    _write_json(output/'status.json',state)


def _log_event(stream,event):
    stream.write(('\n[experiment runner] '+json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
    stream.flush()


def _same_value(actual,expected):
    if isinstance(actual,bool) or isinstance(expected,bool):return type(actual) is type(expected) and actual==expected
    if isinstance(actual,dict) and isinstance(expected,dict):
        return actual.keys()==expected.keys() and all(_same_value(actual[key],value) for key,value in expected.items())
    if isinstance(actual,list) and isinstance(expected,list):
        return len(actual)==len(expected) and all(_same_value(a,b) for a,b in zip(actual,expected))
    if isinstance(actual,(int,float)) and isinstance(expected,(int,float)):return actual==expected
    return type(actual) is type(expected) and actual==expected


def _check_result(spec,run_dir):
    text=spec['path']
    if text=='{run_dir}':path=run_dir
    elif text.startswith(('{run_dir}/','{run_dir}\\')):path=Path(str(run_dir)+text[len('{run_dir}'):])
    else:
        path=Path(text)
        if not path.is_absolute():path=run_dir/path
    path=path.resolve()
    result=load_json(path)
    _require(isinstance(result,dict),'Expected result must be a JSON object: '+str(path))
    checked={}
    for key,expected in spec['checks'].items():
        actual=result
        for segment in key.split('.'):
            _require(isinstance(actual,dict) and segment in actual,'Missing expected result field: '+key)
            actual=actual[segment]
        _require(_same_value(actual,expected),'Expected result check failed: '+key)
        checked[key]=actual
    return dict(path=str(path),phase='verified',checks=checked)


def run_matrix(plan,output_dir):
    """Execute once in order; return a completed/failed JSON status, never retry.

    Invalid plans and nonempty outputs raise before execution.  Runtime failures
    are persisted and returned with phase=failed, and remaining jobs become
    not_started.  Callers such as CLIs should translate that phase to a nonzero
    exit code.  Total wall time includes input/result verification; filesystem
    operations are not preempted, but an overrun fails rather than starting a
    later job.  Child logs preserve merged stdout/stderr bytes.
    """
    plan=validate_plan(plan)
    output=Path(output_dir)
    _require(not output.is_symlink(),'Output directory must not be a symlink')
    if output.exists():
        _require(output.is_dir() and not any(output.iterdir()),'Output directory must be new or empty')
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    # Exclusive creation also prevents two simultaneous callers claiming the
    # same initially empty directory.  The saved plan is part of its history.
    try:
        with (output/'plan.json').open('x',encoding='utf-8') as stream:
            json.dump(plan,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    except FileExistsError as error:raise ValueError('Output directory was already claimed') from error
    for name in ('runs','jobs','logs'):(output/name).mkdir()
    started=time.monotonic();deadline=started+plan['max_total_seconds']
    state=dict(schema_version=1,kind='experiment_jobs_result',phase='running',output_dir=str(output),
        started_at=_utc(),finished_at=None,max_total_seconds=plan['max_total_seconds'],
        elapsed_seconds=0.,completed_jobs=0,current_job_id=None,
        process_scope='Direct child only; detached grandchildren are not managed.',jobs=[])
    for job in plan['jobs']:
        run_dir=output/'runs'/job['id']
        state['jobs'].append(dict(id=job['id'],phase='pending',run_dir=str(run_dir),
            status_path=str(output/'jobs'/(job['id']+'.json')),log_path=str(output/'logs'/(job['id']+'.log')),
            planned_command=job['command'],timeout_seconds=job['timeout_seconds'],started_at=None,
            finished_at=None,elapsed_seconds=0.,returncode=None,process_started=False,errors=[]))
    _persist(output,state,started)
    for index,(job,record) in enumerate(zip(plan['jobs'],state['jobs'])):
        state['current_job_id']=job['id'];job_started=time.monotonic();record['started_at']=_utc()
        record['phase']='running';run_dir=Path(record['run_dir'])
        record['command']=[sys.executable if value=='{python}' else str(run_dir) if value=='{run_dir}' else value
                           for value in job['command']]
        record['cwd']=str(run_dir if job.get('cwd','{run_dir}')=='{run_dir}' else Path(job['cwd']).absolute())
        _persist(output,state,started)
        with Path(record['log_path']).open('wb') as log:
            _log_event(log,dict(event='starting',id=job['id'],command=record['command'],cwd=record['cwd']))
            def failure(kind,message):
                record['errors'].append(dict(kind=kind,message=str(message)))
            try:run_dir.mkdir()
            except OSError as error:failure('setup_error',error)
            frozen=plan.get('frozen_inputs')
            if frozen is not None:
                try:record['frozen_before']=verify_inputs(frozen)
                except (ValueError,OSError) as error:failure('input_changed',error)
            remaining=deadline-time.monotonic()
            record['remaining_budget_before_seconds']=max(0.,remaining)
            if remaining<=0:failure('total_budget_exceeded','Total wall-time budget exhausted before launching job')
            if not record['errors']:
                timeout=min(job['timeout_seconds'],remaining);record['effective_timeout_seconds']=timeout
                _persist(output,state,started)
                process=None;launched=time.monotonic()
                try:
                    process=subprocess.Popen(record['command'],cwd=record['cwd'],stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,shell=False)
                    record.update(process_started=True,pid=process.pid)
                    _persist(output,state,started)
                    record['returncode']=process.wait(timeout=max(0.,timeout-(time.monotonic()-launched)))
                    if process.returncode!=0:failure('nonzero_exit','Child exited with code '+str(process.returncode))
                except subprocess.TimeoutExpired:
                    process.kill();record['returncode']=process.wait()
                    record['timed_out']=True
                    failure('total_budget_exceeded' if remaining<=job['timeout_seconds'] else 'job_timeout',
                            'Direct child exceeded effective timeout of '+str(timeout)+' seconds and was killed/reaped')
                except OSError as error:failure('launch_error',error)
                except KeyboardInterrupt:
                    failure('interrupted','Job execution interrupted')
                finally:
                    if process is not None and process.poll() is None:
                        process.kill();record['returncode']=process.wait()
                if not record['errors'] and 'expected_result' in job:
                    try:record['expected_result']=_check_result(job['expected_result'],run_dir)
                    except (ValueError,OSError,RecursionError) as error:failure('result_check_failed',error)
            if frozen is not None:
                try:record['frozen_after']=verify_inputs(frozen)
                except (ValueError,OSError) as error:failure('input_changed',error)
            if time.monotonic()>deadline and not any(item['kind']=='total_budget_exceeded' for item in record['errors']):
                failure('total_budget_exceeded','Total wall-time budget exceeded during job/verification')
            record['phase']='failed' if record['errors'] else 'completed'
            record['finished_at']=_utc();record['elapsed_seconds']=time.monotonic()-job_started
            if record['errors']:record['error_kind']=record['errors'][0]['kind']
            _log_event(log,dict(event=record['phase'],id=record['id'],returncode=record['returncode'],errors=record['errors']))
        if record['phase']=='failed':
            state.update(phase='failed',failed_job_id=job['id'],error_kind=record['error_kind'],errors=record['errors'])
            for later in state['jobs'][index+1:]:
                later.update(phase='not_started',skip_reason='Earlier job failed: '+job['id'])
            break
        _persist(output,state,started)
    else:state['phase']='completed'
    state.update(current_job_id=None,finished_at=_utc())
    _persist(output,state,started)
    return state
