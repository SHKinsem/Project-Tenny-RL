import ast
import json
from pathlib import Path

src = Path('$HOME/projects/tenny-torque200/src')
lib = Path('$HOME/projects/tenny-reference/.venv-gpu/lib/python3.12/site-packages')
for path, names in ((src/'force_teacher/env.py', ['student_config', '__init__']),
                    (src/'force_teacher/ppo.py', ['train_student', 'restore_fixed_normalizer']),
                    (src/'force_teacher/networks.py', ['make_ppo_networks']),
                    (lib/'brax/training/agents/ppo/train.py', ['train'])):
    text = path.read_text()
    tree = ast.parse(text)
    print(str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            print(ast.unparse(node.args))
            if node.name in ('student_config', 'train_student'):
                print(ast.get_source_segment(text, node)[:12000])
    if path.name == 'train.py':
        lines = text.splitlines()
        indexes = {index for i, line in enumerate(lines) if any(word in line for word in (
            'Evaluator(', 'policy_params_fn(', 'num_evals_after_init', 'num_training_steps_per_epoch'))
                   for index in range(max(0,i-2), min(len(lines),i+5))}
        print('\n'.join(f'{i+1}: {lines[i]}' for i in sorted(indexes)))
cache = Path('$HOME/projects/tenny-torque200/artifacts/force-teacher-pretrain/verified-bootstrap/d9772a5a5cf30d5c8850b13226fe075f0c494c1017138a745b6936a882227990')
print(json.dumps({'initial_checkpoints': [str(p) for p in cache.glob('verification-*/checkpoint')]}))
