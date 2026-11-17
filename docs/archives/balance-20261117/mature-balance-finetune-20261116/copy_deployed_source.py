import ast
import hashlib
from pathlib import Path

source=Path('$HOME/projects/tenny-torque200/src/force_teacher/ppo.py')
target=Path(__file__).resolve().parent/'deployed-force-ppo.py'
target.write_bytes(source.read_bytes())
text=target.read_text()
tree=ast.parse(text)
print('sha256='+hashlib.sha256(source.read_bytes()).hexdigest())
for node in tree.body:
    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ('train_student','restore_fixed_normalizer'):
        print(ast.get_source_segment(text,node))
