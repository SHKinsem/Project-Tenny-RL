from orbax import checkpoint as ocp
import logging
import os
import jax
from typing import Any, Dict, Tuple, Union
from etils import epath
import numpy as np
import json
from brax.training.acme import running_statistics


def _restore_args_from_orbax_metadata(ckpt_path: epath.Path):
    """Builds a restore_args pytree matching the checkpoint structure.

    Orbax OCDBT checkpoints in this repo include a JSON `_METADATA` file with
    per-leaf key paths. We use it to construct a pytree of RestoreArgs so we can
    restore arrays as NumPy on CPU without requiring sharding/device info.
    """
    metadata_file = ckpt_path / '_METADATA'
    if not metadata_file.exists():
        return None

    try:
        metadata_json = json.loads(metadata_file.read_text())
    except Exception:
        return None

    tree_metadata = metadata_json.get('tree_metadata')
    if not isinstance(tree_metadata, dict) or not tree_metadata:
        return None

    paths = []
    for entry in tree_metadata.values():
        key_metadata = entry.get('key_metadata') if isinstance(entry, dict) else None
        if not isinstance(key_metadata, list) or not key_metadata:
            continue
        path = []
        for km in key_metadata:
            if not isinstance(km, dict):
                continue
            key_type = km.get('key_type')
            key = km.get('key')
            if key_type == 1:
                path.append(int(key))
            else:
                path.append(str(key))
        if path:
            paths.append(path)

    if not paths:
        return None

    top_indices = [p[0] for p in paths if isinstance(p[0], int)]
    if not top_indices:
        return None

    root = [None] * (max(top_indices) + 1)

    for path in paths:
        if not path or not isinstance(path[0], int):
            continue

        idx = path[0]
        if len(path) == 1:
            root[idx] = ocp.RestoreArgs(restore_type=np.ndarray)
            continue

        if root[idx] is None or not isinstance(root[idx], dict):
            root[idx] = {}
        node = root[idx]

        for seg in path[1:-1]:
            if isinstance(seg, int):
                # Not expected for these checkpoints; bail out to avoid shape mismatch.
                return None
            if seg not in node or not isinstance(node[seg], dict):
                node[seg] = {}
            node = node[seg]

        leaf_key = path[-1]
        if isinstance(leaf_key, int):
            return None
        node[leaf_key] = ocp.RestoreArgs(restore_type=np.ndarray)

    return root


def load(
    path: Union[str, epath.Path],
):
    """Loads checkpoint."""
    path = epath.Path(path)
    if not path.exists():
        raise ValueError(f'checkpoint path does not exist: {path.as_posix()}')

    logging.info('restoring from checkpoint %s', path.as_posix())

    restore_args = _restore_args_from_orbax_metadata(path)
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    if restore_args is None:
        # Fallback: try default restore (may require sharding info on some setups).
        target = orbax_checkpointer.restore(path)
    else:
        target = orbax_checkpointer.restore(
                path, ocp.args.PyTreeRestore(restore_args=restore_args), item=None
        )

    target[0] = running_statistics.RunningStatisticsState(**target[0])
    return target


def test():
    ckpt_path = os.environ['LEGWHEEL_CKPT_PATH']
    print(f"extract parameters from {ckpt_path}")
    target = load(ckpt_path)
    stats, policy_params, critic_params = target


    # 转换为float32 numpy数组
    state_mean = np.array(stats.mean['state'], dtype=np.float32)
    state_std = np.array(stats.std['state'], dtype=np.float32)
    
    # 保存到C++头文件，按列格式整理
    out_path = (epath.Path(__file__).parent / 'params.cpp.txt').as_posix()
    with open(out_path, 'w') as f:
        # 写入state_mean
        f.write('float state_mean[] = {\n')
        if len(state_mean.shape) == 2:
            for col in range(state_mean.shape[1]):
                row_str = ', '.join(f'{state_mean[row,col]:.6f}' for row in range(state_mean.shape[0]))
                f.write(f'    {row_str}')
                f.write(',\n' if col < state_mean.shape[1]-1 else '\n')
        else:
            f.write(f'    // {state_mean.shape[0]}\n')  
            f.write('    ' + ', '.join(f'{x:.6f}' for x in state_mean.flatten()) + '\n')
        f.write('};\n\n')
        
        # 写入state_std
        f.write('float state_std[] = {\n')
        if len(state_std.shape) == 2:
            for col in range(state_std.shape[1]):
                row_str = ', '.join(f'{state_std[row,col]:.6f}' for row in range(state_std.shape[0]))
                f.write(f'    {row_str}')
                f.write(',\n' if col < state_std.shape[1]-1 else '\n')
        else:
            f.write(f'    // {state_std.shape[0]}\n')  
            f.write('    ' + ', '.join(f'{x:.6f}' for x in state_std.flatten()) + '\n')
        f.write('};\n\n')
        
        # 写入weights
        f.write('float weights[] = {\n')
        for key, value in sorted(policy_params['params'].items()):
            print(key, value.keys(), value['kernel'].shape)
            param = np.array(value['kernel'], dtype=np.float32)
            if len(param.shape) == 2:
                print("param[:10, 0]:", param[:10, 0])
                print("param[:10, 1]:", param[:10, 1])
                f.write(f'    // {param.shape[0]}x{param.shape[1]}\n')  
                for col in range(param.shape[1]):
                    row_str = ', '.join(f'{param[row,col]:.6f}' for row in range(param.shape[0]))
                    f.write(f'    {row_str}, \n')
                f.write('\n\n')  # 层间空行
            else:
                f.write('    ' + ', '.join(f'{x:.6f}' for x in param.flatten()) + '\n\n')
        f.write('};\n\n')
        
        # 写入biases
        f.write('float biases[] = {\n')
        for key, value in sorted(policy_params['params'].items()):
            print(key, value.keys(), value['bias'].shape)
            param = np.array(value['bias'], dtype=np.float32)
            if len(param.shape) == 2:
                for col in range(param.shape[1]):
                    row_str = ', '.join(f'{param[row,col]:.6f}' for row in range(param.shape[0]))
                    f.write(f'    {row_str}, \n')
                f.write('\n\n')  # 层间空行
            else:
                f.write(f'    // {param.shape[0]}\n')  
                f.write('    ' + ', '.join(f'{x:.6f}' for x in param.flatten()) + ',\n\n')
        f.write('};\n')
    
    print(f'参数已按列格式保存到 {out_path}')


if __name__ == "__main__":
    test()