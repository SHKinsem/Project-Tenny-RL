"""Persist an Edge MCP PNG ImageContent staged as base64; never modify pixels."""

import base64
import sys
from pathlib import Path


def save(encoded_path: Path, output_path: Path) -> None:
    encoded_path = encoded_path.resolve(strict=True)
    output_path = output_path.resolve()
    if encoded_path.suffix != '.b64' or output_path.suffix.lower() != '.png':
        raise ValueError('Expected .b64 input and .png output')
    if encoded_path.parent != output_path.parent:
        raise ValueError('Keep staged bytes beside the output image')
    raw = base64.b64decode(''.join(encoded_path.read_text(encoding='utf-8').split()), validate=True)
    if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('ImageContent is not a PNG')
    with output_path.open('xb') as target:
        target.write(raw)
    encoded_path.unlink()
    print(output_path)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: save_screenshot.py input.b64 output.png')
    save(Path(sys.argv[1]), Path(sys.argv[2]))
