import base64
import subprocess
import sys
import tempfile
from pathlib import Path

script = Path(__file__).parent / 'taobao-product-archive/scripts/save_screenshot.py'
fixture = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jWXkAAAAASUVORK5CYII=')

with tempfile.TemporaryDirectory(prefix='taobao-image-test-') as temp:
    root = Path(temp)
    encoded = root / 'image.b64'
    image = root / 'image.png'
    encoded.write_text(base64.b64encode(fixture).decode(), encoding='utf-8')
    result = subprocess.run([sys.executable, str(script), str(encoded), str(image)], capture_output=True)
    assert result.returncode == 0, result.stderr
    assert image.read_bytes() == fixture and not encoded.exists()

    encoded.write_text(base64.b64encode(fixture).decode(), encoding='utf-8')
    result = subprocess.run([sys.executable, str(script), str(encoded), str(image)], capture_output=True)
    assert result.returncode != 0 and image.read_bytes() == fixture and encoded.exists()

    invalid = root / 'invalid.png'
    encoded.write_text(base64.b64encode(b'not an image').decode(), encoding='utf-8')
    result = subprocess.run([sys.executable, str(script), str(encoded), str(invalid)], capture_output=True)
    assert result.returncode != 0 and not invalid.exists() and encoded.exists()

print('PASS: PNG bytes preserved; existing image protected; invalid data rejected')
