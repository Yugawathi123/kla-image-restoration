from pathlib import Path
import numpy as np
from PIL import Image

input_dir = Path(r"C:\Users\yuga\Downloads\restored_test_output_v4")
output_dir = Path(r"C:\Users\yuga\Downloads\restored_test_v4_png")

output_dir.mkdir(parents=True, exist_ok=True)

npy_files = sorted(input_dir.glob("*.npy"))

print(f"Found {len(npy_files)} .npy files")

for npy_file in npy_files:
    arr = np.load(npy_file)

    # Remove unnecessary dimensions if present
    arr = np.squeeze(arr)

    # Convert to float32
    arr = arr.astype(np.float32)

    # Display/output range: [0, 1] -> [0, 255]
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)

    output_file = output_dir / (npy_file.stem + ".png")
    Image.fromarray(arr).save(output_file)

print(f"Done!")
print(f"PNG files saved to: {output_dir}")