#!/usr/bin/env python3
"""Create a small evaluation subset from the full Common Voice dataset."""
import pyarrow as pa
import sys

# Parameters
INPUT_PATH = "/data/audio_eval/asr/common_voice_15_en.arrow"
OUTPUT_PATH = "kb/dockers/asr_eval/data/eval_subset.arrow"
NUM_SAMPLES = 500
OFFSET = 0

def main():
    print(f"Loading {INPUT_PATH}...")
    table = pa.ipc.open_file(INPUT_PATH).read_all()
    print(f"Total samples: {table.num_rows}")

    # Extract subset
    subset = table.slice(OFFSET, NUM_SAMPLES)
    print(f"Extracted {subset.num_rows} samples (offset={OFFSET})")

    # Write to file
    import os
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with pa.OSFile(OUTPUT_PATH, 'wb') as f:
        writer = pa.ipc.new_file(f, subset.schema)
        writer.write_table(subset)
        writer.close()

    # Get file size
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Written to: {OUTPUT_PATH}")
    print(f"File size: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
