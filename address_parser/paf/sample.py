import argparse
import random

import pandas as pd

from address_parser.paf import chunks_from_iter, csv_records_to_dicts

CHUNK_SIZE = 10000


def _get_sample(file_path, sample_proportion=0.01):
    records = []
    chunks_processed = 0
    with open(file_path, "r", encoding='windows-1252') as f:
        for chunk in chunks_from_iter(f.readlines(), CHUNK_SIZE):
            records.extend(random.sample(chunk, int(len(chunk) * sample_proportion)))
            chunks_processed += 1
            if chunks_processed % 100 == 0:
                print(f"Processed {chunks_processed} chunks")

    return csv_records_to_dicts(records)


def main(paf_input_path, output_path):
    # Get a sample of the PAF records
    print(f"Getting sample of PAF addresses.")
    records = _get_sample(paf_input_path)
    # Shuffle them to avoid any sensitivity around ordering, as the PAF data
    # is sorted alphabetically.
    random.shuffle(records)
    # Write output to CSV to be pre-processed.
    print(f"Writing sample of size {len(records)} to {output_path}")
    pd.DataFrame(records).to_csv(output_path, index=False)


if __name__ == "__main__":
    # PAF file contains almost 31M addresses so we want to sample about 1%,
    # which should provide enough diversity to train an RNN.
    parser = argparse.ArgumentParser()
    parser.add_argument('--paf-input-path', required=True, help="Path to PAF CSV file")
    parser.add_argument('--sample-output-path', required=True, help="Output path to save PAF sample CSV file")
    args = parser.parse_args()
    main(args.paf_input_path, args.sample_output_path)
