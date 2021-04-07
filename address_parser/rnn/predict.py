import argparse
import torch

from address_parser.paf.util import chunks_from_iter
from address_parser.rnn.util import parse_raw_address

CHUNK_SIZE = 1000


def _load_model(model_path):
    print(f"Loading trained model from {model_path}")
    if torch.cuda.is_available():
        print("GPU available")
        map_location = lambda loc: loc[0].cuda()
    else:
        print("GPU unavailable, loading model on CPU")
        map_location = 'cpu'
    model = torch.load(model_path, map_location=map_location)
    return model


def main(test_file, model_path):
    model = _load_model(model_path)
    print(f"Running model on address file in batches of size {CHUNK_SIZE}")
    results = []
    chunks = 0
    all_records = []
    with open(test_file, "r") as f:
        for chunk in chunks_from_iter(f.readlines(), CHUNK_SIZE):
            chunk_res = [parse_raw_address(add.strip(), model) for add in chunk]
            results.extend(chunk_res)
            chunks += 1
            if chunks % 10 == 0:
                print(f"Processed {chunks} chunks ({len(results)} records)")
            # Just for comparing results at the end
            all_records.extend(chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # This file should have an address per line
    parser.add_argument('--test-path', required=True, help="Path to test address file")
    parser.add_argument('--model-path', required=True, help="Path to trained model")
    args = parser.parse_args()
    main(args.test_path, args.model_path)
