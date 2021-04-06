import argparse

import numpy as np
import torch

from address_parser.paf.preprocess import preprocess_addresses
from address_parser.paf.util import chunks_from_iter, csv_records_to_dicts
from address_parser.rnn.util import predict, address_components_from_pred

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
            records = csv_records_to_dicts(chunk)[:100]
            # TODO:
            #  - The preprocessing method used here is the same as that used for training, which does some shuffling of components and extra
            #  bits and pieces to make training harder. This needs to be replpaced with a method that just works with an address string and
            #  just maps character in an arbitrary address string to the char index encoded version.
            preprocessed = list(preprocess_addresses(records, model.seq_length))
            addresses_encoded = torch.from_numpy(np.array([t[0] for t in preprocessed]))
            preds = predict(addresses_encoded, model)
            results.extend(address_components_from_pred(addresses_encoded, preds))
            chunks += 1
            if chunks % 10 == 0:
                print(f"Processed {chunks} chunks ({len(results)} records)")
            # Just for comparing results at the end
            all_records.extend(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-path', required=True, help="Path to test address CSV file")
    parser.add_argument('--model-path', required=True, help="Path to trained model")
    args = parser.parse_args()
    main(args.test_path, args.model_path)
