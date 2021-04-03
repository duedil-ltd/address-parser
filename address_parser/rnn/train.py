import argparse

import numpy as np
import torch

from torch import optim, nn
from address_parser.paf import ADDRESS_FIELD_CLASSES, VOCAB_IDX_TO_CHAR
from address_parser.paf.preprocess import preprocess_addresses
from address_parser.paf.util import chunks_from_iter, csv_records_to_dicts
from address_parser.rnn import AddressRNN

CHUNK_SIZE = 1000
# BATCH_SIZE = 128
BATCH_SIZE = 1000
LSTM_DIM = 128
LSTM_LAYERS = 2
OUTPUT_DIM = len(ADDRESS_FIELD_CLASSES)
SEQ_LENGTH = 100
LR = 0.001
VOCAB = VOCAB_IDX_TO_CHAR.keys()
CLIP = 5
EPOCHS = 1


def train(records):
    train_on_gpu = torch.cuda.is_available()
    model = AddressRNN(vocab=VOCAB, lstm_dim=LSTM_DIM, lstm_layers=LSTM_LAYERS,
                       output_dim=OUTPUT_DIM, seq_length=SEQ_LENGTH, train_on_gpu=train_on_gpu,
                       batch_first=True)

    print("Model architecture:")
    print(model)

    hidden = model.init_hidden(BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    if train_on_gpu:
        model.cuda()

    model.train()
    print(f"Starting training")
    for e in range(EPOCHS):
        batches = 0
        loss = None
        for batch in chunks_from_iter(preprocess_addresses(records, seq_length=SEQ_LENGTH), BATCH_SIZE):
            X = torch.from_numpy(np.array([t[0] for t in batch]))
            y = torch.from_numpy(np.array([t[1] for t in batch]))
            # Getting rid of incomplete batches
            # TODO: Do this in pre-processing step
            if X.shape[0] < BATCH_SIZE:
                continue
            if train_on_gpu:
                X = X.cuda()
                y = y.cuda()

            hidden = tuple([h.data for h in hidden])
            model.zero_grad()

            out, hidden = model(X, hidden)
            """
            Align y for computing CrossEntropyLoss. We know that the shape of `out` is
                (BATCH_SIZE * SEQ_LENGTH, OUTPUT_DIM), i.e. a logit score per output class per character in a batch
            so we shape `y` to be
                a vector tensor of dim BATCH_SIZE * SEQ_LENGTH -> True class label per character in the batch.

            This is similar to the below example
            >>> target = torch.empty(3, dtype=torch.long).random_(5)
            >>> target
            tensor([2, 4, 4])
            >>> output = torch.Tensor([[0, 0 , 20, 0, 0], [0, 0, 0, 0, 14], [0, 0, 0, 0, 25]])
            >>> output
            tensor([[ 0.,  0., 20.,  0.,  0.],
                    [ 0.,  0.,  0.,  0., 14.],
                    [ 0.,  0.,  0.,  0., 25.]])
            >>> loss = nn.CrossEntropyLoss()
            >>> loss(output, target).item()
            1.1126181789222755e-06

            where target is a vector tensor of dim 3 and the values being some class label between 0 and 4
            and output is a tensor of dim (3, 5) where each element of row i is the scores for each of the 5 classes.
            The small loss value shows that the cross entropy loss is behaving as expected for the example provided,
            since the output has high scores for the correct class labels at the corresponding indexes and zero
            elsewhere.
            """
            loss = criterion(out, y.reshape(BATCH_SIZE * SEQ_LENGTH))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()

            batches += 1
            if batches % 10 == 0:
                print(f"Finished training on {batches} batches in epoch {e}")
                print(f"Loss so far is {loss.item()}")
        print(f"Finished training for epoch {e}")
        print(f"Loss at end of epoch {e} is {loss.item()}")

    return model


def main(paf_sample_file, model_output_path):
    """
    We have a structured address dataset that we can use to automatically construct a training set
    for address parsing.
    """
    records = []
    print("Loading address data")
    with open(paf_sample_file, "r") as f:
        for chunk in chunks_from_iter(f.readlines(), CHUNK_SIZE):
            records += csv_records_to_dicts(chunk)
    print(f"Starting training with {len(records)} address records")
    model = train(records)
    print("Training complete")
    print("Saving model..")
    torch.save(model, model_output_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paf-sample-path', required=True, help="Path to sample PAF addresses CSV file")
    parser.add_argument('--model-output-path', required=True, help="Output path to save model object")
    args = parser.parse_args()
    main(args.paf_sample_path, args.model_output_path)
