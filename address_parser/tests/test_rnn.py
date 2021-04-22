import numpy as np
import torch
from unittest import TestCase

from address_parser.rnn import AddressRNN


class TestRnn(TestCase):
    def setUp(self) -> None:
        self.model = AddressRNN(
            vocab=[0, 1, 2, 3],
            lstm_dim=8,
            lstm_layers=2,
            output_dim=10,
            embedding_dim=3,
            seq_length=20,
            train_on_gpu=False,
            batch_first=True
        )
        # Batch of 3 x 4
        batch = [[1, 0, 1, 1], [3, 2, 0, 1], [2, 2, 2, 0]]
        self.x = torch.from_numpy(np.array(batch))
        self.hidden = self.model.init_hidden(self.x.size()[0])

    def test_embed_out_dim(self):
        embed_out = self.model._embed(self.x)
        # Embedding layer should produce an embedding of size 3 (specified by embedding_dim) for each element in batch
        self.assertEqual(embed_out.size(), torch.Size([3, 4, 3]))

    def test_lstm_out_dim(self):
        embed_out = self.model._embed(self.x)
        lstm_out, _ = self.model._lstm(embed_out, self.hidden)
        # LSTM layer should produce lstm_dim * 2 outputs per element in the batch (* 2 because it's bi-directional)
        self.assertEqual(lstm_out.size(), torch.Size([3, 4, 16]))

    def test_forward_out_dim(self):
        out, _ = self.model.forward(self.x, self.hidden)
        # Full forward pass through the network should produce a score per class (output_dim = 10)
        # per element in the flattened batch (3 * 4 = 12)
        self.assertEqual(out.size(), torch.Size([12, 10]))
