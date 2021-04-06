from torch import nn, zeros


class AddressRNN(nn.Module):
    def __init__(self, vocab, lstm_dim, lstm_layers, output_dim,
                 seq_length, embedding_dim=10, drop_prob=0.3, train_on_gpu=False,
                 batch_first=False):
        super().__init__()

        self._lstm_dim = lstm_dim
        self._lstm_layers = lstm_layers
        self._train_on_gpu = train_on_gpu
        self.output_dim = output_dim
        self.seq_length = seq_length

        self._embed = nn.Embedding(len(vocab), embedding_dim)
        self._lstm = nn.LSTM(input_size=embedding_dim,
                             hidden_size=lstm_dim,
                             num_layers=lstm_layers,
                             dropout=drop_prob,
                             bidirectional=True,
                             batch_first=batch_first)

        self._dropout = nn.Dropout(drop_prob)
        # * 2 here because it's a bidirectional LSTM
        self._fc = nn.Linear(lstm_dim * 2, output_dim)

    def forward(self, x, hidden):
        embed_out = self._embed(x)
        lstm_out, hidden = self._lstm(embed_out, hidden)
        # We want to flatten all the batches and sequences within the batch.
        # Thus, this operation will produce a tensor of dim (batch_size * seq_length, self._lstm_dim * 2)
        lstm_out = lstm_out.contiguous().view(-1, self._lstm_dim * 2)
        lstm_out = self._dropout(lstm_out)
        # Producing a final output of (batch_size * seq_length, self._output_dim), i.e. a logit score for each
        # address class per each character in the batch.
        out = self._fc(lstm_out)

        return out, hidden

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes lstm_layers x batch_size x lstm_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        # We multiply by 2 here because we're using a bidirectional LSTM
        if self._train_on_gpu:
            hidden = (
                zeros((self._lstm_layers * 2, batch_size, self._lstm_dim)).cuda(),
                zeros((self._lstm_layers * 2, batch_size, self._lstm_dim)).cuda()
            )
        else:
            hidden = (
                zeros((self._lstm_layers * 2, batch_size, self._lstm_dim)),
                zeros((self._lstm_layers * 2, batch_size, self._lstm_dim))
            )

        return hidden
