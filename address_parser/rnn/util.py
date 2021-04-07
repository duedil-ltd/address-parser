from collections import defaultdict

import torch

from address_parser.paf import VOCAB_IDX_TO_CHAR, ADDRESS_FIELD_IDX_TO_CLASS
from address_parser.paf.util import encode_address_str


def address_components_from_pred(encoded_addresses, pred_addresses):
    """
    :param encoded_addresses: Address batch where each address is represented as a tensor the characters mapped to their
    integer index
    :param pred_addresses: Address component predictions from the model where each row is a tensor of the predicted class
    per character in the address (also encoded as the integer index)

    This method reconstructs the address components in text from the original integer encoded address, and the model
    class predictions per character.
    """
    with_preds = list(zip(encoded_addresses, pred_addresses))
    final_structured_addresses = []
    for enc_address, address_preds in with_preds:
        idxes = range(enc_address.shape[0])
        # Map the address chars as well as the class prediction for each character back to the text representation
        mapped_to_text = [(VOCAB_IDX_TO_CHAR[enc_address[i].item()],
                           ADDRESS_FIELD_IDX_TO_CLASS[address_preds[i].item()])
                          for i in idxes]
        predicted_structured_address = defaultdict(str)
        for char_class_pair in mapped_to_text:
            # Build up the address components by reconstructing the string of each component from the characters
            predicted_structured_address[char_class_pair[1]] += char_class_pair[0]
        final_structured_addresses.append(predicted_structured_address)

    return final_structured_addresses


def parse_raw_address(address, model):
    address_enc = encode_address_str(address, model.seq_length)
    pred = predict_one(address_enc, model)
    parsed_components = address_components_from_pred([address_enc], [pred])
    return parsed_components[0]


def predict_one(address_encoded, model):
    # Batch size of 1
    hidden = model.init_hidden(1)
    address_encoded_tensor = torch.from_numpy(address_encoded)
    address_encoded_tensor = address_encoded_tensor.view(1, *address_encoded_tensor.shape)
    pred, _ = model.forward(address_encoded_tensor, hidden)
    # Highest scoring class per character
    return pred.argmax(dim=1)


def predict(addresses_encoded, model):
    # Predicts on a batch in one go, much faster than using predict_one
    batch_size = len(addresses_encoded)
    hidden = model.init_hidden(batch_size)
    preds, _ = model.forward(addresses_encoded, hidden)
    # Model returns the predictions flattened so we need to reshape to maintain the original
    # batch rows, which represent the input addresses.
    preds = preds.view(batch_size, -1, model.output_dim)
    # Highest scoring class per character (first dim is the batch size)
    return preds.argmax(dim=2)


def accuracy(out, target):
    # Accuracy calculated based on the correct classification of each element in each row in a batch.
    class_preds = out.argmax(dim=1)
    return (torch.sum(class_preds == target) / len(target)).item()
