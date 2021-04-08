import argparse
import pickle

from address_parser.paf import ADDRESS_FIELD_CLASSES
from address_parser.paf.util import (
    chunks_from_iter, csv_records_to_dicts, shuffle_components, split_component_chars, encode_address_and_labels
)

CHUNK_SIZE = 1000


def _address_char_level_labels(address, seq_length):
    """
    Initial address prep.

    Steps
    1- Construct an address string from the input, getting rid of things like organisation name, udprn etc, as well as
        adding country at the end (need to think about whether it should be just UK, or break down by regions).
    2- Standardise casing to lower case.
    3- Apply some shuffling of the components with some probability so that the dataset is more realistic.
        However, make sure the shuffling is realistic and in line with how a human would write an address. For example,
        it's uncommon for people to write an address in the form
        <postcode>, <country>, <building_name>, <flat_number>, <street_name>
        Typical variations:
          <building_number>, <thoroughfare_and_descriptor>, <building_name>, <postcode>, <country>
          <building_number>, <building_name>, <thoroughfare_and_descriptor>, <postcode>, <country>
          <building_number>, <building_name>, <thoroughfare_and_descriptor>, <posttown>, <postcode>, <country>
          <building_number>, <building_name>, <thoroughfare_and_descriptor>, <dependent_locality>, <posttown>, <postcode>, <country>
          <building_number>, <building_name>, <thoroughfare_and_descriptor>, <dependent_locality>, <postcode>, <country>

        all of these without the country at the end etc.
    4- Convert each address to a list of tuples of char, address component of the form
          [(<char_0>, <address_comp_for_char_0), (<char_1>, <address_comp_for_char_1),.., (<char_n-1>, <address_comp_for_char_n-1)]

        so for example, if the address is 25 (building number), Christopher st (street_name), EC2A 2BS (postcode)
        we should return
        [('2', 'building_number'),
        ('5', 'building_number'),
        ('{some_separator}', 'separator'),
        ('c', 'street_name'),
        ('h', 'street_name'),
        ('r', 'street_name'),
        ('i', 'street_name'),
        ('s', 'street_name'),
        ('t', 'street_name'),
        ('o', 'street_name'),
        ('p', 'street_name'),
        ('h', 'street_name'),
        ('e', 'street_name'),
        ('r', 'street_name'),
        (' ', 'street_name'),
        ('s', 'street_name'),
        ('t', 'street_name'),
        ('{some_separator}', 'separator'),
        ('e', 'postcode'),
        ('c', 'postcode'),
        ('2', 'postcode'),
        ('a', 'postcode'),
        (' ', 'postcode'),
        ('2', 'postcode'),
        ('b', 'postcode'),
        ('s', 'postcode')
    5 - Convert the char level tuple representation into features and labels numpy arrays
    """
    # Get rid of unneeded address fields and make all fields lower case
    address_dict = dict((t[0], (t[1] or '').lower()) for t in address.items() if t[0] in ADDRESS_FIELD_CLASSES)
    address_parts = shuffle_components(address_dict)
    address_char_components = split_component_chars(address_parts)
    # TODO: Add extra step before the encoding to:
    #     - Introduce typos randomly with some probability
    #     - Introduce some variations of common address words like street, road, avenue, place etc
    address_encoded, address_labels = encode_address_and_labels(address_char_components, seq_length)
    return address_encoded, address_labels


def preprocess_addresses(address_dicts, seq_length):
    for address in address_dicts:
        address_encoded, address_labels = _address_char_level_labels(address, seq_length)
        yield address_encoded, address_labels


def main(paf_sample_file, output_file):
    """
    We have a structured address dataset that we can use to automatically construct a training set
    for address parsing.
    """
    preprocessed = []
    chunks = 0
    with open(paf_sample_file, "r") as f:
        print(f"Processing data in chunks of size {CHUNK_SIZE}")
        for chunk in chunks_from_iter(f.readlines(), CHUNK_SIZE):
            address_dicts = csv_records_to_dicts(chunk)
            preprocessed += list(preprocess_addresses(address_dicts, seq_length=100))
            chunks += 1
            if chunks % 100 == 0:
                print(f"Processed {chunks} chunks")

    print(f"Writing output to {output_file}")
    with open(output_file, "wb") as f_out:
        output = {"address_features": [t[0] for t in preprocessed],
                  "address_labels": [t[1] for t in preprocessed]}
        pickle.dump(output, f_out)


if __name__ == "__main__":
    # PAF file contains almost 31M addresses so we want to sample about 1%,
    # which should provide enough diversity to train an RNN.
    parser = argparse.ArgumentParser()
    parser.add_argument('--paf-sample-path', required=True, help="Path to sample PAF addresses CSV file")
    parser.add_argument('--preprocessed-output-path', required=True, help='Path to pickle output file for pre-processed data')
    args = parser.parse_args()
    main(args.paf_sample_path, args.preprocessed_output_path)
