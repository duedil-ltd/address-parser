import argparse

from address_parser.paf import ADDRESS_FIELD_CLASSES
from address_parser.paf.util import (
    chunks_from_iter, csv_records_to_dicts, shuffle_components, split_component_chars, encode_address_and_labels
)

CHUNK_SIZE = 1000


def _address_char_level_labels(address):
    """
    Initial address prep.

    Steps
    1- Construct an address string from the input, getting rid of things like organisation name, udprn etc, as well as
        adding country at the end (need to think about whether it should be just UK, or break down by regions).
    2- Standardise casing to lower case and remove punctuation.
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
    address_encoded, address_labels = encode_address_and_labels(address_char_components)


def main(paf_sample_file):
    """
    We have a structured address dataset that we can use to automatically construct a training set
    for address parsing.
    """
    with open(paf_sample_file, "r") as f:
        for chunk in chunks_from_iter(f.readlines(), CHUNK_SIZE):
            address_dicts = csv_records_to_dicts(chunk)
            for address in address_dicts:
                address_char_labels = _address_char_level_labels(address)


if __name__ == "__main__":
    # PAF file contains almost 31M addresses so we want to sample about 1%,
    # which should provide enough diversity to train an RNN.
    parser = argparse.ArgumentParser()
    parser.add_argument('--paf-sample-path', required=True, help="Path to sample PAF addresses CSV file")
    args = parser.parse_args()
    main(args.paf_input_path)
