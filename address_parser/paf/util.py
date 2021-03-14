import random
import numpy as np

from address_parser.paf import PAF_SCHEMA, AddressField, VOCAB_CHAR_TO_IDX, PADDING_CHAR, ADDRESS_FIELD_CLASSES, \
    SEPARATORS


def chunks_from_iter(it, n):
    """Yield successive n-sized chunks from iterator"""
    enumerated = enumerate(it)
    elem = next(enumerated, None)
    chunk = []
    while elem:
        if len(chunk) < n:
            chunk.append(elem[1])
        else:
            yield chunk
            chunk = [elem[1]]
        elem = next(enumerated, None)
    if chunk:
        yield chunk


def csv_records_to_dicts(records):
    dict_records = []
    for r in records:
        dict_records.append(dict([(t[0], t[1]) for t in zip(PAF_SCHEMA, r.strip().split(","))]))

    return dict_records


def shuffle_components(address):
    # TODO: Remove duplicate separators at random when a field is missing
    choice = random.choice(range(10))
    # TODO: Maybe more variations of country, also think about ways to include NI, Wales, England.
    #  Also consider getting rid of country altogether.
    country = random.choice(["uk", "u.k", "u.k.", "united kingdom", "united k", "united k."])
    sep = random.choice(SEPARATORS)
    # Different variations of address, all equally likely
    if choice == 0:
        # Full address, typical ordering
        address_parts = [
            (address[AddressField.BUILDING_NUMBER.value], AddressField.BUILDING_NUMBER.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.BUILDING_NAME.value], AddressField.BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.SUB_BUILDING_NAME.value], AddressField.SUB_BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.THOROUGHFARE_AND_DESCRIPTOR.value], AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.DEPENDENT_LOCALITY.value], AddressField.DEPENDENT_LOCALITY.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTTOWN.value], AddressField.POSTTOWN.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTCODE.value], AddressField.POSTCODE.value),
            (sep, AddressField.SEPARATOR.value),
            (country, AddressField.COUNTRY.value),
        ]
    elif choice == 1:
        # No sub-building name
        address_parts = [
            (address[AddressField.BUILDING_NUMBER.value], AddressField.BUILDING_NUMBER.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.BUILDING_NAME.value], AddressField.BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.THOROUGHFARE_AND_DESCRIPTOR.value], AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.DEPENDENT_LOCALITY.value], AddressField.DEPENDENT_LOCALITY.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTTOWN.value], AddressField.POSTTOWN.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTCODE.value], AddressField.POSTCODE.value),
            (sep, AddressField.SEPARATOR.value),
            (country, AddressField.COUNTRY.value),
        ]
    elif choice == 2:
        # No dependent locality
        address_parts = [
            (address[AddressField.BUILDING_NUMBER.value], AddressField.BUILDING_NUMBER.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.BUILDING_NAME.value], AddressField.BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.SUB_BUILDING_NAME.value], AddressField.SUB_BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.THOROUGHFARE_AND_DESCRIPTOR.value], AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTTOWN.value], AddressField.POSTTOWN.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTCODE.value], AddressField.POSTCODE.value),
            (sep, AddressField.SEPARATOR.value),
            (country, AddressField.COUNTRY.value),
        ]
    elif choice == 3:
        # No sub-building name or dependent locality
        address_parts = [
            (address[AddressField.BUILDING_NUMBER.value], AddressField.BUILDING_NUMBER.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.BUILDING_NAME.value], AddressField.BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.THOROUGHFARE_AND_DESCRIPTOR.value], AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTTOWN.value], AddressField.POSTTOWN.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTCODE.value], AddressField.POSTCODE.value),
            (sep, AddressField.SEPARATOR.value),
            (country, AddressField.COUNTRY.value),
        ]
    elif choice == 4:
        # No post town
        address_parts = [
            (address[AddressField.BUILDING_NUMBER.value], AddressField.BUILDING_NUMBER.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.BUILDING_NAME.value], AddressField.BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.SUB_BUILDING_NAME.value], AddressField.SUB_BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.THOROUGHFARE_AND_DESCRIPTOR.value], AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.DEPENDENT_LOCALITY.value], AddressField.DEPENDENT_LOCALITY.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTCODE.value], AddressField.POSTCODE.value),
            (sep, AddressField.SEPARATOR.value),
            (country, AddressField.COUNTRY.value),
        ]
    elif choice == 5:
        # No post town or dependent locality
        address_parts = [
            (address[AddressField.BUILDING_NUMBER.value], AddressField.BUILDING_NUMBER.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.BUILDING_NAME.value], AddressField.BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.SUB_BUILDING_NAME.value], AddressField.SUB_BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.THOROUGHFARE_AND_DESCRIPTOR.value], AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTCODE.value], AddressField.POSTCODE.value),
            (sep, AddressField.SEPARATOR.value),
            (country, AddressField.COUNTRY.value),
        ]
    elif choice == 6:
        # No country
        address_parts = [
            (address[AddressField.BUILDING_NUMBER.value], AddressField.BUILDING_NUMBER.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.BUILDING_NAME.value], AddressField.BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.SUB_BUILDING_NAME.value], AddressField.SUB_BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.THOROUGHFARE_AND_DESCRIPTOR.value], AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.DEPENDENT_LOCALITY.value], AddressField.DEPENDENT_LOCALITY.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTTOWN.value], AddressField.POSTTOWN.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTCODE.value], AddressField.POSTCODE.value),
        ]
    elif choice == 7:
        # No country or sub-building name
        address_parts = [
            (address[AddressField.BUILDING_NUMBER.value], AddressField.BUILDING_NUMBER.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.BUILDING_NAME.value], AddressField.BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.THOROUGHFARE_AND_DESCRIPTOR.value], AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.DEPENDENT_LOCALITY.value], AddressField.DEPENDENT_LOCALITY.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTTOWN.value], AddressField.POSTTOWN.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTCODE.value], AddressField.POSTCODE.value),
        ]
    elif choice == 8:
        # No country or sub-building name or dependent locality
        address_parts = [
            (address[AddressField.BUILDING_NUMBER.value], AddressField.BUILDING_NUMBER.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.BUILDING_NAME.value], AddressField.BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.THOROUGHFARE_AND_DESCRIPTOR.value], AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTTOWN.value], AddressField.POSTTOWN.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTCODE.value], AddressField.POSTCODE.value),
        ]
    else:
        # No country or post town
        address_parts = [
            (address[AddressField.BUILDING_NUMBER.value], AddressField.BUILDING_NUMBER.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.BUILDING_NAME.value], AddressField.BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.SUB_BUILDING_NAME.value], AddressField.SUB_BUILDING_NAME.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.THOROUGHFARE_AND_DESCRIPTOR.value], AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.DEPENDENT_LOCALITY.value], AddressField.DEPENDENT_LOCALITY.value),
            (sep, AddressField.SEPARATOR.value),
            (address[AddressField.POSTCODE.value], AddressField.POSTCODE.value),
        ]

    return address_parts


def split_component_chars(address_parts):
    """
    address_parts: list of the form [(<address_part_1>, <address_part_1_label>), .... ]

    returns [(<char_0>, <address_comp_for_char_0), (<char_1>, <address_comp_for_char_1),.., (<char_n-1>, <address_comp_for_char_n-1)]
    """
    char_arr = []
    for address_part, address_part_label in address_parts:
        # The address part of the tuple (address_part, address_part_label)
        for c in address_part:
            char_arr.append((c, address_part_label))

    return char_arr


def _encode_address(address_char_parts, seq_length):
    """
    address_parts: list of the form [(<char_1>, <char_1_label>), .... ]

    return numpy array of length seq_length with the mapped integer of each character, filling the rest of the array
    with the dedicated padding index.
    """
    address = "".join(t[0] for t in address_char_parts)
    # Truncate past seq_length
    address = address[:seq_length]
    # Initialise array with all padding chars (encoded as integer index)
    address_arr = [VOCAB_CHAR_TO_IDX[PADDING_CHAR]] * seq_length
    char_idxes = [VOCAB_CHAR_TO_IDX[c] for c in address]
    i = 0
    # Replace padding char with address char for the length of the address.
    for c_idx in char_idxes:
        address_arr[i] = c_idx
        i += 1

    return np.array(address_arr)


def _encode_labels(address_char_parts, seq_length):
    """
    address_parts: list of the form [(<char_1>, <char_1_label>), .... ]

    returns a numpy array where each character is encoded as the class to which it belongs (inc padding)
    """
    labels_arr = [ADDRESS_FIELD_CLASSES[AddressField.PADDING.value]] * seq_length
    # Truncate past seq_length
    address_char_parts = address_char_parts[:seq_length]
    i = 0
    for _, address_class in address_char_parts:
        labels_arr[i] = ADDRESS_FIELD_CLASSES[address_class]
        i += 1

    return np.array(labels_arr)


def encode_address_and_labels(address_parts, seq_length=128):
    """
    Receives an address of the form of [(<address_part_1>, <address_part_1_label>), .... ]
    and converts that to a tuple of Numpy arrays x, y
    where
    x: a sequence of characters (mapped to their index) of fixed length given by seq_length
    y: Represented as a single vector of the corresponding
        encoded classes for each character

        e.g. ("25", "building_number"), (" ", "separator") ("christopher st", "thoroughfare_and_descriptor")
        should yield
        x = [[2, 5, 68, 12, 17, 18, 28, 29, 24, 25, 17, 14, 27, 68, 28, 29, 74, 74, 74, ...]]  shape = (1, seq_length)
        y = [[0, 0, 8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 9, 9, ....]]  shape = (1, seq_length)
    """
    x = _encode_address(address_parts, seq_length)
    y = _encode_labels(address_parts, seq_length)

    return x, y
