import string
from enum import Enum

PAF_SCHEMA = [
    "postcode",
    "posttown",
    "dependent_locality",
    "double_dependent_locality",
    "thoroughfare_and_descriptor",
    "dependent_thoroughfare_and_descriptor",
    "building_number",
    "building_name",
    "sub_building_name",
    "po_box",
    "department_name",
    "organisation_name",
    "udprn",
    "postcode_type",
    "su_organisation_indicator",
    "delivery_point_suffix",
    "address_key",
    "organisation_key",
    "number_of_households",
    "locality_key"
]


class AddressField(Enum):
    BUILDING_NUMBER = 'building_number'
    BUILDING_NAME = 'building_name'
    SUB_BUILDING_NAME = 'sub_building_name'
    THOROUGHFARE_AND_DESCRIPTOR = 'thoroughfare_and_descriptor'
    DEPENDENT_LOCALITY = 'dependent_locality'
    POSTTOWN = 'posttown'
    POSTCODE = 'postcode'
    COUNTRY = 'country'
    SEPARATOR = 'separator'
    PADDING = 'padding'


ADDRESS_FIELD_CLASSES = {
    AddressField.BUILDING_NUMBER.value: 0,
    AddressField.BUILDING_NAME.value: 1,
    AddressField.SUB_BUILDING_NAME.value: 2,
    AddressField.THOROUGHFARE_AND_DESCRIPTOR.value: 3,
    AddressField.DEPENDENT_LOCALITY.value: 4,
    AddressField.POSTTOWN.value: 5,
    AddressField.POSTCODE.value: 6,
    AddressField.COUNTRY.value: 7,
    AddressField.SEPARATOR.value: 8,
    AddressField.PADDING.value: 9
}


# Need the reverse mapping for re-constructing address components from model outputs.
ADDRESS_FIELD_IDX_TO_CLASS = dict((t[1], t[0]) for t in ADDRESS_FIELD_CLASSES.items())

SEPARATORS = [",", " ", " ", ", "]
PADDING_CHAR = "|"
# Copied from https://github.com/jasonrig/address-net/blob/master/addressnet/dataset.py#L70
VOCAB = list(string.digits + string.ascii_lowercase + string.punctuation + string.whitespace) + [PADDING_CHAR]
VOCAB_IDX_TO_CHAR = dict(enumerate(VOCAB))
VOCAB_CHAR_TO_IDX = dict((t[1], t[0]) for t in VOCAB_IDX_TO_CHAR.items())
