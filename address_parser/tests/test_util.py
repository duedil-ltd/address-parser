import numpy as np
from unittest import TestCase

from address_parser.paf import AddressField
from address_parser.paf.util import csv_records_to_dicts, split_component_chars, encode_address_and_labels


class TestUtils(TestCase):
    def test_csv_records_to_dicts(self):
        csv_records = [
            "TN22 3BD,UCKFIELD,Fairwarp,,Nursery Lane,, ,The Old Vicarage,,,,,24708670,S, ,1W,21108384,0,1,9983",
        ]
        dicts = csv_records_to_dicts(csv_records)
        self.assertEqual(
            dicts, [{
                "postcode": "TN22 3BD",
                "posttown": "UCKFIELD",
                "dependent_locality": "Fairwarp",
                "double_dependent_locality": "",
                "thoroughfare_and_descriptor": "Nursery Lane",
                "dependent_thoroughfare_and_descriptor": "",
                "building_number": " ",
                "building_name": "The Old Vicarage",
                "sub_building_name": "",
                "po_box": "",
                "department_name": "",
                "organisation_name": "",
                "udprn": "24708670",
                "postcode_type": "S",
                "su_organisation_indicator": " ",
                "delivery_point_suffix": "1W",
                "address_key": "21108384",
                "organisation_key": "0",
                "number_of_households": "1",
                "locality_key": "9983"
            }]
        )

    def test_split_component_chars(self):
        address_parts = [("25", AddressField.BUILDING_NUMBER.value),
                         (" ", AddressField.SEPARATOR.value),
                         ("christopher st", AddressField.THOROUGHFARE_AND_DESCRIPTOR.value)]

        address_char_components = split_component_chars(address_parts)
        self.assertEqual(address_char_components, [
            ('2', AddressField.BUILDING_NUMBER.value),
            ('5', AddressField.BUILDING_NUMBER.value),
            (' ', AddressField.SEPARATOR.value),
            ('c', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('h', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('r', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('i', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('s', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('t', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('o', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('p', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('h', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('e', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('r', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            (' ', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('s', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('t', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value)
        ])

    def test_encode_address_and_labels(self):
        address_char_components = [
            ('2', AddressField.BUILDING_NUMBER.value),
            ('5', AddressField.BUILDING_NUMBER.value),
            (' ', AddressField.SEPARATOR.value),
            ('c', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('h', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('r', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('i', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('s', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('t', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('o', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('p', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('h', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('e', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('r', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            (' ', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('s', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value),
            ('t', AddressField.THOROUGHFARE_AND_DESCRIPTOR.value)
        ]
        seq_length = 20

        x, y = encode_address_and_labels(address_char_components, seq_length=seq_length)
        self.assertTrue(
            np.array_equal(x,
                           np.array([2, 5, 68, 12, 17, 27, 18, 28, 29, 24, 25, 17, 14, 27, 68, 28, 29, 74, 74, 74]))
        )
        self.assertTrue(
            np.array_equal(y,
                           np.array([0, 0, 8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 9, 9]))
        )
