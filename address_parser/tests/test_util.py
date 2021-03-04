from unittest import TestCase

from address_parser.paf.util import csv_records_to_dicts


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
