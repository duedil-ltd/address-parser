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
