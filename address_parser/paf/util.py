from address_parser.paf import PAF_SCHEMA


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
