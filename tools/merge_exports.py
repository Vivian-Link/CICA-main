"""Merge complete-view exports while checking duplicate record identities."""
import argparse
import csv
import glob
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    rows, seen, statuses = [], set(), set()
    files = []
    for pattern in a.input:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(pattern)
        files.extend(matches)
    for name in files:
        with open(name, encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                key = tuple(row[x] for x in ['dataset', 'split', 'seed', 'model', 'utterance_id'])
                if key in seen:
                    raise ValueError('Duplicate prediction: ' + str(key))
                if any(row.get('observed_' + m, '1') != '1' for m in ['L', 'V', 'A']):
                    raise ValueError('Missingness exports cannot enter complete-input paired audits')
                seen.add(key); statuses.add(row['data_status']); rows.append(row)
    if len(statuses) != 1:
        raise ValueError('Do not mix measured and software-test records')
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Destination already exists')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    main()
