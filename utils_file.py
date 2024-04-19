import csv


def import_csv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        return list(reader)
