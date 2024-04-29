import csv


def import_csv(filename, directory: str = "./data_source/", format: str = "csv"):
    filepath = f"{directory}/{filename}.{format}"
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        return list(reader)
