import csv


def read_csv():
    rows = []
    with open("horror_movies.csv", mode='r') as file:
        csv_file = csv.DictReader(file)
        for line in csv_file:
            if len(line['overview']) > 15 and line['overview'] != 'No overview found.':
                rows.append(line)
    return rows
