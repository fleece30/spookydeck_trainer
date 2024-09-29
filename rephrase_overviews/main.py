import csv


def read_csv():
    rows = []
    with open("/home/fleece/Desktop/spookydeck_trainer/new_movies.csv", mode='r') as file:
        csv_file = csv.DictReader(file)
        for line in csv_file:
            if line['overview'] is not None and len(line['overview']) > 15 and line['overview'] != 'No overview found.':
                rows.append(line)
    return rows
