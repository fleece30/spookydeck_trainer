import csv
import sys
import os
from os.path import isfile, join
from constants import fields, file_path


def read_csv():
    rows = []
    with open("rephrased_overviews_movies_unique.csv", mode='r') as file:
        csv_file = csv.DictReader(file)
        for line in csv_file:
            if len(line['overview']) > 15 and line['overview'] != 'No overview found.':
                rows.append(line)
    return rows


def merge_csv():
    files = [f for f in os.listdir(file_path) if isfile(join('./outputs/', f))]
    with open(f'rephrased_overviews_movies_with_cast.csv', 'a') as write_file:
        writer = csv.DictWriter(write_file, fieldnames=fields)
        writer.writeheader()
        for file in files:
            with open(f'{file_path}/{file}', mode='r') as part_file:
                csv_file = csv.DictReader(part_file)
                writer.writerows(csv_file)


def remove_duplicates():
    with open('rephrased_overviews_movies_with_cast.csv', 'r') as input_file, open(
            'rephrased_overviews_movies_with_cast_unique.csv', 'w') as output_file:
        seen = set()
        csv_file = csv.DictReader(input_file)
        csv_output = csv.DictWriter(output_file, fieldnames=fields)
        for line in csv_file:
            movie_id = line['id']
            if movie_id in seen:
                print(movie_id)
                continue
            seen.add(movie_id)
            csv_output.writerow(line)
