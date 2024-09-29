import csv
import time

import requests
from rephrase_overviews.constants import fields


def fetch_movie_data_from_tmdb_api(imdb_id):
    url = f"https://api.themoviedb.org/3/find/{imdb_id}?external_source=imdb_id"

    response = requests.get(url, headers={
        "Authorization": "Bearer "})

    if len(response.json()['movie_results']) == 0:
        return None

    tmdb_id = response.json()['movie_results'][0]['id']

    tmdb_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    response = requests.get(tmdb_url, headers={
        "Authorization": "Bearer "})

    return response.json()


def populate_new_movies_csv(imdb_ids):
    with open('new_movies.csv', 'w') as csvfile:
        csv_output = csv.DictWriter(csvfile, fieldnames=fields)
        csv_output.writeheader()
        for index, imdb_id in enumerate(imdb_ids):
            print(f"processed {index}/{len(imdb_ids)} ids...", end="\r")
            data = fetch_movie_data_from_tmdb_api(imdb_id)
            if data:
                filtered_data = {key: data.get(key, '') for key in fields if key}
                genres = ""
                for genre in map(lambda x: x['name'], data['genres']):
                    genres = genres + genre + ", "
                filtered_data['genre_names'] = genres[:-2]
                if data['belongs_to_collection'] is not None:
                    filtered_data['collection'] = data['belongs_to_collection']['id']
                    filtered_data['collection_name'] = data['belongs_to_collection']['name']

                csv_output.writerow(filtered_data)
            time.sleep(2)
