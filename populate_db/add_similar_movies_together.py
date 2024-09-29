import csv

from neo4j import GraphDatabase

uri = "bolt://localhost:7688"
username = "neo4j"
password = "Tz12ep34f3012@"


class DBPopulatorForSimilarMovies:
    def __init__(self):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def create_indexes(self):
        with self.driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Movie) ON (m.id)")

    def insert_data(self, row, similar_row):
        with self.driver.session() as session:
            try:
                row_dict = row.to_dict()
                similar_row_dict = similar_row.to_dict()
                session.write_transaction(self.create_movie, row_dict)
                session.write_transaction(self.create_movie, similar_row_dict)
                session.write_transaction(self.create_similarity_relationship, row_dict['id'], similar_row_dict['id'])
            except Exception as e:
                print(f"Error: {e} in row {row}")

    def get_nodes_not_in_db(self):
        skipped_movies = []
        count = 0
        with self.driver.session() as session:
            try:
                with open("final_movie_data.csv", 'r') as csv_file:
                    reader = csv.DictReader(csv_file)
                    rows = list(reader)
                    for row in rows:
                        nodes = session.run("match (n) where n.id = $id return n", {"id": int(row['id'])})
                        results = [record for record in nodes.data()]
                        if len(results) == 0:
                            skipped_movies.append(row['id'])
                        count += 1
                        print(f"processed {count} movies", end="\r")
            except Exception as e:
                print(f"Error: {e} in row {row}")
        print(f"Skipped movies: {skipped_movies}")
        print(f"Total skipped movies: {len(skipped_movies)}")

    @staticmethod
    def create_similarity_relationship(tx, movie1_id, movie2_id):
        similarity_query = """
                MERGE (movie1:Movie {id: $movie1_id})
                MERGE (movie2:Movie {id: $movie2_id})
                MERGE (movie1)-[:SIMILAR_TO]->(movie2)
                MERGE (movie2)-[:SIMILAR_TO]->(movie1)
                """
        tx.run(similarity_query, movie1_id=movie1_id, movie2_id=movie2_id)

    @staticmethod
    def create_movie(tx, row):
        # Create movie node
        movie_query = """
        MERGE (movie:Movie {id: $id})
        ON CREATE SET 
        movie.title = $title,
        movie.original_title = $original_title,
        movie.original_language = $original_language,
        movie.overview = $overview,
        movie.tagline = $tagline,
        movie.release_date = date($release_date),
        movie.poster_path = $poster_path,
        movie.popularity = toFloat($popularity),
        movie.vote_count = toInteger($vote_count),
        movie.vote_average = toFloat($vote_average),
        movie.budget = toInteger($budget),
        movie.revenue = toInteger($revenue),
        movie.runtime = toInteger($runtime),
        movie.status = $status,
        movie.adult = $adult,
        movie.backdrop_path = $backdrop_path,
        movie.genre_names = $genre_names,
        movie.collection = $collection,
        movie.collection_name = $collection_name
        """

        tx.run(movie_query, row)
