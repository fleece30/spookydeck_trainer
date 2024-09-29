import csv
from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
username = "neo4j"
password = "*123"


class DBPopulator:
    def __init__(self):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def create_indexes(self):
        with self.driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Movie) ON (m.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (p:Director) ON (p.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (p:Cast_Member) ON (p.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (cn:Collection) ON (cn.name)")

    def import_csv(self, input_file):
        with self.driver.session() as session:
            with open(input_file, 'r') as csv_file:
                reader = csv.DictReader(csv_file)
                rows = list(reader)  # Read all rows into memory
                total = len(rows)  # Get total number of rows
                for counter, row in enumerate(rows, start=1):
                    try:
                        print(f"Finished {counter}/{total}\r", end="")
                        session.execute_write(self.create_data, row)
                    except Exception as e:
                        print(f"Error: {e} in row {row}")


    @staticmethod
    def create_data(tx, row):
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

        # Create collection node
        if row['collection_name'] != "NA":
            collection_query = """
            MERGE (collection:Collection {name: $collection_name})
            MERGE (movie:Movie {id: $id})
            MERGE (movie)-[r:BELONGS_TO]->(collection)
            """
            tx.run(collection_query, id=row['id'], collection_name=row['collection_name'])

        # Create collection node
        if row['director']:
            director_query = """
                MERGE (director:Director {name: $director})
                MERGE (movie:Movie {id: $id})
                MERGE (director)-[r:DIRECTED]->(movie)
                """
            tx.run(director_query, id=row['id'], director=row['director'])

        # Create cast relationships
        if row['cast']:
            cast_members = [name.strip() for name in row['cast'].split(',')]
            for cast_member in cast_members:
                if cast_member:
                    cast_query = """
                    MERGE (person:Cast_Member {name: $name})
                    MERGE (movie:Movie {id: $id})
                    MERGE (person)-[:ACTED_IN]->(movie)
                    """
                    tx.run(cast_query, id=row['id'], name=cast_member)

