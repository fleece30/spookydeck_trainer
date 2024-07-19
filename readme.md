
# Spookydeck Data-side

This repo is a library for all the data-oriented functions for the spookydeck application.


## Supported functions

### `main.py`

`predict(movie_id_1, movie_id_2)`: returns cosine similarity between 2 movies

`rephrase_overviews_main()`: runs flow for rephrasing overviews

`run_cast_scraper()`: runs flow for scraping cast info

`populate_db()`: runs flow for populating the Neo4j database

`create_embeddings()`: runs flow for creating embedding and index mapping files for movie overviews

### `preprocessor.py`

`apply_text_to_lower_and_strip_to_dataframe(df)`: preprocesses dataframe to strip overviews and convert to lowercase

`text_to_lower_and_strip(text)`: Helper function for `apply_text_to_lower_and_strip_to_dataframe`

&nbsp;
### `sbert.py`

`setup_sbert()`: creates an instance of `SentenceTransformer`

`extract_embeddings(texts, model)`: uses the sentence-BERT model to tokenize the overviews and extract embeddings

&nbsp;
### `data.py`

`create_data_for_training(clean_overviews, movie_ids)`: sets up teh sentence-BERT model, extracts embeddings using the `sbert` functions and creates files for embeddings and movie_id mapping to corresponding indexes in the embedding file

&nbsp;
### `rephrase_overviews/main.py`

`read_csv()`: reads the csv file and create an array of movies

### `rephrase_overviews/scraper.py`

`login(driver)`: logs in to quillbot

`rephrase_overviews(driver, rows, output_file_name_suffix)`: pushes overviews to quillbot, rephrase, fetch and add to result

&nbsp;
### `cast_scraper/main.py`

`read_csv()`: reads the csv file and create an array of movies

`merge_csv()`: merges output csvs from batch processing into a single file

`remove_duplicates()`: removes duplicate entries from the result file

### `rephrase_overviews/scraper.py`

`setup()`: sets up the mozilla driver for selenium
`get_people()`: browses pages to fetch cast inofrmation

&nbsp;
### `populate_db/main.py`

`DBPopulator`: class for inporting data, creating indexed and pushing the data to the Neo4j DB

## Task checklist
- [x] Find movie databse
- [ ] Scrape movie ids for movies after 2022 from imdb
- [ ] Scrape data about these movies
- [x] Rephrase overviews
- [x] Fetch cast information
- [x] Create and populate Neo4j database
- [x] Create embeddings for overviews
- [ ] Create lists of similar and issimilar movies for every movie
- [ ] Create and train a Siamese Neural Network using this data
- [ ] Test