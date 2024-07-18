# Function to convert text to lowercase and strip space from either side
def text_to_lower_and_strip(text):
    return text.lower().strip()


# Apply the text transformation to the rows in the dataframe
def apply_text_to_lower_and_strip_to_dataframe(df):
    df['clean_overviews'] = df['overview'].apply(text_to_lower_and_strip)
