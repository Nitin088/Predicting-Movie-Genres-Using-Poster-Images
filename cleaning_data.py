
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

save_location = r'D:\Nitin\pythonProject1moviegenre\posters/'


def encoded_data():
    # encoding data
    data = pd.read_csv("MovieGenre.csv", encoding="latin1")
    data['Genre'] = data['Genre'].astype(str)
    data['Genre'] = data['Genre'].fillna('')
    mlb = MultiLabelBinarizer()
    encoded_genres = mlb.fit_transform(data['Genre'].str.split('|'))
    encoded_data_genre = pd.DataFrame(encoded_genres, columns=mlb.classes_)
    #cleaning data with less than 150 entries
    low_value_threshold = 150
    for col in encoded_data_genre.columns:
        if encoded_data_genre[col].value_counts().min() < low_value_threshold:
            encoded_data_genre = encoded_data_genre.drop(col, axis=1)
    encoded_data = pd.concat([data, encoded_data_genre], axis=1)
    return encoded_data, encoded_data_genre


def Graph(graph_data):
    plt.figure(figsize=(10, 6))
    graph_data.sum().sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title("Label Frequency Distribution")
    plt.xlabel("Genres")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def check_posters(df, folder_path):
    # Loop through each row in the dataframe
    rows_to_delete = []  # List to store indices of rows to delete

    for index, row in df.iterrows():
        poster_name = str(row['imdbId']) + '.jpg'
        # Construct the full path to the poster file
        poster_path = os.path.join(folder_path, poster_name)
        # Check if the poster file exists
        if not os.path.exists(poster_path):
            # If the poster doesn't exist, mark the row for deletion
            rows_to_delete.append(index)
    # Drop the rows where posters don't exist
    df.drop(rows_to_delete, inplace=True)
    return df

encoded_data, encoded_data_genre = encoded_data()
folder_path=r'D:\Nitin\pythonProject1moviegenre\posters/'
checked_data=check_posters(encoded_data,folder_path)
checked_data=checked_data.drop(columns=['Genre','Poster','Imdb Link','IMDB Score','Title'])
checked_data.to_csv('cleaned_data.csv', index=False)




