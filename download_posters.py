
from tqdm import tqdm
import urllib.request
from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()



def Download_posters(poster):
    save_location = r'D:\Nitin\pythonProject1moviegenre\posters/'
    for index, row in tqdm(poster.iterrows(), total=poster.shape[0]):
        url = row['Poster']  # Image URL
        image_name = str(row['imdbId']) + '.jpg'  # Use imdb_id as the file name
        file_path = save_location + image_name  # Full file path

        try:
        # Download the image
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded {image_name}")
        except Exception as e:
            print(f"Failed to download {image_name}: {e}")
