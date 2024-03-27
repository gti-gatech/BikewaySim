import requests
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def download_laz_files(url_file, output_directory):
    
    #read the list of raster links needed for 1m resolution elevation data
    urls = open(url_file,'r')
    urls = [text.split('\n')[0] for text in urls.readlines()]

    output_path = Path(output_directory)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    for url in urls:
        url = url.strip()
        file_name = url.split('/')[-1]
        output_file_path = output_path / file_name

        if output_file_path.exists():
            print(f"Skipping {url}. File already exists.")
        else:
            print(f"Downloading {url.split('/')[-1]}...")
            response = requests.get(url)
            
            if response.status_code == 200:
                with open(output_file_path, 'wb') as output_file:
                    output_file.write(response.content)
                print("Download successful.")
            else:
                print(f"Failed to download {url}. Status code: {response.status_code}")

if __name__ == "__main__":
    url_file_path = Path.home() / 'Documents/GitHub/BikewaySimDev/add_elevation_data/dem_links.txt'  # Replace with your actual file path
    output_directory = Path("D:/dem_files")  # Replace with your desired output directory

    download_laz_files(url_file_path, output_directory)
