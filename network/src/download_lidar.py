import requests
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time
import json

#1271

def download_with_retry(url,MAX_RETRIES,RETRY_DELAY):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.get(url, timeout=10)  # Set your desired timeout
            response.raise_for_status()  # Raise an HTTPError for bad responses
            return response
        except requests.exceptions.Timeout:
            print(f"Timeout error for {url}. Retrying...")
            retries += 1
            time.sleep(RETRY_DELAY)
        except requests.exceptions.RequestException as e:
            print(f"Error for {url}: {e}")
            break

    print(f"Failed to download {url} after {MAX_RETRIES} retries.")
    return None

def download_laz_files(url_file, output_directory,MAX_RETRIES,RETRY_DELAY):
    
    df = pd.read_csv(url_file,header=None)
    df['dataset'] = df.loc[:,0].apply(lambda txt: txt.split(' ')[-2])
    datasets = ['GA_Statewide_2018_B18_DRRA','GA_Central_2019_B19']
    df = df[df['dataset'].isin(datasets)]
    urls = df.loc[:,17].tolist()

    output_path = Path(output_directory)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    for url in tqdm(urls):
        url = url.strip()
        file_name = url.split('/')[-1]
        output_file_path = output_path / file_name

        if output_file_path.exists():
            #print(f"Skipping {url}. File already exists.")
            continue
        else:
            #print(f"Downloading {url.split('/')[-1]}...")
            #response = requests.get(url)
            response = download_with_retry(url,MAX_RETRIES,RETRY_DELAY)
            
            if response == None:
                continue
                #print(f"Failed to download {url}, too many tries")

            if response.status_code == 200:
                with open(output_file_path, 'wb') as output_file:
                    output_file.write(response.content)
                #print("Download successful.")
            else:
                print(f"Failed to download {url}. Status code: {response.status_code}")

if __name__ == "__main__":
    MAX_RETRIES = 3  # Maximum number of retries
    RETRY_DELAY = 60  # Delay in seconds between retries
    
    config = json.load((Path.cwd().parent / 'config.json').open('rb'))
    url_file_path = Path(config['usgs']) / 'lidar_links.csv' # Replace with your actual file path
    output_directory = Path(config['usgs']) / 'lidar_files'  # Replace with your desired output directory

    download_laz_files(url_file_path, output_directory,MAX_RETRIES,RETRY_DELAY)
