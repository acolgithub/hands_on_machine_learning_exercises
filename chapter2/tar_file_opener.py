from pathlib import Path
import tarfile
import urllib.request
import requests
import pandas as pd

def load_tgz_data(tarfile_input, tarurl=""):

    # first check that tarfile has .tgz file extension and only one period
    tarfile_split = tarfile_input.split(".")
    if len(tarfile_split) != 2 or tarfile_split[1] != "tgz":
        print("Not a tarfile.")
        return None

    # create expected path of tarfile
    tarfile_expected_path = f"datasets/{tarfile_input}"  # expected tarfile path string
    tarfile_path = Path(tarfile_expected_path)  # convert to path

    # next check if dataset file exists
    if not tarfile_path.is_file():

        # if file does not exist create datasets directory
        Path("datasets").mkdir(parents=True, exist_ok=True)

        # check if url was given to get tarfile
        if tarurl:
            url = tarurl
        else:
            print("No such tgz file.\nNeed a URL.")
            return None
        
        # check that url ends with proper tgz file
        tarfile_length = len(tarfile_input)
        if url[-tarfile_length:] != tarfile_input:
            print(f"tgz file does not match url end.")
            return None
                
        # check if url exists
        result = requests.head(url)
        if result.status_code not in [200, 302]:
            print(f"Status code: {result.status_code}\n{url} was not found.")
            return None
        
        # if url is given try to get file
        urllib.request.urlretrieve(url=url, filename=tarfile_path)

        # extract from tgz file
        with tarfile.open(tarfile_path, mode="r:gz") as open_file:
            open_file.extractall(path="datasets")

    # remove .tgz to name file
    split_file_name = tarfile_input[:-4]
    
    # read in extracted csv file
    return pd.read_csv(Path(f"datasets/{split_file_name}/{split_file_name}.csv"))




























