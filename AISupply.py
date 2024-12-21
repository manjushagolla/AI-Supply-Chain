import requests
import os

class SupplyChainDataHandler:
    def __init__(self, api_url: str, dataset_path: str = "dataimport.csv"):#add a file name to data set
        """
        Initialize the SupplyChainDataHandler with the API URL and dataset path.
        
        :param api_url: The URL to fetch the dataset.
        :param dataset_path: The local file path where the dataset will be saved.
        """
        if not api_url:
            raise ValueError("API URL cannot be empty.")
        if not api_url.startswith("http"):
            raise ValueError("API URL must start with 'http' or 'https'.")
        self.api_url = api_url
        self.dataset_path = dataset_path  

    def download_dataset(self):
        """
        Download the dataset from the given API URL and save it to the local machine.
        """
        try:
            
            response = requests.get(self.api_url)
            response.raise_for_status()  
            
            # Save the content to a file
            with open(self.dataset_path, 'wb') as file:
                file.write(response.content)
                
           
            print("Dataset downloaded successfully.")
            print(f"Dataset saved at: {os.path.abspath(self.dataset_path)}")  
            
        except requests.exceptions.RequestException as e:
            # Print error message if the request fails
            print(f"Error downloading dataset: {e}")
if __name__ == "__main__":
    API_URL = "https://newsapi.org/v2/everything?q=AI&apiKey=ba1bd98460234da9a7667d975d1e4fa8"#you can modify the api if needed

    # Initialize the handler with the API URL and optional dataset file path
    data_handler = SupplyChainDataHandler(api_url=API_URL)

    # Download the dataset
    data_handler.download_dataset()

