import requests
import azure.functions as func
from concurrent.futures import ThreadPoolExecutor


def main(myblob: func.InputStream):
    speech_model_url = "UBUNTU SERVER IP 1"
    text_model_url = "UBUNTU SERVER IP 2"



    # PARALLEL ARCHITECTURE
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(requests.get, url) for url in [speech_model_url, text_model_url]]
        results = [future.result() for future in futures]



    for response in results:
        print(f"{response}")