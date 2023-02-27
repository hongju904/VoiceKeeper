import logging
import json
import azure.functions as func
from azure.storage.blob import BlobServiceClient


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')



    # CHECK JSON DATA
    connection_string = "CONNECTION KEY"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = "JSON CONTAINER"

    filename = req.params.get('filename')
    blob_names = [filename+'_text.json', filename+'_speech.json']
    container_client = blob_service_client.get_container_client(container_name)
    
    for blob_name in blob_names:
        blob_client = container_client.get_blob_client(blob_name)
        if not blob_client.exists():
            return func.HttpResponse(f"{blob_name} does not exist in {container_name} container", status_code=404)



    # GET JSON DATA
    data = {}
    for blob_name in blob_names:
        blob_client = container_client.get_blob_client(blob_name)
        data[blob_name] = json.loads(blob_client.download_blob().readall().decode())

    speech_dict = data[blob_names[0]]
    text_dict = data[blob_names[1]]



    # CALCULATE EMOTION RATIO : SPEECH + TEXT
    for key, value in text_dict['RATIO'].items():
        speech_dict['RATIO'][key] += value

    total = sum(speech_dict['RATIO'].values())
    for key in speech_dict['RATIO']:
        speech_dict['RATIO'][key] = round(speech_dict['RATIO'][key] / total, 2)



    # SAVE OR NOT
    save = 0
    if((speech_dict['RATIO']['fear'] > 0.3) or (speech_dict['RATIO']['angry'] > 0.3) or (speech_dict['RATIO']['sad'] > 0.3)):
        save = 1
    save_dict = {'SAVE':save}



    # MERGE
    del text_dict['RATIO']
    merged_dict = speech_dict.copy()
    merged_dict.update(text_dict)
    merged_dict.update(save_dict)



    # INITIALIZE JSON CONTAINER
    try:
        for blob in container_client.list_blobs():
            container_client.delete_blob(blob)
        print(f"\nREMOVE ALL BLOB")
    except Exception as ex:
        print(f"\nError: {ex}")



    # INITIALIZE M4A CONTAINER
    container_client = blob_service_client.get_container_client(container='M4A CONTAINER')
    try:
        for blob in container_client.list_blobs():
            container_client.delete_blob(blob)
        print(f"\nREMOVE ALL BLOB")
    except Exception as ex:
        print(f"\nError: {ex}")



    return func.HttpResponse(json.dumps(merged_dict, ensure_ascii=False), status_code=200)