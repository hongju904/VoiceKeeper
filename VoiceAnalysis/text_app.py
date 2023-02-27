# UBUNTU VIRTUAL MACHINE FOR TEXT ANALYSIS


# (0) UBUNTU SETUP
# sudo apt-get update
# sudo apt-get upgrade

# (1) CONDA VENV SETUP
# conda create -n VK.venv python
# conda activate VK.venv

# (2) INSTALL MODULES
# conda install ffmpeg
# python -m pip install --upgrade pip
# pip install -r requirements.txt
# sudo apt-get install libsndfile1

# (3) RUN FLASK APP
# python -m flask run --host=0.0.0.0 --port=5000

# (4) START RECORDING


import os
import time
import json

import azure.cognitiveservices.speech as speechsdk
from azure.storage.blob import BlobServiceClient

import numpy as np
import pandas as pd

from pydub import AudioSegment
import torch
import torch.nn.functional as F

from keybert import KeyBERT
# from kobert_tokenizer import KoBERTTokenizer
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
from transformers import AutoModelForSequenceClassification, BertModel, BertTokenizerFast, AutoModelForSequenceClassification, PreTrainedTokenizerFast, BartForConditionalGeneration

from flask import Flask

app = Flask(__name__)

@app.route('/')
def model():
    def from_file(file_path):
        speech_config = speechsdk.SpeechConfig(subscription='SUBSCRIPTION ID', region='REGION')
        speech_config.speech_recognition_language='ko-kr'
        audio_config = speechsdk.audio.AudioConfig(filename=file_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        done = False

        def stop_cb(evt):
            print('CLOSING on {}'.format(evt))
            nonlocal done
            done = True

        speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
        all_results = []

        def handle_final_result(evt):
            all_results.append(evt.result.text)
        speech_recognizer.recognized.connect(handle_final_result)
        speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
        speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
        speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
        speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)
        speech_recognizer.start_continuous_recognition()
        while not done:
            time.sleep(.5)
        speech_recognizer.stop_continuous_recognition()
        return all_results


    # def predict_voicephishing(text):
    #     tokenized = voicephishing_tokenizer(text, max_length=256, truncation=True, return_tensors='pt')
    #     logits = voicephishing_model(**tokenized).logits
    #     logits = F.softmax(logits, dim=-1)
    #     logits = logits.detach().cpu().numpy()
    #     idx = np.argmax(logits, axis=-1)

    #     if idx == 0:
    #         return {"voicephishing":0, "probability":(logits[0][0]*100).round(2)}
    #     else:
    #         return {"voicephishing":1, "probability":(logits[0][1]*100).round(2)}


    def predict_sentiment(text):
        tokenized = sentiment_tokenizer(text, max_length=256, truncation=True, return_tensors='pt')
        logits = sentiment_model(**tokenized).logits
        logits = F.softmax(logits, dim=-1)
        logits = logits.detach().cpu().numpy()
        pred_df = pd.DataFrame(index=['anger', 'sadness', 'anxiety', 'hurt', 'panic', 'happiness'], data=logits[0], columns=['RATIO'])
        return pred_df


    def noun(text):
        results = []
        result = kiwi.tokenize(text, normalize_coda=True, stopwords=stopwords)
        for i in range(len(result)):
            token, tag, _, _ = result[i]
            if len(token) != 1 and tag.startswith('N') or tag.startswith('SL'):
                results.append(token)
        results = ' '.join(results)
        return results


    def predict_keyword(text, method=noun):
        text = method(text)
        keyword = keyword_model.extract_keywords(text, keyphrase_ngram_range=(1,1), stop_words=None, top_n=3, use_mmr=True, diversity=0.6)
        keyword = " ".join(["#"+keyword[i][0].replace(" ", "_") for i in range(len(keyword))])
        return keyword


    def predict_summary(text):
        raw_input_ids = summarize_tokenizer.encode(text)
        input_ids = [summarize_tokenizer.bos_token_id] + raw_input_ids + [summarize_tokenizer.eos_token_id]
        input_ids = torch.tensor([input_ids])
        summary_ids = summarize_model.generate(
            input_ids=input_ids,
            eos_token_id=1,
            length_penalty=1.,
            max_length=48,
            num_beams=4,
        )
        result = summarize_tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        sentences = [x.strip() for x in result.split('.') if len(x) >= 10][:2]  # 2개의 문장으로 제한
        result = '. '.join(sentences).strip(' ?')
        return result + '.'



    # SET MODEL
    print("\nSTART TEXT ANALYSIS\n\n\n")

    # voicephishing_model_id = 'mk9165/mk-bert-voicefishing-classification'
    # voicephishing_tokenizer = KoBERTTokenizer.from_pretrained(voicephishing_model_id)
    # voicephishing_model = AutoModelForSequenceClassification.from_pretrained(voicephishing_model_id, num_labels=2)
    sentiment_model_id = 'mk9165/mk-bert-sentiment-classification'
    sentiment_tokenizer = BertTokenizerFast.from_pretrained(sentiment_model_id)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_id, num_labels=6)
    kw_model = BertModel.from_pretrained('skt/kobert-base-v1')
    keyword_model = KeyBERT(kw_model)
    kiwi = Kiwi(num_workers=0, load_default_dict=True, integrate_allomorph=True, model_type='sbg', typos='basic')
    stopwords = Stopwords()
    summarize_model_id = 'mk9165/mk-bart-small-v5'
    summarize_tokenizer = PreTrainedTokenizerFast.from_pretrained(summarize_model_id)
    summarize_model = BartForConditionalGeneration.from_pretrained(summarize_model_id)



    # GET AUDIO DATA FROM BLOB STORAGE
    connect_str = 'CONNECTION KEY'
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container='M4A CONTAINER')

    file_list = []
    for blob in container_client.list_blobs():
        if os.path.splitext(blob.name)[1] == '.m4a':
            file_list.append(blob.name)
            break
    if len(file_list)==0:
        print("NO AUDIO DATA")
        exit()

    file_name = file_list[0]
    try:
        with open(file=file_name, mode='wb') as download_file:
            download_file.write(container_client.download_blob(blob.name).readall())
        print(f"\nAUDIO FILE DOWNLOADED : {file_name}")
    except Exception as ex:
        print(f"\ERROR: {ex}")



    # CONVERT '.m4a' TO '.wav'
    m4a_audio = AudioSegment.from_file(file_name)
    file_name = os.path.splitext(file_name)[0]+'.wav'
    m4a_audio.export(file_name, format='wav')



    # RUN MODEL
    print(f"\nANALYZE AUDIO FILE ...\n")
    text = from_file(file_name)
    text = ' '.join(text)
    # voicephishing = predict_voicephishing(text)
    keyword = predict_keyword(text)
    summary = predict_summary(text)
    text_df = predict_sentiment(text)
    print("\nANALYSIS COMPLETE!")



    # MAKE RESULT DATA
    # (1) TEXT ANALYSIS
    text_dict = {
        # 'VOICEPHISHING':voicephishing,
        'KEYWORD':keyword,
        'SUMMARY':summary
    }

    # (2) EMOTION RATIO - TEXT
    emotion_df = pd.DataFrame(columns=['RATIO'])
    emotion_df.loc['angry'] =  text_df.loc['anger'].values*0.1
    emotion_df.loc['positive'] = text_df.loc['happiness'].values*0.5
    emotion_df.loc['fear'] = (text_df.loc['anxiety'].values + text_df.loc['panic'].values + text_df.loc['hurt'].values)*0.6
    emotion_df.loc['sad'] = text_df.loc['sadness'].values*0.2
    emotion_df = emotion_df.sort_values(by=['RATIO'], ascending=[False])
    emotion_dict = emotion_df.to_dict()

    # (3) FINAL RESULT
    final_dict = text_dict.copy()
    final_dict.update(emotion_dict)
    final_json = json.dumps(final_dict, ensure_ascii=False)



    # UPLOAD RESULT DATA IN BLOB STORAGE
    connect_str = 'CONNECTION KEY'
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container='JSON CONTAINER', blob=os.path.splitext(file_name)[0]+'_text'+'.json')

    try:
        blob_client.upload_blob(final_json)
    except Exception as ex:
        print(f"\nError: {ex}")
    else:
        print(f"\nJSON FILE UPLOADED")



    # INITIALIZE LOCAL DIRECTORY
    os.remove(file_name)
    os.remove(os.path.splitext(file_name)[0]+'.m4a')
    print("\nDONE")



    return "OK"