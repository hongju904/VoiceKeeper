# UBUNTU VIRTUAL MACHINE FOR VOICE ANALYSIS


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
import json

from azure.storage.blob import BlobServiceClient

import numpy as np
import pandas as pd

from pydub import AudioSegment
from pydub.silence import split_on_silence

import torch
import torch.nn.functional as F
import torchaudio
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel, Wav2Vec2Model

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput

from flask import Flask

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

@app.route('/')
def model():
    @dataclass
    class SpeechClassifierOutput(ModelOutput):
        loss: Optional[torch.FloatTensor] = None
        logits: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None'


    class Wav2Vec2ClassificationHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            x = features
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)
            return x


    class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.pooling_mode = config.pooling_mode
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = Wav2Vec2ClassificationHead(config)
            self.init_weights()

        def freeze_feature_extractor(self):
            self.wav2vec2.feature_extractor._freeze_parameters()

        def merged_strategy(self, hidden_states, mode="mean"):
            if mode == "mean":
                outputs = torch.mean(hidden_states, dim=1)
            elif mode == "sum":
                outputs = torch.sum(hidden_states, dim=1)
            elif mode == "max":
                outputs = torch.max(hidden_states, dim=1)[0]
            else:
                raise Exception("The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")
            return outputs

        def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            outputs = self.wav2vec2(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
            logits = self.classifier(hidden_states)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SpeechClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


    def speech_file_to_array_fn(path):
        speech_array, _sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech


    def predict(path, sampling_rate):
        speech = speech_file_to_array_fn(path)
        speech = speech.reshape(-1)
        inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(device) for key in inputs}
        with torch.no_grad():
            logits = model(**inputs).logits
        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
        return outputs


    def remove_silence(path):
        global duration_in_sec
        sound = AudioSegment.from_file(path)
        duration_in_sec = len(sound) / 1000
        audio_chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=-40) # -35, -30
        sum = 0
        for chunk in audio_chunks:
            sum = sum + chunk
        sum.export('silence_removed_audio', format='wav')


    def split_and_predict(data):
        total = []
        t1 = 0
        t2 = 4000
        Audio = AudioSegment.from_wav(data)
        for i in range (len(Audio)):
            newAudio = Audio[t1:t2]
            filename = 'splitted audio/'+'audio_cut'+'_'+ str(i)+'.wav'
            newAudio.export(filename, format='wav')
            result = predict(filename, sampling_rate)
            max = float(result[0]['Score'][:-1])
            max_idx = 0
            for idx, data in enumerate(result):
                if max <= float(data['Score'][:-1]):
                    max = float(data['Score'][:-1])
                    max_idx = idx
            total.append(result[max_idx]['Emotion'])
            os.remove(filename)

            t1 = t1 + 4000
            t2 = t2 + 4000
            if t2 > len(Audio):
                break
        return total


    def test_return_df(path):
        print("\nREMOVE SILENCE")
        remove_silence(path)
        print(f"\nANALYZE AUDIO FILE ...")
        result = split_and_predict('silence_removed_audio')
        print("\nANALYSIS COMPLETE!")
        unique, counts = np.unique(result, return_counts=True)
        pred_df = pd.DataFrame(data=counts, index=unique, columns=['COUNT'])
        pred_df['RATIO'] = pred_df.div(pred_df.sum(axis=0), axis=1)
        pred_df = pred_df.sort_values(by=['RATIO'], ascending=[False])
        pred_df.drop(['COUNT'], axis=1, inplace=True)

        for idx in ['anger', 'disgust', 'fear', 'sadness', 'happiness']:
            if idx not in pred_df.index:
                pred_df.loc[idx] = 0
        return pred_df



    # SET MODEL 
    print("\nSTART SPEECH ANALYSIS\n\n\n")

    if not os.path.exists('splitted audio'):
        os.mkdir('splitted audio')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name_or_path = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
    config = AutoConfig.from_pretrained(model_name_or_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    sampling_rate = feature_extractor.sampling_rate
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)



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
        print("NO AUDIO FILE")
        exit()

    file_name = file_list[0]
    try:
        with open(file=file_name, mode='wb') as download_file:
            download_file.write(container_client.download_blob(blob.name).readall())
        print(f"AUDIO FILE DOWNLOADED : {file_name}")
    except Exception as ex:
        print(f"ERROR: {ex}")



    # RUN MODEL
    speech_df = test_return_df(file_name)



    # MAKE RESULT DATA
    # (1) RECORD INFORMATION
    num_info = os.path.splitext(file_name)[0][:-14]
    time_info = os.path.splitext(file_name)[0][-4:]
    minutes, seconds = divmod(duration_in_sec, 60)
    duration_info = f"{int(minutes):02d}:{int(seconds):02d}"
    date_info = os.path.splitext(file_name)[0][-13:-5]
    info = {
        'NUMBER':num_info,
        'TIME':time_info,
        'DURATION':duration_info,
        'DATE':date_info
        }
    info_dict = {'INFO':info}

    # (2) EMOTION RATIO - SPEECH
    emotion_df = pd.DataFrame(columns=['RATIO'])
    emotion_df.loc['angry'] = speech_df.loc['anger'].values*0.9
    emotion_df.loc['positive'] = (speech_df.loc['disgust'].values + speech_df.loc['happiness'].values)*0.4
    emotion_df.loc['fear'] = speech_df.loc['fear'].values*0.4
    emotion_df.loc['sad'] = speech_df.loc['sadness'].values*0.8
    emotion_df = emotion_df.sort_values(by=['RATIO'], ascending=[False])
    emotion_dict = emotion_df.to_dict()

    # (3) FINAL RESULT
    final_dict = info_dict.copy()
    final_dict.update(emotion_dict)
    final_json = json.dumps(final_dict, ensure_ascii=False)



    # UPLOAD RESULT DATA IN BLOB STORAGE
    connect_str = 'CONNECTION KEY'
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container='JSON CONTAINER', blob=os.path.splitext(file_name)[0]+'_speech'+'.json')

    try:
        blob_client.upload_blob(final_json)
    except Exception as ex:
        print(f"\ERROR: {ex}")
    else:
        print(f"\nJSON FILE UPLOADED")



    # INITIALIZE LOCAL DIRECTORY
    os.remove(file_name)
    os.remove('/home/USER/voicekeeper/silence_removed_audio')
    os.rmdir('/home/USER/voicekeeper/splitted audio')
    print("\nDONE")



    return "OK"