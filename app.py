from typing import MutableMapping
from flask import Flask, render_template, request
import requests
import pickle
import pandas as pd
import numpy as np
import sklearn
import librosa
import soundfile
import os
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

def load_data(file_name_1):
    x,y=[],[]
    feature=extract_feature(file_name_1, mfcc=True, chroma=True, mel=True)
    file_name = os.path.basename(file_name_1)
    emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
    emotion=emotions[file_name.split("-")[2]]
    x.append(feature)
    y.append([emotion,file_name])
    x.append(feature)
    y.append([emotion,"test"])
    return train_test_split(np.array(x), y, test_size=1, random_state=9)


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        print(request.files['file'])
        
        file = request.files['file'].filename
        request.files['file'].save(f"{file}")
        
        x_train,x_test,y_trai,y_tes=load_data(file)
        prediction = model.predict(x_test)
        label_map =   ['calm', 'happy', 'fearful', 'disgust']
        
        

        #final_prediction = label_map[prediction]

        print("Done")
        print(prediction,"<<<<<<")
    #return final_prediction
    return render_template('index.html',prediction_text=f'Emotion={prediction}')
        
if __name__=="__main__":
    app.run(debug=True)
