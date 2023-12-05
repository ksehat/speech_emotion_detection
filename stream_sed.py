import pyaudio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from stream_sed_model import HubertForSpeechClassification


def create_polar_chart(i):
    plt.clf()

    df = get_data(i)

    values = df['Score'].tolist()
    categories = df['Label'].tolist()
    N = len(categories)

    # Compute angle each bar is centered on:
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Add the first value to the end of the values list
    values += values[:1]

    # Initialise the polar plot
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.2, 0.3, 0.4], ["10", "20", "30", "40"], color="grey", size=7)
    plt.ylim(0, 0.5)

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "D:\speech_emotion_detection\pretrained_model"
config = AutoConfig.from_pretrained(model_name_or_path + 'config')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path + 'feature_extractor')
sampling_rate = feature_extractor.sampling_rate
model = HubertForSpeechClassification.from_pretrained(model_name_or_path + 'model').to(device)

CHUNK = 1024
SAMPLE_FORMAT = pyaudio.paFloat32  # 16 bits per sample
CHANNELS = 1
FS = 44100  # Record at 44100 samples per second
SECONDS = 3
p = pyaudio.PyAudio()
stream = p.open(format=SAMPLE_FORMAT,
                channels=CHANNELS,
                rate=FS,
                frames_per_buffer=CHUNK,
                input=True)


def get_data(i):
    frames = []
    for i in range(0, int(FS / CHUNK * SECONDS)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))
    numpydata = np.hstack(frames)
    mono_data = numpydata.reshape(-1, 1).mean(axis=1)
    speech = librosa.resample(mono_data, orig_sr=FS, target_sr=sampling_rate)
    features = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Label": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
               enumerate(scores)]
    df = pd.DataFrame(outputs)
    df['Score'] = df['Score'].str.rstrip('%').astype('float') / 100.0
    return df


fig = plt.figure()
ani = animation.FuncAnimation(fig, create_polar_chart, frames=10, repeat=True)
plt.show()
print('ok')
