import time
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
from emotion_model import HubertForSpeechClassification


def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = librosa.load(path, sr=None)
    speech = librosa.resample(y=speech_array, orig_sr=_sampling_rate, target_sr=sampling_rate)
    return speech


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    features = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Label": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
               enumerate(scores)]
    return outputs


def prediction(df_row):
    path, label = df_row["path"], df_row["label"]
    outputs = predict(path, sampling_rate)
    result = pd.DataFrame(outputs)
    return result


def get_data(test_df):
    return prediction(test_df)


def p2f(x):
    return float(x.strip('%')) / 100


df = pd.read_csv('label_path_data.csv').set_index('Unnamed: 0')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'], shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "D:\speech_emotion_detection\pretrained_model"
config = AutoConfig.from_pretrained(model_name_or_path + 'config')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path + 'feature_extractor')
sampling_rate = feature_extractor.sampling_rate
model = HubertForSpeechClassification.from_pretrained(model_name_or_path + 'model').to(device)

# save_directory = "D:\speech_emotion_detection\pretrained_model"
# config.save_pretrained(save_directory + 'config')
# feature_extractor.save_pretrained(save_directory + 'feature_extractor')
# model.save_pretrained(save_directory + 'model')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def get_data(i):
    df = prediction(test_df.iloc[i])
    df['Score'] = df['Score'].str.rstrip('%').astype('float') / 100.0
    return df


def create_polar_chart(i):
    plt.clf()

    df = get_data(i)

    values = df['Score'].tolist()
    values += values[:1]
    categories = df['Label'].tolist()
    N = len(categories)

    # Compute angle each bar is centered on:
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

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


# Create a figure
fig = plt.figure()

# Animate
ani = animation.FuncAnimation(fig, create_polar_chart, frames=len(test_df), repeat=True)

# Display the animation
plt.show()
