import pandas as pd
import os

# Paths for data.
Ravdess = "D:\speech_emotion_detection\data\kaggle dataset/ravdess/audio_speech_actors_01-24/"
Crema = "D:\speech_emotion_detection\data\kaggle dataset\cremad\AudioWAV/"
Tess = "D:\speech_emotion_detection\data\kaggle dataset/tess\TESS Toronto emotional speech set data/"
Savee = "D:\speech_emotion_detection\data\kaggle dataset\savee\ALL/"

ravdess_directory_list = os.listdir(Ravdess)
file_emotion = []
file_path = []
for dir in ravdess_directory_list:
    # as there are 20 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        # third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)

emotion_df = pd.DataFrame(file_emotion, columns=['emotion'])
path_df = pd.DataFrame(file_path, columns=['path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)
Ravdess_df.emotion.replace(
    {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)



crema_directory_list = os.listdir(Crema)
file_emotion = []
file_path = []
for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part = file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')

emotion_df = pd.DataFrame(file_emotion, columns=['emotion'])
path_df = pd.DataFrame(file_path, columns=['path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)



tess_directory_list = os.listdir(Tess)
file_emotion = []
file_path = []
for dir in tess_directory_list:
    directories = os.listdir(Tess + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[2]
        if part == 'ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(Tess + dir + '/' + file)

emotion_df = pd.DataFrame(file_emotion, columns=['emotion'])
path_df = pd.DataFrame(file_path, columns=['path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)



savee_directory_list = os.listdir(Savee)
file_emotion = []
file_path = []
for file in savee_directory_list:
    file_path.append(Savee + file)
    part = file.split('_')[1]
    ele = part[:-6]
    if ele == 'a':
        file_emotion.append('angry')
    elif ele == 'd':
        file_emotion.append('disgust')
    elif ele == 'f':
        file_emotion.append('fear')
    elif ele == 'h':
        file_emotion.append('happy')
    elif ele == 'n':
        file_emotion.append('neutral')
    elif ele == 'sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')

emotion_df = pd.DataFrame(file_emotion, columns=['emotion'])
path_df = pd.DataFrame(file_path, columns=['path'])
Savee_df = pd.concat([emotion_df, path_df], axis=1)

# Define the path to your folder containing the audio files
folder_path = 'D:\speech_emotion_detection\data/org audio files'
path_list, label_list = [], []
for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):
        # Extract the label from the filename
        label = [letter1 for letter1 in filename.split('.')[0] if letter1.isalpha()][1]
        file_path = os.path.join(folder_path, filename)
        # Append the file path and label to the data list
        path_list.append(file_path)
        label_list.append(label)

# Create a dataframe from the data list
Shemo_df = pd.DataFrame({
    'path': path_list,
    'emotion': label_list
})
Shemo_df.emotion.replace(
    {'N': 'neutral', 'H': 'happy', 'S': 'sad', 'A': 'angry', 'F': 'fear', 'W': 'surprise'}, inplace=True)

final_df = pd.concat([Ravdess_df,
                      Crema_df,
                      Tess_df,
                      Savee_df,
                      Shemo_df], axis=0)

final_df.to_csv('all_path_emotion.csv')
