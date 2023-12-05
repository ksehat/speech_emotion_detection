import os
import pandas as pd

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
df = pd.DataFrame({
    'path': path_list,
    'label': label_list
})
df.to_csv('label_path_data.csv')

