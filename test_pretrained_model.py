import numpy as np
import librosa
import joblib
import pandas as pd
from dataset_generator import AudioDataset
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # A batch is a list of tuples with (input_values, label)
    input_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    input_values = pad_sequence(input_values, batch_first=False)
    return input_values, torch.tensor(labels)


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model = model.to('cuda')

df = pd.read_csv('D:\speech_emotion_detection\label_path_data.csv').set_index('Unnamed: 0')
labels = df["label"].unique().tolist()

le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])
joblib.dump(le, 'ohe.joblib')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_data = AudioDataset(train_df, processor)
test_data = AudioDataset(test_df, processor)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Remove the last layer
model.lm_head = None

# Add your custom layers
model.lm_head = torch.nn.Sequential(
    torch.nn.Linear(model.config.hidden_size, len(labels)),
    # torch.nn.Softmax()
)

# Define loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Train the model
for epoch in range(1000):
    for batch in train_loader:
        input_values = torch.stack(batch[0]).to("cuda")
        labels = batch[1].to("cuda")
        outputs = model(input_values)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(epoch)


# label_encoder = joblib.load('ohe.joblib')