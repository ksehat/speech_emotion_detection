import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForAudioFrameClassification
import torch
import torch.nn as nn
import torchaudio
import keyboard


class EmotionClassifier(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.feature_extractor = Wav2Vec2ForAudioFrameClassification.from_pretrained("anton-l/wav2vec2-base-superb-sd")
        self.feature_extractor.to(device)
        # self.conv_layer = nn.Sequential(
        #     nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1),
        #     nn.ReLU())
        self.flattening = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(624, 700),
            nn.ReLU(),
            nn.Linear(700, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            # nn.Linear(250, 100),
            # nn.ReLU(),
            nn.Linear(300, num_classes)
        )

    def forward(self, input_ids):
        features = self.feature_extractor(input_ids)
        # features_pooled = self.pooling(features.transpose(1, 2)).squeeze(-1)
        features_flatten = self.flattening(features['logits'])
        output = self.classifier(features_flatten)
        return output


class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, processor, max_length, label_mapping):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.max_length = max_length
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.file_paths[idx])
        if waveform.shape[0] != 1:
            waveform = waveform[0:1]
        # waveform = Normalizer().fit_transform(waveform)
        inputs = self.processor(waveform.squeeze(), return_tensors="pt", padding="max_length", truncation=True,
                                max_length=self.max_length, sampling_rate=16000)
        label = torch.tensor(self.label_mapping[self.labels[idx]], dtype=torch.long)
        one_hot_label = nn.functional.one_hot(label, num_classes=len(self.label_mapping)).squeeze(0)
        return inputs['input_values'].squeeze(), one_hot_label


# Example usage
if __name__ == "__main__":
    df = pd.read_csv('D:\speech_emotion_detection/all_path_emotion.csv').set_index('Unnamed: 0')
    train_df, test_df = train_test_split(df, test_size=0.01, stratify=df["emotion"], random_state=42, shuffle=True)
    train_df.reset_index(inplace=True)
    test_df.reset_index(inplace=True)

    max_length = 100000
    label_mapping = {
        'neutral': 0,
        'calm': 1,
        'happy': 2,
        'sad': 3,
        'angry': 4,
        'fear': 5,
        'disgust': 6,
        'surprise': 7
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sd")

    train_dataset = CustomDataset(train_df['path'], train_df['emotion'], processor, max_length,
                                  label_mapping)
    test_dataset = CustomDataset(test_df['path'], test_df['emotion'], processor, max_length,
                                 label_mapping)
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    model = EmotionClassifier(feature_size=32, num_classes=len(label_mapping)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    break_training = 0
    for epoch in range(50):
        if break_training:
            break
        model.train()
        for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device).to(dtype=torch.float32))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if keyboard.is_pressed('q'):
            #     break_training = 1
            #     print("Training interrupted by user.")
            #     break
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for inputs, labels in tqdm(test_dataloader, desc=f"Epoch {epoch + 1}"):
                outputs = model(inputs.to(device))
                predicted = torch.argmax(outputs, 1)
                total_correct += (predicted == torch.argmax(labels.to(device), 1)).sum().item()
                total_samples += labels.size(0)

            accuracy = total_correct / total_samples
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Validation Accuracy: {accuracy}")

        torch.save(model.state_dict(), f"D:\speech_emotion_detection/pretrained_model_2/emotion_classifier_{epoch}.pth")
        model.feature_extractor.save_pretrained(
            f"D:\speech_emotion_detection/pretrained_model_2/wav2vec2_model_{epoch}.pth")

    test_df.to_csv('D:\speech_emotion_detection/pretrained_model_2/evaluation_data_for_emotion_model.csv')

# torch.save(model.feature_extractor.state_dict(), 'path/to/save/model_weights.pth')
# model.feature_extractor.config.save_pretrained('path/to/save')
