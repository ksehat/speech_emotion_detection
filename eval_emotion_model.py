import torch
from emotion_model import CustomDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForAudioFrameClassification
from tqdm import tqdm
import torch
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.feature_extractor = Wav2Vec2ForAudioFrameClassification.from_pretrained(f"D:\speech_emotion_detection/pretrained_models/wav2vec2_model_40.pth")
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

df = pd.read_csv('label_path_data.csv').set_index('Unnamed: 0')

max_length = 100000
label_mapping = {
    'S': 0,
    'A': 1,
    'H': 2,
    'W': 3,
    'F': 4,
    'N': 5
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2FeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sd")

eval_dataset = CustomDataset(df['path'], df['emotion'], processor, max_length,
                              label_mapping)
eval_dataloader = DataLoader(eval_dataset, batch_size=5, shuffle=False)


model = EmotionClassifier(feature_size=32, num_classes=len(label_mapping))
model.load_state_dict(torch.load("D:\speech_emotion_detection/pretrained_models/emotion_classifier_40.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for inputs, labels in tqdm(eval_dataloader):
        outputs = model(inputs.to(device))
        predicted = torch.argmax(outputs, 1)
        total_correct += (predicted == torch.argmax(labels.to(device), 1)).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(accuracy)
