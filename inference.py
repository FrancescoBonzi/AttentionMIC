from typing import Union
import librosa
import openmic.vggish
import torch
from torch import Tensor

from pyannote.core import Segment, Annotation

import numpy as np
import matplotlib.pyplot as plt

from Attention import DecisionLevelSingleAttention
from utils import get_class_map
from classes import *

#filepath = "/Users/francesco.bonzi/Documents/datasets/jamendo/audio/01 - A smile on your face.mp3"

class Inference():

    def __init__(self, freq_bins=128, classes_num=20, emb_layers=3, hidden_units=128, drop_rate=0.6):
        model = DecisionLevelSingleAttention(freq_bins, classes_num, emb_layers, hidden_units, drop_rate)
        model_weights = torch.load("./log/ISMIR2019/attention/decisionlevelsingleattention_128_3_0.6_lr0.0005_noannealing_res_seed_0/best_val_loss.pth", map_location=torch.device('cpu'))
        model.load_state_dict(model_weights)
        model.eval()
        self.model = model
        self.classes_dict = get_class_map()

    def __call__(self, filepath: str, plot_=False) -> Tensor:
        audio, rate = librosa.load(filepath)
        _, features = openmic.vggish.waveform_to_features(audio, rate)
        features = features / 255.
        features = features[:features.shape[0] // 10 * 10, :]
        features = np.reshape(features, (-1, 10, 128))

        X = torch.tensor(features, requires_grad=False, dtype=torch.float32)
        embedding = self.model.emb(X)
        attention = torch.sigmoid(self.model.attention.att(embedding))
        prediction = self.model.attention(embedding)

        prediction_updated = { instr: [] for instr in self.classes_dict.keys() }
        for frame_idx in range(prediction.shape[0]):
            for instr in self.classes_dict.keys():
                att = attention[frame_idx, self.classes_dict[instr], :, 0]
                min_ = torch.min(att)
                max_ = torch.max(att)
                range_ = max_ - min_
                mean = range_ / 2
                att = att + 1 - mean
                prediction_updated[instr] += [prediction[frame_idx, self.classes_dict[instr]] * att]

        for instr in self.classes_dict.keys():
            prediction_updated[instr] = torch.cat(prediction_updated[instr], dim=0)

        if plot_:
            self.plot(prediction_updated)

        return prediction_updated


def plot(prediction: dict, plot_colorbar=True):
    for instr, pred in prediction.items():
        print(instr)
        plt.imshow(pred.detach().numpy().reshape((1, -1)), vmin=0, vmax=1, aspect=15)
        if plot_colorbar:
            plt.colorbar(location="bottom")
        plt.show()

def to_annotation(prediction: dict, uri: str, threshold: float = 0.5):
    annotation = Annotation(uri=uri)
    for instr, pred in prediction.items():
        pred = pred.detach().numpy() >= threshold
        for start in range(pred.shape[0]):
            if pred[start]:
                annotation[Segment(start, start + 1.), instr] = instr
    return annotation.support(0.5)

def merge_classes(prediction: Union[dict, Annotation]):
    
    # load merged class dict and invert it
    label_mapping = dict()
    for key, value in merged_classes.items():
        for v in value:
            label_mapping[v] = key

    if isinstance(prediction, dict):
        merge_classes_dict = {}
        for instr, pred in prediction.items():
            if label_mapping[instr] in merge_classes_dict:
                pred = np.maximum(merge_classes_dict[label_mapping[instr]], pred)
            merge_classes_dict[label_mapping[instr]] = pred
        prediction = merge_classes_dict
    elif isinstance(prediction, Annotation):
        prediction = prediction.rename_labels(label_mapping).support()
    else:
        raise NotImplementedError(type(prediction) + " is not supported")

    return prediction