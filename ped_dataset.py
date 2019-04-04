import os
import sys
import copy
import glob
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_utils import *


class PedDataset(Dataset):
    """
    Datset of pedestrian trajectories.
    """
    def __init__(self, dataset_path, sequence_length, observed_history, min_sequence_length):
        super(PedDataset, self).__init__()
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length
        self.observed_history = observed_history

        self.detection_timestamps, self.detection_paths = None, None
        self.name = None
        self.samples = []
        self.size = 0

        if self.dataset_path is not None:
            self.initialize_dataset()

    def initialize_dataset(self):
        data_path_expanded = os.path.join(self.dataset_path, 'data', '*.json')
        detection_paths = glob.glob(data_path_expanded)
        assert(len(detection_paths) > 0)
        self.detection_timestamps, self.detection_paths = self._ordered_timestamp_detection_paths(detection_paths)
        assert(len(self.detection_timestamps) == len(self.detection_paths))

        self._set_name(self.dataset_path)
        self._create_sample_sequences()
        self.size = len(self.samples)

    def __len__(self):
        return self.size
        
    def _ordered_timestamp_detection_paths(self, detection_paths):
        ordered_detections = []
        for detection_path in detection_paths:
            detection_file = open(detection_path, 'r')
            detection = json.load(detection_file)
            detection_file.close()
            detection = detection
            ordered_detections.append((detection['timestamp'], detection_path))

        ordered_detections = sorted(ordered_detections)
        timestamps, new_detection_paths = map(list, zip(*ordered_detections))
        return timestamps, new_detection_paths

    def _create_sample_sequences(self):
        detections = self._load_all_detections()
        ids_to_samples = self._create_samples(detections)
        self.samples = list(ids_to_samples.values())
        self.samples = self._slice_samples_by_sequence_length()

    def _create_samples(self, detections):
        ids_to_samples = dict()

        for detection in detections:
            timestamp = detection.timestamp

            for obj in detection.objects():
                if obj.id not in ids_to_samples:
                    ids_to_samples[obj.id] = Sample(obj.id, timestamp)
                sample = ids_to_samples[obj.id]
                sample.add_position(obj.position)
        return ids_to_samples

    def _load_all_detections(self):
        detections = []
        for detection_path in self.detection_paths:
            detection = self._load_detection(detection_path)
            detections.append(detection)
        return detections

    def _load_detection(self, detection_path):
        with  open(detection_path, 'r') as detection_file:
            detection_json = json.load(detection_file)
        detection = Detection.from_json(detection_json)
        return detection

    def _slice_samples_by_sequence_length(self):
        expanded_samples = []
        for sample in self.samples:
            sliced_samples = sample.slice(self.sequence_length, self.min_sequence_length)
            expanded_samples.extend(sliced_samples)
        return expanded_samples

    def __getitem__(self, index):
        sample = copy.deepcopy(self.samples[index])

        mask = self._compute_label_mask(sample)[self.observed_history:]
        mask = [torch.tensor(x) for x in mask]
        mask = torch.stack(mask)
        self._pad_sequence(sample)

        observed = torch.tensor(sample.trajectory.positions[:self.observed_history], dtype=torch.float32)
        y_trajectory = np.array(sample.trajectory.positions)
        y_delta = y_trajectory[self.observed_history:] - y_trajectory[self.observed_history-1:-1]
        y_delta = torch.tensor(y_delta, dtype=torch.float32)

        return observed, [y_delta, mask]

    def _compute_label_mask(self, sample):
        mask = np.ones(self.sequence_length)
        mask[len(sample):] = 0.
        return mask

    def _pad_sequence(self, sample):
        if len(sample) >= self.sequence_length:
            return

        padding_size = (self.sequence_length) - len(sample)
        for i in range(padding_size):
            sample.add_position([0, 0])

    def _set_name(self, dataset_path):
        info_path = os.path.join(dataset_path, 'dataset_info.json')
        dataset_info = json.load(open(info_path, 'r'))
        self.name = dataset_info['dataset_name']