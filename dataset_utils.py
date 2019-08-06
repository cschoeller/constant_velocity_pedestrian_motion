import json


class DetectedObject:

    def __init__(self, obj_id, position):
        self.id = obj_id
        self.position = position
    
    def from_json(obj_json):
        return DetectedObject(obj_json['id'], obj_json['position'])

class Detection:

    def __init__(self, timestamp, size, object_list):
        self.timestamp = timestamp
        self.size = size
        self.object_list = object_list

    def from_json(detection_json):
        timestamp = detection_json['timestamp']
        size = detection_json['size']
        object_list = []
        for obj_json in detection_json['object_list']:
            object_list.append(DetectedObject.from_json(obj_json))
        return Detection(timestamp, size, object_list)

    def objects(self):
        return self.object_list

    def __len__(self):
        return len(self.object_list)

class Trajectory():

    def __init__(self, obj_id, start_time, positions=None):
        self.obj_id = obj_id
        self.start_time = start_time
        self.positions = [] if positions is None else positions

    def add_position(self, position):
        self.positions.append(position)

    def __len__(self):
        return len(self.positions)

class Sample():
    def __init__(self, obj_id, start_time, positions=None):
        self.trajectory = Trajectory(obj_id, start_time, positions=positions)
    
    @classmethod
    def from_trajectory(cls, trajectory):
        return cls(trajectory.obj_id, trajectory.start_time, positions=trajectory.positions)

    def add_position(self, position):
        self.trajectory.add_position(position)

    @property
    def positions(self):
        return self.trajectory.positions

    @property
    def obj_id(self):
        return self.trajectory.obj_id

    def __len__(self):
        return len(self.trajectory)
    
    def slice(self, sequence_length, min_length):
        """
        Slice the sample into multiple samples of at least length min_length and maximum sequence_length with
        a sliding window and step size 1.
        """

        if len(self) < min_length:
            return []

        split_samples = []
        start_index = 0
        end_index = sequence_length
        while start_index < len(self):
            new_trajectory = Trajectory(self.obj_id, self.trajectory.start_time + start_index, positions=self.positions[start_index:end_index])
            new_sample = Sample.from_trajectory(new_trajectory)
            start_index += 1
            end_index = start_index + sequence_length
            
            if len(new_sample) < min_length: # don't consider too short samples
                continue

            split_samples.append(new_sample)

        return split_samples
