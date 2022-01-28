import tensorflow as tf
import numpy as np
import cv2

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


class DeepSort:

    def gpu_cfg(self, mem_lmt):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_lmt)])
            except RuntimeError as e:
                print(e)


    def initialize(self, filename):
        print(filename)
        self.gpu_cfg(2048)
        self.model = tf.saved_model.load("C:/Users/sr996/source/repos/deep_sort_v2/saved_model")
        self.min_height = 0
        self.min_confidence = 0.3
        self.nms_max_overlap = 0.8
        self.image_shape = [128,64]
        max_cosine_distance = 0.2
        nn_budget = 100
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)


    def extract_image_patch(self, image, bbox, patch_shape):
        bbox = np.array(bbox)
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image


    def __call__(self, arg):
        img_arg = arg[0]
        image = img_arg[0]
        
        dets_arg = arg[1]
        frame_detections = dets_arg[0]

        patches = []
        for fd in frame_detections:
            fd_details = fd[0:4]
            patch = self.extract_image_patch(image, fd_details, self.image_shape)
            patches.append(patch)

        input = tf.convert_to_tensor(np.asarray(patches))
        output = self.model.signatures['serving_default'](input)

        detection_list = []
        for i in range(len(frame_detections)):
            feature = output['out'][i]
            frame_detection = frame_detections[i][0:4]
            confidence = frame_detections[i][4]
            detection_list.append(Detection(frame_detection, confidence, feature))

        detection_list = [d for d in detection_list if d.confidence >= self.min_confidence]
        boxes = np.array([d.tlwh for d in detection_list])
        scores = np.array([d.confidence for d in detection_list])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detection_list = [detection_list[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(detection_list)

        results = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return np.array(results)

