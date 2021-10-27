import tensorflow as tf
import cv2
import os
import numpy as np

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

def extract_image_patch(image, bbox, patch_shape):
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image

def run():
    dir = 'MOT16/test/MOT16-06'
    model = tf.saved_model.load("saved_model")
    min_height = 0
    min_confidence = 0.3
    nms_max_overlap = 0.8
    max_cosine_distance = 0.2
    nn_budget = 100

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []
        
    image_filenames = os.listdir(dir + '/img1')
    detections = np.loadtxt(dir + '/det/det.txt', delimiter=',')

    for image_filename in image_filenames:
        #print(image_filename)
        image = cv2.imread(dir + '/img1/' + image_filename)
        frame_index = int(os.path.splitext(image_filename)[0])
        image_shape = [128, 64]
        frame_indices = detections[:,0].astype(np.int)
        mask = frame_indices == frame_index
        frame_detections = detections[mask]
        patches = []
        for fd in frame_detections:
            fd_details = fd[2:6]
            patch = extract_image_patch(image, fd_details, image_shape)
            #cv2.imshow('path', patch)
            #cv2.waitKey(0)
            patches.append(patch)

        input = tf.convert_to_tensor(np.asarray(patches))
        output = model.signatures['serving_default'](input)

        detection_list = []
        for i in range(len(frame_detections)):
            feature = output['out'][i]
            frame_detection = frame_detections[i][2:6]
            confidence = frame_detections[i][6]
            detection_list.append(Detection(frame_detection, confidence, feature))

        detection_list = [d for d in detection_list if d.confidence >= min_confidence]
        boxes = np.array([d.tlwh for d in detection_list])
        scores = np.array([d.confidence for d in detection_list])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detection_list = [detection_list[i] for i in indices]

        #for i in range(len(detection_list)):
        #    print(detection_list[i].tlwh)

        tracker.predict()
        tracker.update(detection_list)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([frame_index, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        #for d in detection_list:
        #    p1 = int(d.tlwh[0]), int(d.tlwh[1])
        #    p2 = int(d.tlwh[0] + d.tlwh[2]), int (d.tlwh[1] + d.tlwh[3])
        #    cv2.rectangle(image, p1, p2, [255,255,255], 1)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            p1 = int(bbox[0]), int(bbox[1])
            p2 = int(bbox[0] + bbox[2]), int (bbox[1] + bbox[3])
            cv2.rectangle(image, p1, p2, [255, 0, 0], 1)
            cv2.putText(image, str(track.track_id), p1, cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

        cv2.imshow('image', image)
        cv2.waitKey(1)


    f = open('v2_results.txt', 'w')
    for result in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            result[0], result[1], result[2], result[3], result[4], result[5]),file=f)

if __name__ == "__main__":
    run()