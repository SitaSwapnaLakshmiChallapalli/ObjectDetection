import cv2
import numpy as np

class VehicleDetector:

    def __init__(self):
        # GIVE YOUR FILE PATH
        net = cv2.dnn.readNet("../venv/dnn_model/objects.weights", "../venv/dnn_model/objects.cfg")
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(832, 832), scale=1 / 255)
        self.object = []
        self.classes_allowed = [2, 3, 5, 6, 7]

    def getObjectName(self):
        return self.object

    def detect_vehicles(self, img):
        with open("coco.names", "r", encoding="utf-8") as f:
            labels = f.read().strip().split("\n")
        vehicles_boxes = []
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)
        if len(class_ids) > 1:
            print(labels[class_ids[0]])
        for class_id, score, box in zip(class_ids, scores, boxes):
            if score < 0.5:
                continue

            if class_id in self.classes_allowed:
                vehicles_boxes.append(box)
                self.object.append(labels[class_id])

        return vehicles_boxes

