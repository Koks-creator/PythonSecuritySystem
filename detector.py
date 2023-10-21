from dataclasses import dataclass
from typing import Union, List
from typing import Tuple
import cv2
import numpy as np


np.random.seed(10)


@dataclass
class ObjectDetection:
    weights_path: str = r"model/yolov4.weights"
    config_path: str = r"model/yolov4.cfg"
    classes_file: str = r"model/classes.txt"
    nms_threshold: float = .3
    conf_threshold: float = .3
    image_w: int = 416
    image_h: int = 416

    def __post_init__(self) -> None:
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.class_list, self.class_colors = self.load_classes()
        self.class_list_index = [i for i in range(len(self.class_list))]

    def load_classes(self) -> Tuple[list, np.array]:
        with open(self.classes_file) as f:
            class_names = f.read().rstrip('\n').split("\n")
            color_list = np.random.uniform(low=0, high=255, size=(len(class_names), 3))
            return class_names, color_list

    def prepare_color(self, class_id: int) -> Tuple[int, ...]:
        class_color = [int(c) for c in self.class_colors[class_id]]
        return tuple(class_color)

    def detect(self, img: np.array, allowed_classes: Union[bool, List[int]] = False, draw: bool = False) -> Tuple[np.array, list]:
        ih, iw, _ = img.shape
        bbox = []
        class_ids = []
        confs = []

        if not allowed_classes:
            allowed_classes = self.class_list_index

        blob = cv2.dnn.blobFromImage(img, (1/255), (self.image_w, self.image_h), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        output_names = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_names)

        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id in allowed_classes:
                    if confidence > self.conf_threshold:
                        w, h = int(det[2] * iw), int(det[3] * ih)
                        x, y = int((det[0] * iw) - w / 2), int((det[1] * ih) - h / 2)

                        bbox.append((x, y, w, h))
                        class_ids.append(class_id)
                        confs.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(bbox, confs, self.conf_threshold, self.nms_threshold)

        bbox_list = []
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            class_id = int(class_ids[i])
            class_name = self.class_list[class_id].upper()
            class_color = self.prepare_color(class_id)

            bbox_list.append((x, y, w, h, class_name, class_color))
            if draw:
                cv2.rectangle(img, (x, y), (x + w, y + h), class_color, 2)
                cv2.putText(img, f"{self.class_list[class_id].upper()} {int(confs[i] * 100)}%", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 2)
        return img, bbox_list


if __name__ == '__main__':
    ob = ObjectDetection()

    img = cv2.imread("autgo.png")

    img, _ = ob.detect(img, draw=True)
    cv2.imshow("res", img)
    cv2.waitKey(0)


