from dataclasses import dataclass
from time import sleep
import os
from time import time
from typing import Union, Tuple, List
from datetime import datetime
import json
import requests
import cv2
import numpy as np
import pickle

from AlertSystem.detector import ObjectDetection


@dataclass
class AlertSystem:
    areas_file: str = "areas.pkl"
    alert_interval: int = 30
    telegram_messages: bool = True
    alert_msg = "Suspicious traffic in your secured area :O"
    recordings_path: Union[str, bool] = "recordings"
    recording_delay: float = 0.5
    __tokens_file_path: str = "tokens.json"

    def __post_init__(self) -> None:
        try:
            with open(self.areas_file, "rb") as f:
                self.areas = pickle.load(f)

            if self.telegram_messages:
                with open(self.__tokens_file_path) as f:
                    data = json.load(f)
                    self.__bot_token, self.__chat_id = data["BotToken"], data["ChatId"]

        except (FileNotFoundError, FileExistsError):
            self.areas = []
        self.__area_points = []

        if self.recordings_path:
            if not os.path.isdir(self.recordings_path):
                os.mkdir(self.recordings_path)

    @staticmethod
    def __nothing(x) -> None:
        pass

    def __mouse_click(self, event, x, y, flags, params) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__area_points.append([x, y])

        if event == cv2.EVENT_RBUTTONDOWN:
            for index, area in enumerate(self.areas):
                ppt = cv2.pointPolygonTest(area, (x, y), False)
                if ppt in (1, 0):
                    self.areas.pop(index)

    def send_tg_message(self, msg: str) -> None:
        url = f"https://api.telegram.org/bot{self.__bot_token}/sendMessage?" \
              f"chat_id={self.__chat_id}&parse_mode=Markdown&text={msg}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error when making sending tg message: {response.status_code}, {response.content}")

    @staticmethod
    def setup_detector(weights_path: str = r"model/yolov4-tiny.weights", config_path: str = r"model/yolov4-tiny.cfg",
                       classes_file: str = r"model/classes.txt", nms_threshold: float = .3,
                       conf_threshold: float = .3, image_w: int = 416, image_h: int = 416) -> ObjectDetection:

        return ObjectDetection(weights_path=weights_path,
                               config_path=config_path,
                               classes_file=classes_file,
                               nms_threshold=nms_threshold,
                               conf_threshold=conf_threshold,
                               image_w=image_w,
                               image_h=image_h)

    def run_main(self, detector: ObjectDetection, allowed_classes: Union[bool, List[int]] = False, draw: bool = True,
                 capture_input: Union[int, str] = 0, max_corner_number: int = 20,
                 border_color: Tuple[int, int, int] = (255, 255, 255),
                 fill_color: Tuple[int, int, int] = (255, 0, 0),
                 point_color: Tuple[int, int, int] = (255, 0, 255)) -> None:

        cap = cv2.VideoCapture(capture_input)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        cv2.namedWindow("Options")
        cv2.createTrackbar("MaxNumberOfCorners", "Options", 4, max_corner_number, self.__nothing)
        cv2.setTrackbarMin("MaxNumberOfCorners", "Options", 3)

        alpha = 0.4
        alert_sent_time = None
        record = False
        out = None
        p_time = 0
        detection_time = None

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if alert_sent_time:
                diff = int(time() - alert_sent_time)
                if diff >= self.alert_interval:
                    alert_sent_time = None

            if record:
                out.write(frame)
                cv2.putText(frame, f"Recording...", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 4)

            frame, detections = detector.detect(frame, allowed_classes, draw)
            overlay = frame.copy()

            max_corners = cv2.getTrackbarPos("MaxNumberOfCorners", "Options")
            ppts = []
            for area in self.areas:
                cv2.polylines(overlay, [area], True, border_color, 8)
                cv2.fillPoly(overlay, [area], fill_color)

                for detection in detections:
                    d_x, d_y, d_w, d_h = detection[:4]
                    d_center = (d_x + d_w // 2, d_y + d_h // 2)

                    cv2.circle(frame, d_center, 5, point_color, -1)

                    ppt = cv2.pointPolygonTest(area, d_center, False)
                    if ppt in (1, 0):
                        cv2.fillPoly(overlay, [area], (0, 0, 200))
                        ppts.append(1)

            # check if there is at least 1 detection in secured areas
            if any(ppts):
                if not alert_sent_time and self.telegram_messages:
                    self.send_tg_message(msg=f"Alert: {self.alert_msg}")
                    alert_sent_time = time()

                if not detection_time:
                    detection_time = time()

                if not record and self.recordings_path and time()-detection_time > self.recording_delay:
                    record = True
                    file_path = rf"{self.recordings_path}/{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.mp4"
                    out = cv2.VideoWriter(file_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
                    detection_time = None
            else:
                detection_time = None
                if record:
                    out.release()
                    record = False

            final_img = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            if len(self.__area_points) == max_corners:
                area_points = np.array([point for point in self.__area_points], np.int32)
                self.areas.append(area_points)
                self.__area_points = []
            else:
                for point in self.__area_points:
                    cv2.circle(final_img, tuple(point), 5, point_color, -1)

            c_time = time()
            fps = int(1/(c_time - p_time))
            p_time = c_time

            cv2.putText(final_img, f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 4)
            cv2.putText(final_img, f"FPS: {fps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

            cv2.setMouseCallback("res", self.__mouse_click)
            key = cv2.waitKey(1)
            if key == 27:
                break

            if key == ord("s"):
                cv2.putText(final_img, f"Areas saved to: {self.areas_file}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
                with open(self.areas_file, "wb") as f:
                    print(f"Areas saved to {self.areas_file}")
                    pickle.dump(self.areas, f)

            cv2.imshow("res", final_img)
        cap.release()
        if record:
            out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    alert_system = AlertSystem()
    detector = alert_system.setup_detector()
    alert_system.run_main(detector, allowed_classes=[0, 2, 3, 7], capture_input="Videos/thief_video2.mp4")
    # alert_system.run_main(detector, allowed_classes=[0, 2, 3, 7])
