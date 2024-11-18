import cv2
from ultralytics import YOLO
from collections import defaultdict
from typing import List, Dict


class CucumberDetection:
    """
    A class to detect cucumbers in images using a YOLO-based object detection model.
    """

    def __init__(self, model_path) -> None:
        """
        Initializes the CucumberDetection class with the specified YOLO model path.

        Args:
            model_path (str):
                The file path to the YOLO model weights (e.g., "models/runs/detect/train/weights/best.pt").
        """ 
        self.model_path = model_path            
        self.model = YOLO(self.model_path)

    # Placeholder detection function (replace with actual model)
    def detect_cucumbers(self, image) -> list:
        """
        Detects cucumbers in the provided image using the YOLO model.

        Args:
            image (Any):
                The input image to be processed by the YOLO model.

        Returns:
            list:
                 A list of detection results,
                 where each result contains bounding boxes and confidence scores.
        """
        return self.model(image)
    
    def extract_model_result_information(self, results: list) -> List[Dict]:
        """
        Extracts detailed information from the YOLO detection results.

        Args:
            results (list):
                The raw detection results returned by the YOLO model.

        Returns:
            List[Dict]
                A list of dictionaries where each dictionary contains:
                - 'bonding_box': The coordinates of the bounding box (top-left and bottom-right corners).
                - 'confidence': The confidence score for the detection.
                - 'label': The class label for the detected object.
        """
        objects_detected = []
        for result in results:
            for box in result.boxes:
                object_details = defaultdict()
                # Extract bounding box coordinates, detection and confidence
                object_details['bonding_box'] = map(int, box.xyxy[0])  # Get top-left and bottom-right corners
                object_details['confidence'] = box.conf[0]  # Confidence score for the detection
                object_details['label'] = self.model.names[int(box.cls[0])]  # Class label
                objects_detected.append(object_details)
        return objects_detected
    
    def draw_detections(self, original_img, detection_result):
        """
        Draws bounding boxes, class labels, and confidence scores on the original image.

        Args:
            original_img (Any):
                The input image on which the detections will be drawn.

            detection_result (List[Dict]):
                The list of detected objects with their bounding boxes, confidence scores, and labels.

        Returns:
            Any: The image with the drawn detections.
        """
        for result in detection_result:
            # Extract detection information for object
            x1, y1, x2, y2 = result['bonding_box']
            label = result['label']
            confidence = result['confidence']

            # Draw the bounding box on the image
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label and confidence score on the box
            cv2.putText(
                original_img,
                f"{label}: {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )
        return original_img

    
