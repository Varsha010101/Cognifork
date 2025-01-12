import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

class ObjectClassifier:
    def __init__(self, model_path='yolov8n.pt'):
        # Initialize the YOLO model with the specified model path
        self.model = YOLO(model_path)
        self.model.overrides['conf'] = 0.3  # Confidence threshold
        self.model.overrides['iou'] = 0.5  # IoU threshold for NMS

        # Class label definitions with unique colors for bounding boxes
        self.classes = {
            0: ('person', (64, 128, 128)),
            2: ('car', (0, 0, 255)),
            3: ('motorcycle', (0, 255, 0)),
            5: ('bus', (255, 165, 0)),
            6: ('train', (128, 0, 128)),
            7: ('truck', (255, 0, 0)),
            8: ('forklift', (0, 255, 255))
        }

    def annotate_frame(self, frame, box, class_id):
        """Annotate the frame with bounding boxes and class labels."""
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        label, color = self.classes.get(class_id, ('Unknown', (0, 255, 0)))
        confidence = box.conf[0]  # Confidence score

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Add label and confidence above the box
        label_text = f"{label} {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def process_frame(self, frame):
        """Process a single frame to perform object detection and annotation."""
        results = self.model(frame)  # YOLO detection
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])  # Get the detected class ID
                if class_id in self.classes:
                    self.annotate_frame(frame, box, class_id)  # Annotate the frame for recognized classes
        return frame

    def process_live_camera(self):
        """Process live camera feed."""
        cap = cv2.VideoCapture(0)  # Open the first camera device

        if not cap.isOpened():
            print("Error: Could not access the camera.")
            return False

        print("Press 'q' to quit the live camera feed.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow("Live Classification", processed_frame)

                # Exit loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return True

    def process_image(self, input_path, output_path):
        """Process a single image."""
        if not os.path.exists(input_path):
            print(f"Error: Image file {input_path} not found.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image = cv2.imread(input_path)
        processed_image = self.process_frame(image)
        cv2.imwrite(output_path, processed_image)
        print(f"Processed image saved at: {output_path}")

    def process_video(self, input_path, output_path):
        """Process a video."""
        if not os.path.exists(input_path):
            print(f"Error: Video file {input_path} not found.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            out.write(processed_frame)

        cap.release()
        out.release()
        print(f"Processed video saved at: {output_path}")


def main():
    classifier = ObjectClassifier()

    # Choose processing mode: live camera, image, or video
    mode = "image"  # Change to "live_camera" or "video" as needed

    if mode == "live_camera":
        classifier.process_live_camera()
    elif mode == "image":
        input_image = './Datasets/img.png'
        output_image = './results/processed_image.jpg'
        classifier.process_image(input_image, output_image)
    elif mode == "video":
        input_video = './Datasets/sample_video.mp4'
        output_video = './results/processed_video.avi'
        classifier.process_video(input_video, output_video)


if __name__ == "__main__":
    main()
