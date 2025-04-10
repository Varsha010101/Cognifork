import cv2
from ultralytics import YOLO
import time
import os
from pathlib import Path


class VehicleDetector:
    def __init__(self):
        # Load the YOLOv8 model
        self.model = YOLO('yolov8n')  # Adjust model for faster processing
        # Set model parameters
        self.model.overrides['conf'] = 0.40  # Higher confidence threshold
        self.model.overrides['iou'] = 0.45
        self.model.overrides['max_det'] = 1000

        # Define classes with unique colors
        self.classes = {
            0: ('person', (0, 128, 128)),  # Dark Yellow for human detection
            2: ('car', (0, 0, 128)),      # Dark Blue for car
            3: ('motorcycle', (0, 128, 0)),  # Dark Green for motorcycle
            5: ('bus', (0, 255, 255)),      # Dark Cyan for bus
            6: ('train', (128, 0, 128)),    # Dark Magenta for train
            7: ('truck', (128, 128, 0)),    # Dark Olive for truck
            8: ('boat', (128, 0, 0)),       # Dark Red for boat
            'forklift': ('forklift', (128, 165, 0))  # Dark Orange for forklift
        }

    def process_frame(self, frame):
        """Process a single frame and return the annotated frame."""
        results = self.model.predict(frame)
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class and confidence
                class_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Ensure forklift detection is separated from truck detection
                if class_id == 7 and 'forklift' in self.classes:
                    label, color = self.classes['forklift']
                elif class_id in self.classes:
                    label, color = self.classes[class_id]
                else:
                    continue  # Skip objects that aren't in our classes

                label_text = f"{label} {conf:.2f}"

                # Draw bounding box with class-specific color
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Add label
                label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated_frame

    def process_image(self, input_path, output_path):
        """Process a single image."""
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not read image {input_path}")
            return

        # Process the image
        processed_image = self.process_frame(image)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, processed_image)
        print(f"Processed image saved to {output_path}")

    def process_video(self, input_path, output_path, frame_interval=1):
        """Process video with specified frame interval."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create VideoWriter object
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps / frame_interval,
                              (frame_width, frame_height))

        frame_count = 0
        processed_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every nth frame
            if frame_count % frame_interval == 0:
                processed_count += 1

                # Process the frame
                processed_frame = self.process_frame(frame)

                # Write the frame
                out.write(processed_frame)

                # Calculate and display progress
                elapsed_time = time.time() - start_time
                fps_current = processed_count / elapsed_time
                progress = (frame_count / total_frames) * 100

                print(f"Processing frame {frame_count}/{total_frames} "
                      f"({progress:.1f}%) - FPS: {fps_current:.1f}")

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video processing complete. Output saved to {output_path}")

    def process_live_camera(self):
        """Process live camera feed."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access the camera.")
            return False

        print("Press 'q' to quit the live camera feed.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frame
                processed_frame = self.process_frame(frame)

                # Display the processed frame
                cv2.imshow("Live Vehicle Detection", processed_frame)

                # Exit loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return True


def process_all_images(input_folder, output_folder):
    """Process all images in the input folder and save in the output folder."""
    detector = VehicleDetector()

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # Process each file in the input folder
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # Check if the file is an image
        if Path(input_path).suffix.lower() in image_extensions:
            print(f"Processing image: {input_path}")
            detector.process_image(input_path, output_path)
        else:
            print(f"Skipped unsupported file: {input_path}")


def main():
    # Choose processing mode
    mode = "live_camera"  # Options: 'live_camera', 'image', 'video', 'all_images'

    detector = VehicleDetector()

    if mode == "live_camera":
        detector.process_live_camera()
    elif mode == "image":
        input_image = './Datasets/img.png'
        output_image = './results/processed_image.jpg'
        detector.process_image(input_image, output_image)
    elif mode == "video":
        input_video = './Datasets/sample_video.mp4'
        output_video = './results/processed_video.avi'
        detector.process_video(input_video, output_video)
    elif mode == "all_images":
        input_folder = './dataset'
        output_folder = './result1'
        process_all_images(input_folder, output_folder)


if __name__ == "__main__":
    main()
