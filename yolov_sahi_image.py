import os
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.ultralytics import download_yolo11n_model

from ultralytics.utils.plotting import Annotator, colors


class SAHIInference:
    def __init__(self):
        self.model = None

    def load_model(self, weights="yolo11n.pt"):
        # Check if the model file exists and load it
        download_yolo11n_model(weights)
        try:
            self.model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=weights, device="cpu"
            )
        except Exception as e:
            print("Error loading model:", e)
            raise

    def run(self, input_folder="aa", output_folder="run"):
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Load the model
        self.load_model()

        # Get all image files from the input folder
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not image_files:
            print(f"No images found in folder: {input_folder}")
            return

        for image_file in image_files:
            # Read the image
            image_path = os.path.join(input_folder, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Failed to read image: {image_path}")
                continue

            # Perform sliced prediction
            results = get_sliced_prediction(
                frame[..., ::-1],  # Convert BGR to RGB
                self.model,
                slice_height=512,
                slice_width=512,
            )

            # Extract detection data from results
            detection_data = [
                (
                    det.category.name,
                    det.category.id,
                    (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy),
                    det.score.value,
                )
                for det in results.object_prediction_list
                if det.score.value > 0.5  # Only keep detections with confidence > 0.5
            ]

            # Initialize annotator for plotting detection results
            annotator = Annotator(frame)

            # Annotate frame with detection results
            for det in detection_data:
                label = f"{det[0]} {det[3]:.2f}"  # Include class name and confidence score
                annotator.box_label(det[2], label=label, color=colors(int(det[1]), True))

            # Save the result image
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, frame)
            print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    inference = SAHIInference()
    inference.run(input_folder="../image_video/image", output_folder="../image_video/run")