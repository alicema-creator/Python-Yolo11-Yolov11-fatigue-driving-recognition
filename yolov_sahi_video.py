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

    def run(self):
        cap = cv2.VideoCapture("../image_video/test2.mp4")  # Open default camera
        assert cap.isOpened(), "Error accessing camera"

        self.load_model()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform sliced prediction
            results = get_sliced_prediction(
                frame[..., ::-1],  # Convert BGR to RGB
                self.model,
                slice_height=512,
                slice_width=512,
            )



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

            # Show the frame

            cv2.namedWindow("CameraFeed", 0);
            cv2.resizeWindow("CameraFeed", 650, 480);
            cv2.moveWindow("CameraFeed",100,100)
            cv2.imshow("CameraFeed", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    inference = SAHIInference()
    inference.run()