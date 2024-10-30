import cv2
from ultralytics import YOLO

def load_model(model_path: str):
    """
    Загружает модель YOLO из указанного пути.

    Args:
        model_path (str): Путь к файлу модели.
    Returns:
        YOLO: Загруженная модель YOLO.
    """
    return YOLO(model_path)

def detect_people(video_path: str, model, output_path: str = 'output_crowd_detection.mp4'):
    """
    Выполняет детекцию людей на видео и сохраняет результат.

    Args:
        video_path (str): Путь к входному видеофайлу.
        model (YOLO): Предобученная модель YOLO для детекции.
        output_path (str, optional): Путь для сохранения обработанного видео. По умолчанию 'output_crowd_detection.mp4'.
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 0:  # Класс "человек"
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'Person: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow('Crowd Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = load_model("yolo11n.pt")
    detect_people("crowd.mp4", model)
