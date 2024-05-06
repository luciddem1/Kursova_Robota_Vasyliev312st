import cv2
import numpy as np
import os

def process_video(file_path: str):
    # Відкриття відеофайлу за вказаним шляхом
    cap = cv2.VideoCapture(file_path)
    # Перевірка, чи вдалося відкрити відеофайл
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Вибір операції для відеозапису
    operation = choose_operation()

    # Отримання параметрів відео для вихідного файлу
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Визначення кодеку та створення об'єкта для запису відео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Використовуйте 'mp4v' для MP4 з H.264
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    # Перевірка, чи вдалося відкрити вихідний файл для запису
    if not out.isOpened():
        print("Error opening video writer")
        cap.release()
        return

    # Створення вікна для відображення обробленого відео
    cv2.namedWindow('Processed Video', cv2.WINDOW_NORMAL)

    # Цикл обробки відео
    while cap.isOpened():
        # Зчитування кадру з відео
        ret, frame = cap.read()
        # Перевірка, чи вдалося зчитати кадр
        if not ret:
            break
        # Застосування операції до кадру
        processed_frame = apply_operation(frame, operation)
        # Відображення обробленого кадру
        cv2.imshow('Processed Video', processed_frame)
        # Запис обробленого кадру у вихідний файл
        out.write(processed_frame)
        # Очікування натискання клавіші 'q' для виходу
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Звільнення ресурсів
    cap.release()
    out.release()
    # Закриття всіх відкритих вікон
    cv2.destroyAllWindows()

def capture_video():
    # Створення вікна для відображення захопленого відео
    cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
    # Відкриття потоку відеозахоплення (за замовчуванням - перша доступна камера)
    cap = cv2.VideoCapture(0)

    # Перевірка, чи вдалося відкрити потік відеозахоплення
    if not cap.isOpened():
        print("Error opening video capture")
        cap.release()
        return

    # Отримання параметрів відеопотоку для вихідного файлу
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Визначення кодеку та створення об'єкта для запису відео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    # Вибір операції для відеозапису
    operation = choose_operation()

    # Цикл захоплення та обробки відео
    while cap.isOpened():
        # Зчитування кадру
        ret, frame = cap.read()
        # Перевірка, чи вдалося зчитати кадр
        if not ret:
            print("Error reading frame, skipping...")
            continue
        # Застосування операції до кадру
        processed_frame = apply_operation(frame, operation)
        # Відображення кадру
        cv2.imshow('Video', processed_frame)
        # Запис кадру у вихідний файл
        out.write(processed_frame)
        # Очікування натискання клавіші 'q' для виходу
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Звільнення ресурсів
    cap.release()
    out.release()
    # Закриття вікна
    cv2.destroyAllWindows()

def apply_effects(frame):
    """ Застосовує ефекти до кадру відео. """
    operation = choose_operation()  # Вибір операції для обробки відео
    processed_frame = apply_operation(frame, operation)  # Застосування вибраної операції до кадру
    return processed_frame

def choose_operation():
    """ Вибір операції для обробки відео. """
    print("\nChoose an operation:")
    print("1. Horizontal Shift")
    print("2. Edge Detection with Sobel Thresholds")
    print("3. YUV Conversion with Contrast")
    print("4. Low Pass Filter with Various Masks")
    operation = int(input("Enter operation number: "))
    return operation

def apply_operation(frame, operation):
    """ Застосовує обрану операцію до кадру. """
    if operation == 1:
        return horizontal_shift(frame, shift_value=50)
    elif operation == 2:
        return edge_detection_sobel(frame, threshold=100)
    elif operation == 3:
        return convert_yuv_contrast(frame)
    elif operation == 4:
        return apply_low_pass_filter(frame, kernel_size=5)
    else:
        return frame

def horizontal_shift(frame, shift_value):
    """ Здійснює горизонтальний зсув кадру на вказану величину. """
    num_rows, num_cols = frame.shape[:2]
    translation_matrix = np.float32([[1, 0, shift_value], [0, 1, 0]])
    return cv2.warpAffine(frame, translation_matrix, (num_cols, num_rows))

def edge_detection_sobel(frame, threshold):
    """ Виконує виявлення країв за допомогою оператора Собеля. """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    edges = np.uint8(np.absolute(edges))
    ret, edges = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def convert_yuv_contrast(frame):
    """ Конвертує кадр в простір YUV та регулює контраст. """
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv_frame[:, :, 0] = cv2.equalizeHist(yuv_frame[:, :, 0])
    return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)

def apply_low_pass_filter(frame, kernel_size):
    """ Застосовує низькочастотний фільтр до кадру. """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(frame, -1, kernel)

if __name__ == "__main__":
    file_type = int(input("Choose file type (1: Video file, 2: Capture from camera): "))
    if file_type == 1:
        file_path = input("Enter the path of the video file: ")
        process_video(file_path)
    elif file_type == 2:
        capture_video()