import os
import cv2
import numpy as np

def capture_photo():
    os.system('cls' if os.name == 'nt' else 'clear')
    cap = cv2.VideoCapture(0)  # Використання камери за замовчуванням
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Captured Photo', frame)
        cv2.waitKey(0)  # Очікування натискання будь-якої клавіші
        # Опціонально зберегти захоплене фото
        save_option = input("Save this photo? (y/n): ").lower()
        if save_option == 'y':
            file_path = input("Enter the path to save the photo: ")
            cv2.imwrite(file_path, frame)
            print("Photo saved successfully.")
    else:
        print("Failed to capture photo from camera.")
    cap.release()
    cv2.destroyAllWindows()


def load_photo(file_path):
    """ Завантажує фото з файлу. """
    return cv2.imread(file_path)  # Завантаження фото за вказаним шляхом

def save_photo(image, save_path):
    """ Зберігає фото у файл. """
    cv2.imwrite(save_path, image)  # Збереження фото за вказаним шляхом

def process_photo(file_path):
    os.system('cls' if os.name == 'nt' else 'clear')
    """ Обробляє фото та відображає його на екрані. """
    image = load_photo(file_path)  # Завантаження фото
    operation = choose_operation()  # Вибір операції для обробки фото
    processed_image = apply_operations(image, operation)  # Застосування операцій до фото
    cv2.imshow('Processed Photo', processed_image)  # Відображення обробленого фото
    cv2.waitKey(0)  # Очікування натискання будь-якої клавіші
    save_option = input("Save processed photo? (y/n): ").lower()  # Запит на збереження
    if save_option == 'y':
        save_path = input("Enter the path to save the processed photo: ")
        save_photo(processed_image, save_path)
        print("Processed photo saved successfully.")
    cv2.destroyAllWindows()  # Закриття всіх вікон OpenCV

def choose_operation():
    os.system('cls' if os.name == 'nt' else 'clear')
    """ Вибір операції для обробки фото. """
    print("\nChoose an operation:")
    print("1. Horizontal Shift")
    print("2. Edge Detection with Sobel Thresholds")
    print("3. YUV Conversion with Contrast")
    print("4. Low Pass Filter with Various Masks")
    operation = int(input("Enter operation number: "))
    return operation

def apply_operations(image, operation):
    """ Застосовує обрані операції до фото. """
    if operation == 1:
        return horizontal_shift(image)
    elif operation == 2:
        return edge_detection_sobel(image)
    elif operation == 3:
        return yuv_conversion_contrast(image)
    elif operation == 4:
        return low_pass_filter(image)

def horizontal_shift(image):
    """ Здійснює горизонтальний зсув зображення на вказану величину. """
    num_rows, num_cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, 100], [0, 1, 0]])  # Приклад зсуву на 100 пікселів
    return cv2.warpAffine(image, translation_matrix, (num_cols, num_rows))

def edge_detection_sobel(image):
    """ Виконує виявлення країв за допомогою оператора Собеля. """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    edges = np.uint8(np.absolute(edges))
    ret, edges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)  # Поріг для виявлення країв
    return edges

def yuv_conversion_contrast(image):
    """ Конвертує зображення в простір YUV та регулює контраст. """
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

def low_pass_filter(image):
    """ Застосовує низькочастотний фільтр до зображення. """
    kernel = np.ones((5, 5), np.float32) / 25  # Розмір ядра та значення фільтра
    return cv2.filter2D(image, -1, kernel)