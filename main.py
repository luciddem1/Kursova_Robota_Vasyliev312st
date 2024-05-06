import os
import photo
import video

def main():
    while True:
        file_type = choose_file_type()
        if file_type == 1:
            file_path = input("Enter the path of the video file: ")
            video.process_video(file_path)
        elif file_type == 2:
            video.capture_video()
        elif file_type == 3:
            file_path = input("Enter the path of the photo file: ")
            photo.process_photo(file_path)
        elif file_type == 4:
            photo.capture_photo()
        elif file_type == 5:
            break

def choose_file_type() -> int:
    """ Вибір типу файлу для обробки. """
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("==== CHOOSE TYPE OF FILE ====")
        print("1. Video File")
        print("2. Capture from Camera (Video)")
        print("3. Photo File")
        print("4. Capture from Camera (Photo)")
        print("5. Exit")
        user_input = input("\nInput number: ")
        try:
            file_type = int(user_input)
            if file_type in [1, 2, 3, 4, 5]:
                return file_type
        except ValueError:
            pass

if __name__ == "__main__":
    main()