import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import download_audio, add_metadata, _format_duration


def test_download():
    url = input("Введите ссылку (YouTube/SoundCloud): ").strip()
    
    if not url:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        print(f"Использую тестовую ссылку: {url}")
    
    print("\nСкачиваю...")
    filename, metadata, temp_dir = download_audio(url, for_analysis=False)
    
    if not filename:
        print("Ошибка скачивания")
        return
    
    print(f"\nФайл: {filename}")
    print(f"Метаданные: {metadata}")
    
    print("\nДобавляю ID3 теги...")
    add_metadata(filename, metadata["title"], metadata["channel"])
    
    print(f"\nГотово! Проверь файл: {filename}")
    print(f"Папка: {temp_dir}")
    
    input("\nНажми Enter чтобы удалить файл...")


if __name__ == "__main__":
    test_download()
