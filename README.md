# Проект по детекции людей на видео с использованием YOLO

## Описание

Данный проект выполняет детекцию людей на видеофайле `crowd.mp4` и сохраняет результат с обведенными рамками вокруг людей. Используется модель yolo11n для детекции объектов.

## Структура проекта
```
people_detection_project/
│
├── src/
│   ├── main.py                # Главный файл программы
│   └── crowd.mp4              # Видео с людьми
├── README.md                  # Описание проекта, установка и запуск
├── results.docx               # Выводы по результатам работы программы
└── requirements.txt           # Зависимости проекта
```
## Установка

1. Клонируйте репозиторий и перейдите в папку проекта:
   git clone https://github.com/nikita202031/people_detection.git
   
   cd people_detection
   
3. Установите зависимости из файла requirements.txt:
  pip install -r requirements.txt
  
### Запуск

1. Убедитесь, что файл модели YOLO (`yolo11n.pt`) находится в папке проекта, а видео `crowd.mp4` — в папке `src/`.
2. Запустите программу командой:
   python src/main.py

## Примечания

1. Для корректной работы, убедитесь, что у вас установлены все зависимости, указанные в requirements.txt.
2. Видео output_crowd_detection.mp4 будет сохранено в корневую папку проекта.



   


