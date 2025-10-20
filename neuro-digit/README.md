# Handwritten-Digit-Classifier

Простой проект: сверточная нейронная сеть (Keras/TensorFlow), распознающая цифры 0–9.
Обучение на MNIST, скрипт для предсказания изображений, нарисованных в Paint.

## Файлы
- `train_model.py` — обучает и сохраняет `model.h5`.
- `preprocess_paint.py` — функции предобработки изображений из Paint.
- `predict.py` — пример предсказания для одного файла.
- `demo_paint_examples/` — папка для примеров от руки.

## Быстрый старт (локально)
1. Создать виртуальное окружение и установить зависимости:

python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt

2. Обучение (MNIST сам скачается)

python train_model.py

3. Ну и сделать предсказание

python predict.py demo_paint_examples/1.png