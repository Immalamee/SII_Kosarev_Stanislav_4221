from PIL import Image, ImageOps
import numpy as np

def preprocess_image(path, target_size=(28,28)):
    """
    1) Загружает картинку (любого формата).
    2) Конвертирует в grayscale.
    3) Делаем инверсию при необходимости.
    4) Обрезаем по не-пустой области, масштабируем и вписываем в квадрат, затем resize до 28x28.
    Возвращает numpy array shape (28,28,1) с нормализацией [0,1].
    """
    img = Image.open(path).convert('L')  
    # Если фон светлый и цифра тёмная — инвертируем.
    arr = np.array(img)
    # Оцениваем средний цвет: если среднее < 128, значит фон тёмный -> не инвертируем
    if arr.mean() > 127:
        img = ImageOps.invert(img)
    bbox = Image.fromarray(np.array(img)).getbbox()
    if bbox:
        img = img.crop(bbox)
    max_side = max(img.size)
    new_img = Image.new('L', (max_side, max_side), color=0)  
    paste_pos = ((max_side - img.size[0]) // 2, (max_side - img.size[1]) // 2)
    new_img.paste(img, paste_pos)
    new_img = new_img.resize(target_size, Image.ANTIALIAS)
    arr = np.array(new_img).astype('float32') / 255.0
    arr = np.expand_dims(arr, -1)
    return arr
