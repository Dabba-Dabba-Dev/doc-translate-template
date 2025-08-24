from pdf2image import convert_from_path
from PIL import Image

def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

def save_image(image, path):
    image.save(path)

def load_image(path):
    return Image.open(path).convert("RGB") 