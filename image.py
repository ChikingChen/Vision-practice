from PIL import Image

def is_image(path):
    try:
        with Image.open(path) as _:
            return True
    except:
        return False