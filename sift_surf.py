from image import is_image
import os, cv2

def features(path, algorithm):
    if not os.path.exists(path) or not is_image(path):
        return
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kp, des = algorithm.detectAndCompute(img, None)
    return des