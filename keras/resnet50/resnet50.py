from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import datetime

framework = 'keras'
model = ResNet50(weights='imagenet')
csv = open('csv.txt', 'w')


def work(folder):
    for x in os.walk(folder):
        try:
            directory = x[0]
            last_directory_name = os.path.basename(os.path.normpath(directory))
            files = x[2]
            for f in files:
                predict(os.path.join(directory, f), last_directory_name)
        except:
            continue


def predict(img_path, ground_truth):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    start = datetime.datetime.now()
    preds = model.predict(x)
    end = datetime.datetime.now()
    difference = end - start
    for predict in decode_predictions(preds, top=5)[0]:
        predicted_class = predict[0]
        predicted_procent = predict[2]
        s = ','.join(map(str, [predicted_class, predicted_procent, img_path, ground_truth, framework, difference, 'resnet50']))
        csv.write(s)
        csv.write('\n')

if __name__ == "__main__":
    work('../result')
    csv.close()
