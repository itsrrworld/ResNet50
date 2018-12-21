from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras.applications.resnet50 import ResNet50
import numpy as np
import cv2

def object_classifier():
    
    rec = cv2.VideoCapture(0)

    while(True):
        ret, frame = rec.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        cv2.imshow('frame', rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            out = cv2.imwrite('image.jpg', frame)
            break
    rec.release()
    cv2.destroyAllWindows()
    
    model = ResNet50()

    #model.summary()

    image = load_img('image.jpg', target_size=(224, 224))

    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    image = preprocess_input(image)
    
    result = model.predict(image)

    label = decode_predictions(result)
    label = label[0][0]

    print(label[1])

object_classifier()
