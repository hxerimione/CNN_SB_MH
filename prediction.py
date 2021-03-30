import numpy as np

import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator,load_img ,img_to_array,array_to_img
import sklearn.metrics as metrics
from keras.models import load_model
model = load_model('/content/drive/MyDrive/model-85')
testGenerator = ImageDataGenerator()
root_path='/content/drive/MyDrive'
testGen=testGenerator.flow_from_directory(os.path.join(root_path,'data2'),
                                           target_size=(200,200),
                                            )
cls_index = ['민혁','수빈']

imgs = testGen.next()
arr = imgs[0][0]
img = array_to_img(arr).resize((200, 200))
plt.imshow(img)
result = model.predict_classes(arr.reshape(1, 200, 200, 3))
print('예측: {}'.format(cls_index[result[0]]))
print('정답: {}'.format(cls_index[np.argmax(imgs[1][0])]))
