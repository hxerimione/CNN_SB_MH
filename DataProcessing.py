from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

dataGen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             brightness_range=[.2,.2],
                             horizontal_flip=True,
                             vertical_flip=True,)

img=load_img('./data/traindata/soobin/soobin01.jpg')
x=img_to_array(img)
x=x.reshape((1,)+x.shape)

i=0

for batch in dataGen.flow(x,batch_size=1,save_to_dir='./data2/soobin'
                          ,save_prefix='sb'
                          ,save_format='jpg'):
    i+=1
    if i>19:
        break

print("50")
