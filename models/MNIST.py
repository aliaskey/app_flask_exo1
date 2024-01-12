import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models

# Chemin du répertoire contenant les images
data_dir = '../digits'
print(os.path.abspath(data_dir))


images = []
labels = []

# Parcourir le répertoire
for filename in os.listdir(data_dir):
    if filename.endswith('.png'):
        img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        images.append(img)
        label_part = filename.split('_')[1]  # Extrait 'x.png' de 'digits_x.png'
        label = int(label_part.split('.')[0])  # Extrait 'x' de 'x.png'
        labels.append(label)
    else:
        print(f"Nom de fichier inattendu: {filename}")

images = np.array(images)
labels = np.array(labels)

# Normalisation des images
images = images / 255.0

# Création du modèle
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(images, labels, epochs=10)

# Sauvegarde du modèle
model.save('handwritten.model')



###########################################################################################################################
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf

# # Charger le modèle préalablement entraîné
# # model = tf.keras.models.load_model('handwritten.model')
# model = tf.keras.models.load_model('models/handwritten.model')

# image_number = 1
# while os.path.isfile(f"digits/digits{image_number}.png"):
#     try:
#         img = cv2.imread(f"digits/digits{image_number}.png")[:, :, 0]
#         img = np.invert(np.array([img]))
#         prediction = model.predict(img)
#         predicted_digit = np.argmax(prediction)
#         print(f"L'image {image_number} est probablement le chiffre : {predicted_digit}")
#         plt.imshow(img[0], cmap=plt.cm.binary)
#         plt.show()
#     except Exception as e:
#         print(f"Erreur lors de l'analyse de l'image {image_number} : {str(e)}")
#     finally:
#         image_number += 1

#####################################################################################################
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf

# # mnist = tf.keras.datasets.mnist
# # (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # x_train = tf.keras.utils.normalize(x_train, axis=1)
# # x_test = tf.keras.utils.normalize(x_test, axis=1)

# # model = tf.keras.models.Sequential()
# # model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# # model.add(tf.keras.layers.Dense(128, activation='relu'))
# # model.add(tf.keras.layers.Dense(128, activation='relu'))
# # model.add(tf.keras.layers.Dense(10, activation='softmax'))

# # model.compile(optimizer='adam', loss='sparse_categorical_crossntropy', metrics=['accurrcy'])

# # model.fit(x_train, y_train, epochs=3)

# # model.save('handwritten.model')

# model = tf.keras.models.load_model('handwritten.model')

# # loss, accuracy = model.evaluation(x_test, y_test)

# # print(loss)
# # print(accuracy)
# image_number = 1
# while os.path.isfile(f"digits/digits{image_number}.png"):
#     try :
#         img : cv2.imread(f"digits/digits{image_number}.png")[:,:,0]
#         img = np.invert(np.array([img]))
#         predict = model.predict(img)
#         print("This digit is probably a {np.argmax(prediction)}")
#         plt.imshow(img[0], cmap=plt.cm.binary)
#         plt.show()
        
#     except:
#         print("error ! ")
#     finally:
#         image_number += 1
        
        