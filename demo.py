import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


#hello world of ML.  Dataset of 28x28 images of hand-written digits 0-9
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Scaling or normalizing the data.  Pixels have values between 0 and 255.  It's easier to work with the data when it's scaled down to numbers between 0 and 1.  Ie: 254 ---> ~.998 & 1 ---> ~.001 quick maffs :p
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#building model. Creating deep neural network.
model = tf.keras.models.Sequential()
#Flatten Input layer
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Flatten())

#Add dense hidden layers. 128 Neurons, and then activation funciton.  tf.nn.relu is the default/goto function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#output layer. Has the number of classfications we are doing (10) Use softmax here because its good for probability distribution.
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


#Parameters for training the model.
#Use Adaptive Moment Estimation as a default because it is fast than most other gradient decent optimziers.  Espcially on smaller datasets.  Adam has a very good time effiency and is a the go-to for optimization algorithms.
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#fitting the model. Epoch is a 'full pass' through the training dataset.
model.fit(x_train, y_train, epochs=3)

#we want to calculate the validation loss and accuracy to make sure our model didn't overtrain.
#We don't want it to memorize each item... We want it to learn the general patterns..
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


#a tesnor is basically a multidemenional array
print(x_train[1])

#to save the model
model.save('num_reader.model')
#to load the model that we saved
new_model = tf.keras.models.load_model('num_reader.model')
#to make a predicion.  It TAKES  A LIST.
predictions = new_model.predict([x_test])
# predictions = model.predict([x_test])


print(predictions)
# plt.imshow(x_test[4])
# plt.set_cmap('binary')
# plt.show()
# print(np.argmax(predictions[4]))

