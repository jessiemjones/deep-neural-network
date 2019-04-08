import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


#hello world of ML.  Dataset of 28x28 images of hand-written digits 0-9
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Scaling or normalizing the data.
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#building model. Creating deep neural network.
model = tf.keras.models.Sequential()
#Flatten Input layer
model.add(tf.keras.layers.Flatten())
#Add dense hidden layers. 128 Neurons, and then activation funciton.  tf.nn.relu is the default/goto function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#output layer. Has the number of classfications we are doing (10) Use softmax because its good for probability distribution.
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


#Parameters for training the model.
#Use Adaptive Moment Estimation as a default because it is fast than most other gradient decent optimziers.  Espcially on smaller datasets.  Adam has a very good time effiency.
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#fitting the model. 
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


#tesnor is a multidemenional array
plt.imshow(x_train[0])
plt.show()
print(x_train[0])

model.save('num_reader.model')
new_model = tf.keras.models.load_model('num_reader.model')
predictions = new_model.predict([x_test])

print(predictions)
print(np.argmax(predictions[0]))

