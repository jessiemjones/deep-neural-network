import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#import test data
mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = tf.keras.utils.normalize(x_test, axis=1)

new_model = tf.keras.models.load_model('num_reader.model')
predictions = new_model.predict([x_test])


print(np.argmax(predictions[3]))
plt.set_cmap('binary')
plt.imshow(x_test[3])
plt.show()