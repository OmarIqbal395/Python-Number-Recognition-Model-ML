import tensorflow as tf
import matplotlib.pyplot as plt # Import matplotlib library
from pandas import np
import numpy as nm
from PIL import Image
import matplotlib.image as mpimg

mnist = tf.keras.datasets.mnist # Object of the MNIST dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # Load data

plt.imshow(x_train[0], cmap="gray")  # Import the image
plt.show() # Plot the image

# Normalize the train dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
# Normalize the test dataset
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model object
model = tf.keras.models.Sequential()
# Add the Flatten Layer
model.add(tf.keras.layers.Flatten())
# Build the input and the hidden layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# Build the output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x=x_train, y=y_train, epochs=5) # Start training process
print("The model has successfully trained")
model.save('mnist.h5')
print("Saving the model as mnist.h5")

predictions = model.predict([x_test]) # Make prediction

print(np.argmax(predictions[1000])) # Print out the number

plt.imshow(x_test[1000], cmap="gray") # Import the image
plt.show() # Show the image


print(np.argmax(predictions[1001])) # Print out the number

plt.imshow(x_test[1001], cmap="gray") # Import the image
plt.show() # Show the image


plt.imshow(x_test[999], cmap="gray") # Import the image
plt.show()  # Show the image

print(np.argmax(predictions[999])) # Print out the number

# img = mpimg.imread("test_img.png")
# imgplot = plt.imshow(img)
# img =
# plt.show()


# img = nm.invert(Image.open("test_img.png").convert('L')).ravel()
# #prediction = model.predict(img)
# #print(np.argmax(prediction))
# plt.imshow(img)
# plt.show()
