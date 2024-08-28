import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to the range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Make predictions
predictions = model.predict(x_test)
 
# Function to plot the image, predicted label, and confidence bar in a horizontal layout
def plot_image_and_prediction(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    predicted_label = np.argmax(predictions_array)
    
    # Plot the image and the confidence bar in a smaller layout
    plt.subplot(2, num_images, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"Predicted: {predicted_label}, True: {true_label}", color=color)
    # Plot the confidence bar
    plt.subplot(2, num_images, num_images+i+1)
    plt.grid(False)
    plt.xticks(range(10), fontsize=8)
    plt.yticks([])
    bar_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    bar_plot[predicted_label].set_color('red')
    bar_plot[true_label].set_color('blue')

# Display 10 test images horizontally
num_images = 10
indices = np.random.choice(len(x_test), num_images, replace=False)

plt.figure(figsize=(num_images * 2, 4))  # Adjust the figure size based on the number of images

for idx, i in enumerate(indices):
    plot_image_and_prediction(idx, predictions, y_test, x_test)

plt.tight_layout()
plt.show()

# Model summary
print("\nModel Summary:")
model.summary()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, np.argmax(predictions, axis=1))

plt.figure(figsize=(10,10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()