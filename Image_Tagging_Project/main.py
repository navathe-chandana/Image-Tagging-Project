# main.py - Image Tagging Project (CIFAR-10 CNN)

import os
import matplotlib
matplotlib.use('Agg')   # avoids blocking windows
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

print("✅ Step 0: Libraries imported")
print("TensorFlow version:", tf.__version__)

# Step 1: Load CIFAR-10 dataset
print("\n✅ Step 1: Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
print("Training samples:", x_train.shape[0], "Test samples:", x_test.shape[0])

# Step 2: Save dataset sample images
print("\n✅ Step 2: Saving dataset sample images (samples.png)")
fig = plt.figure(figsize=(5,5))
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    ax.axis('off')
    ax.imshow(x_train[i])
    ax.set_title(class_names[int(y_train[i][0])], fontsize=8)
plt.tight_layout()
plt.savefig("samples.png")
plt.close(fig)

# Step 3: Build CNN model
print("\n✅ Step 3: Building CNN model")
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Step 4: Train model
print("\n✅ Step 4: Training model (10 epochs)...")
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)

# Step 5: Evaluate model
print("\n✅ Step 5: Evaluating on test set")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Step 6: Save predictions for first 5 test images
print("\n✅ Step 6: Saving predictions for first 5 test images")
prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = prob_model.predict(x_test)
for i in range(5):
    fig = plt.figure(figsize=(2,2))
    plt.axis('off')
    plt.imshow(x_test[i])
    pred_label = class_names[int(np.argmax(predictions[i]))]
    true_label = class_names[int(y_test[i][0])]
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=8)
    plt.savefig(f"prediction_{i}.png", bbox_inches='tight', dpi=150)
    plt.close(fig)

# Step 7: Save training history plot
print("\n✅ Step 7: Saving training history graph (history.png)")
fig, axs = plt.subplots(1,2,figsize=(12,4))
axs[0].plot(history.history['accuracy'], label='train acc')
axs[0].plot(history.history['val_accuracy'], label='val acc')
axs[0].set_title('Accuracy')
axs[0].legend()

axs[1].plot(history.history['loss'], label='train loss')
axs[1].plot(history.history['val_loss'], label='val loss')
axs[1].set_title('Loss')
axs[1].legend()

plt.tight_layout()
plt.savefig("history.png")
plt.close(fig)

# Step 8: Save model
model.save("image_classifier.h5")
print("\n✅ Step 8: Model saved as image_classifier.h5")

print("\n🎉 Project completed! Check your folder for:")
print("- samples.png")
print("- prediction_0.png ... prediction_4.png")
print("- history.png")
print("- image_classifier.h5")
