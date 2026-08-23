import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ==========================================
# 1. Load MNIST Dataset
# ==========================================

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", X_test.shape)
print("Testing labels:", y_test.shape)


# ==========================================
# 2. Display Images
# ==========================================

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Digit: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# 3. Calculate Basic Statistics
# ==========================================

pixel_values = X_train.flatten()

mean_pixel = np.mean(pixel_values)
median_pixel = np.median(pixel_values)
std_pixel = np.std(pixel_values)

print("Mean Pixel Value:", mean_pixel)
print("Median Pixel Value:", median_pixel)
print("Standard Deviation:", std_pixel)


# ==========================================
# 4. Create Pandas DataFrame
# ==========================================

df = pd.DataFrame(
    X_train[:10000].reshape(10000, -1)
)

print("\nDataFrame Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ==========================================
# 5. Histogram
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(pixel_values, bins=50)

plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Distribution of MNIST Pixel Values")

plt.show()


# ==========================================
# 6. Normalize Data
# ==========================================

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# ==========================================
# 7. Reshape for CNN
# ==========================================

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# ==========================================
# 8. Build CNN
# ==========================================

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    MaxPooling2D((2, 2)),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),

    Dense(10, activation="softmax")
])


# ==========================================
# 9. Compile
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# 10. Train
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)


# ==========================================
# 11. Evaluate
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print("Test Accuracy:", test_accuracy)


# ==========================================
# 12. Plot Accuracy
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.show()


# ==========================================
# 13. Plot Loss
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.show()


# ==========================================
# 14. Test Prediction
# ==========================================

prediction = model.predict(X_test[0:1])

predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)
print("Actual Digit:", y_test[0])

plt.imshow(
    X_test[0].reshape(28, 28),
    cmap="gray"
)

plt.title(
    f"Predicted: {predicted_digit}, Actual: {y_test[0]}"
)

plt.axis("off")
plt.show()