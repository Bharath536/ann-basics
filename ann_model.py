import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. Load dataset
iris = load_iris()

X = iris.data
y = iris.target


# 2. Convert to binary classification
# Classes 0 and 1 -> 0
# Class 2 -> 1
y = (y == 2).astype(int)


# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. Build ANN
model = Sequential([
    Dense(10, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(1, activation="sigmoid")
])


# 6. Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# 7. Train model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)


# 8. Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nANN Test Accuracy:", round(accuracy, 4))
print("ANN Test Loss:", round(loss, 4))


# 9. Predictions
y_probability = model.predict(X_test, verbose=0)
y_pred = (y_probability >= 0.5).astype(int).flatten()


# 10. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 11. Save model
model.save("ann_model.keras")


# 12. Plot training loss
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN Training and Validation Loss")
plt.legend()

plt.savefig("training_loss.png")
plt.close()

print("\nANN model saved successfully!")
print("Training loss graph saved successfully!")