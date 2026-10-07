import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense


# 1. Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# 2. Convert into binary classification
y = (y == 2).astype(int)


# 3. Split dataset
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


# 5. Different hidden neuron counts
neuron_counts = [5, 10, 20, 50]

results = []


# 6. Train ANN for each neuron count
for neurons in neuron_counts:

    print(f"\nTraining ANN with {neurons} hidden neurons...")

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(neurons, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=16,
        verbose=0
    )

    # Predictions
    y_probability = model.predict(X_test, verbose=0)
    y_pred = (y_probability >= 0.5).astype(int).flatten()

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    final_loss = history.history["loss"][-1]

    results.append({
        "Hidden Neurons": neurons,
        "Accuracy": accuracy,
        "Final Training Loss": final_loss
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Final Training Loss: {final_loss:.4f}")


# 7. Create results DataFrame
results_df = pd.DataFrame(results)

print("\nNeuron Experiment Results:")
print(results_df)


# 8. Save results
results_df.to_csv(
    "ann_neuron_comparison.csv",
    index=False
)


# 9. Plot accuracy comparison
plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Hidden Neurons"],
    results_df["Accuracy"],
    marker="o"
)

plt.xlabel("Number of Hidden Neurons")
plt.ylabel("Test Accuracy")
plt.title("ANN Performance vs Hidden Neurons")
plt.xticks(neuron_counts)
plt.grid(True)

plt.savefig("ann_neuron_comparison.png")
plt.close()


print("\nResults saved to ann_neuron_comparison.csv")
print("Comparison graph saved to ann_neuron_comparison.png")