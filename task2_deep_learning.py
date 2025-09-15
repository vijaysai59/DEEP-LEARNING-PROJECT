#!/usr/bin/env python3
"""
task2_deep_learning.py
Run this in IDLE (F5) after installing:
    pip install tensorflow matplotlib scikit-learn
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report

# ------------------  Data loader  ------------------
def load_dataset(name="cifar10"):
    if name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        num_classes = 10
    elif name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        num_classes = 10
    else:
        raise ValueError("Choose cifar10 or mnist")
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    y_train = y_train.reshape(-1)
    y_test  = y_test.reshape(-1)
    return x_train, y_train, x_test, y_test, x_train.shape[1:], num_classes

# ------------------  Model  ------------------
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

# ------------------  Plot helpers  ------------------
def plot_history(history, outdir):
    # Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history.get("val_accuracy", []), label="val")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.savefig(os.path.join(outdir, "acc.png"))
    plt.show()
    # Loss
    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history.get("val_loss", []), label="val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(os.path.join(outdir, "loss.png"))
    plt.show()

# ------------------  Main flow  ------------------
def main(dataset="cifar10", epochs=25, batch_size=64):
    os.makedirs("outputs", exist_ok=True)
    x_train, y_train, x_test, y_test, input_shape, n_classes = load_dataset(dataset)

    model = build_cnn(input_shape, n_classes)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    cbs = [
        callbacks.ModelCheckpoint("outputs/best_model.h5",
                                  save_best_only=True,
                                  monitor="val_accuracy"),
        callbacks.EarlyStopping(monitor="val_accuracy",
                                patience=6,
                                restore_best_weights=True)
    ]

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=cbs,
        verbose=2
    )

    model.save("outputs/final_model.h5")
    np.save("outputs/history.npy", history.history)
    plot_history(history, "outputs")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(x_test), axis=1)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # For quick IDLE use, just run F5. Edit these defaults if desired:
    main(dataset="cifar10", epochs=25, batch_size=64)
