import os
import keras
from keras import layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

def build_fashion_mnist_cnn() -> keras.Sequential:
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

def train_and_eval_prototype():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    model = build_fashion_mnist_cnn()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFashion-MNIST Prototype Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    return model

if __name__ == "__main__":
    train_and_eval_prototype()

