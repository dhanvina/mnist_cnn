import tensorflow as tf
import numpy as np
import os
from datetime import datetime

def train_and_save_model():
    # Create models directory if it doesn't exist
    if not os.path.exists('../models'):
        os.makedirs('../models')

    # Load MNIST dataset
    print("Loading MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize and reshape data
    print("Preprocessing data...")
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    x_trainer = np.array(x_train).reshape(-1, 28, 28, 1)
    x_tester = np.array(x_test).reshape(-1, 28, 28, 1)
    
    # Build model
    print("Building model...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Create checkpoints
    checkpoint_path = "../models/checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1,
        save_weights_only=True,
        period=5)
    
    # Train model
    print("Training model...")
    history = model.fit(x_trainer, y_train,
                       epochs=10,
                       validation_split=0.3,
                       batch_size=128,
                       verbose=1,
                       callbacks=[cp_callback])
    
    # Save model
    print("Saving model...")
    model.save('../models/mnist_cnn_model.h5')
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(x_tester, y_test)
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    # Save model info
    with open('../models/model_info.txt', 'w') as f:
        f.write(f"Model trained on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write("\nModel architecture:\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))

if __name__ == "__main__":
    train_and_save_model()