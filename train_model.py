import tensorflow as tf

# todo: create a ui for user to test their own handwriting

# loading and preprocessing data
data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# reshaping data for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# defining model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compiling model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training with early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[early_stop])

# saving the model
model.save('digit-recognition.keras')