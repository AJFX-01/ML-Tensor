import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# Sequenntial API (Very convenient not very flexible

# model = keras.Sequential(
#     [
#         keras.Input(shape=(28*28)),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(10),
#     ]

#
# model = keras.Sequential()
# model.add(keras.Input(shape=(28*28)))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(256, activation='relu', name='third_layer'))
# model.add(layers.Dense(10))

# Debug or Examine a specific layer
# model = keras.Model(inputs=model.inputs, outputs=[model.layers[-2].output]) or
# model = keras.Model(inputs=model.input, outputs=[model.get_layer('third_layer').output])

# Get the layer using for loop comprehension
# model = keras.Model(inputs=model.input, outputs=[layer.output for layer in model.layers])
#
# features = model.predict(x_train)
# [print(feature.shape) for feature in features]
# print(feature.shape)

# functional API (more flexible)
inputs = keras.Input(shape=784)
x = layers.Dense(512, activation="relu", name='first_layer')(inputs)
x = layers.Dense(256, activation="relu", name='second_layer')(x)
x1 = layers.Dense(256, activation="relu", name='second2_layer')(x)
outputs = layers.Dense(10, activation='softmax')(x1)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
