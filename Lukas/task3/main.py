import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import pandas as pd
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet

target_shape = (256, 256)

train_data = pd.read_csv('train_triplets.txt', sep=" ", header=None, dtype=str)
train_data.columns = ["a", "b", "c"]

cache_dir = Path(Path.home()) / "Documents\Python_Scripts\iml-projects\Lukas\\task3"
images_path = cache_dir / "food"


def preprocess_image(filename):

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(a, b, c):
    return (
        preprocess_image(a),
        preprocess_image(b),
        preprocess_image(c),
    )
#train_data = train_data[:10]
a_images, b_images, c_images = [], [], []
for index, row in train_data.iterrows():
    a_image_name = row['a'].strip() + ".jpg"
    b_image_name = row['b'].strip() + ".jpg"
    c_image_name = row['c'].strip() + ".jpg"
    a_images.append(str(images_path / a_image_name))
    b_images.append(str(images_path / b_image_name))
    c_images.append(str(images_path / c_image_name))

image_count = len(a_images)

a_dataset = tf.data.Dataset.from_tensor_slices(a_images)
b_dataset = tf.data.Dataset.from_tensor_slices(b_images)
c_dataset = tf.data.Dataset.from_tensor_slices(c_images)


dataset = tf.data.Dataset.zip((a_dataset, b_dataset, c_dataset))
#dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

train_dataset = dataset.batch(10, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)

def visualize(a, b, c):

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 3))

    axs = fig.subplots(1, 3)
    show(axs[0], a[0])
    show(axs[1], b[0])
    show(axs[2], c[0])

#visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])

base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable
    
class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the a
    embedding and the b embedding, and the a embedding and the
    c embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, a, b, c):
        ap_distance = tf.reduce_sum(tf.square(a - b), -1)
        an_distance = tf.reduce_sum(tf.square(a - c), -1)
        return (ap_distance, an_distance)


a_input = layers.Input(name="a", shape=target_shape + (3,))
b_input = layers.Input(name="b", shape=target_shape + (3,))
c_input = layers.Input(name="c", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(resnet.preprocess_input(a_input)),
    embedding(resnet.preprocess_input(b_input)),
    embedding(resnet.preprocess_input(c_input)),
)

siamese_network = Model(
    inputs=[a_input, b_input, c_input], outputs=distances
)
class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the a and the b example, and the a and
        # the c example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a cvalue.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.fit(train_dataset, epochs=1, batch_size=32) #, validation_data=val_dataset)


test_data = pd.read_csv('test_triplets.txt', sep=" ", header=None, dtype=str)
test_data.columns = ["a", "b", "c"]

#test_data = test_data[:10]
with open('siamese.txt', 'w') as f:
    
    for index, row in test_data.iterrows():
        a_image_name = row['a'].strip() + ".jpg"
        b_image_name = row['b'].strip() + ".jpg"
        c_image_name = row['c'].strip() + ".jpg"
        a_testdataset = tf.data.Dataset.from_tensors(str(images_path / a_image_name))
        b_testdataset = tf.data.Dataset.from_tensors(str(images_path / b_image_name))
        c_testdataset = tf.data.Dataset.from_tensors(str(images_path / c_image_name))
        test_dataset = tf.data.Dataset.zip((a_testdataset, b_testdataset, c_testdataset))
        test_dataset = test_dataset.map(preprocess_triplets)
        test_dataset = test_dataset.batch(1, drop_remainder=False)
        t = siamese_model(list(test_dataset.as_numpy_iterator()))
        if t[0].numpy()[0] < t[1].numpy()[0]:
            f.write('1')
            f.write('\n')
        else:
            f.write('0')
            f.write('\n')

    
