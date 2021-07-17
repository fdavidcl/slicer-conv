import tensorflow as tf
import numpy as np
import time
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

def standard_loss(ly):
    """
    Computes the binary cross-entropy of the two layers passed as parameters

    Args:
        ly: tuple or list of two Keras layers (input and output)
    
    Returns:
        A tensor with the averaged binary cross-entropy
    """
    inp_data, out = ly
    return tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(inp_data, out))

class SlicerLoss(tf.keras.layers.Layer):
    """
    Slicer loss function (Keras layer)

    This class implements the special loss function used in Slicer autoencoders.

    Examples:
        inp_data = tf.keras.layers.Input(shape=input_shape)
        inp_label = tf.keras.layers.Input(shape=(1,))
        encoder_t # output tensor of encoder model
        decoder_t # output tensor of decoder model
        svm_layer = SVMLayer()
        svm_t = svm_layer(encoder_t)
        loss = SlicerLoss(self.alpha)([inp_data, inp_label, svm_t, decoder_t])
    """
    def __init__(self, alpha=0.1, **kwargs):
        """
        Initialization of a Slicer loss

        Args:
            alpha: weighting coefficient for the SVM term within the loss
            kwargs: arguments for `tensorflow.keras.layers.Layer`
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        
    def build(self, input_shape):
        pass
    
    def call(self, x, mask = None):
        """
        Computation of the Slicer loss

        This function is automatically called when the loss is used in a Keras model and the 
        results are computed when a forward pass is executed.
        """
        inp_data, inp_label, svm_out, out = x
        rec_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(inp_data, out)) # Use mean squared error if data not in [0,1]
        # compute svm error term
        t_n = 2 * inp_label - 1 # now class is in {-1, 1}
        svm_loss = tf.math.reduce_sum(tf.math.square(t_n - svm_out)) / 2
        loss = rec_loss + self.alpha * svm_loss
        return loss
        
    def compute_output_shape(input_shape):
        """
        Output shape of the layer
        """
        (input_shape[0], 1)
    
    def get_config(self):
        """
        Configuration for saving purposes
        """
        return {"alpha": self.alpha}

    @classmethod
    def from_config(cls, config):
        """
        Configuration loader
        """
        return cls(**config)
        
def SVMLayer(units=None, mu=0.01, **kwargs):
    """
    A Keras layer/model simulating a linear (support vector) classifier

    Args:
        units: not used, this arg is here to remove it from kwargs
        mu: the coefficient of the L2 kernel regularization
        kwargs: other parameters for `tf.keras.layers.Dense`

    Returns:
        A Keras model where the output is the predicted class of the classifier
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l2(mu), **kwargs)
    ])

def resnetv2_block(x: tf.Tensor, downsample: bool, filters: int, kernel_size: int = 3, deconv = False) -> tf.Tensor:
    """
    Residual block (from ResNetV2)

    Takes an input tensor and applies a residual block to it, optionally obtaining 
    a smaller sized tensor. This can also use transpose convolutional layers to 
    perform the complementary operation.

    Args:
        x: input tensor
        downsample: boolean indicating whether to reduce the tensor side (or increase it in the deconvolutional case)
        filters: number of filters
        kernel_size: side of the kernels
        deconv: boolean indicating whether to perform the transpose convolutional

    Returns:
        Output tensor of the residual block
    """
    convclass = tf.keras.layers.Conv2DTranspose if deconv else tf.keras.layers.Conv2D
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    y = convclass(kernel_size=kernel_size,
           strides= (1 if not downsample else 2),
           filters=filters,
           padding="same")(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = convclass(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)
    
    if downsample or filters != x.shape[3]:
        x = convclass(kernel_size=1,
                                  strides=2 if downsample else 1,
                                  filters=filters,
                                  padding="same")(x)
    
    out = tf.keras.layers.Add()([x, y])
    return out

class Autoencoder(object):
    """
    Convolutional autoencoder

    Can optionally use the Slicer loss function. This is not a Keras model but it implements 
    the essential interface of a model.

    Examples:
        autoencoder = Autoencoder(slicer=False, n_resnet_blocks=5).build(instance_shape)
        slicer = Autoencoder(slicer=True, n_resnet_blocks=5, alpha=0.1).build(instance_shape)
    """
    def __init__(self, slicer=True, n_resnet_blocks=5, alpha=0.1, encoding_dim=128, **kwargs):
        """
        Autoencoder initialization

        Args:
            slicer: True to use the Slicer loss
            n_resnet_blocks: number of residual (ResNetV2) blocks to be applied (in the encoder and the decoder)
            alpha: coefficient of the SVM loss, only useful when `slicer` is selected
            encoding_dim: dimension of the inner encoding layer
            kwargs: unused
        """
        self.slicer = slicer
        self.n_resnet_blocks = n_resnet_blocks
        self.depths = lambda i: 2**(i//2+3)
        self.alpha = alpha
        self.encoding_dim = encoding_dim
        
    def build(self, input_shape):
        """
        Build the autoencoder according to the input shape

        Args:
            input_shape: the shape of each input image
        
        Returns:
            This object
        """
        inp_data = tf.keras.layers.Input(shape=input_shape)
        inp_label = tf.keras.layers.Input(shape=(1,))
        encoder = inp_data
        
        # first conv2d in a resnet:
        encoder = tf.keras.layers.Conv2D(kernel_size=7, strides=1, filters=self.depths(0), padding="same", use_bias=False)(encoder)
        
        # each of the residual blocks
        for i in range(self.n_resnet_blocks):
            encoder = resnetv2_block(encoder, True, self.depths(i))
            
        last_conv_shape = encoder.shape[1:]
        encoder = tf.keras.layers.Flatten()(encoder)
        encoder = tf.keras.layers.Dense(self.encoding_dim)(encoder)
        encoder_base = tf.keras.models.Model(inp_data, encoder)

        print(f"Conv encoding: {last_conv_shape}, internal encoding: {self.encoding_dim}")
        
        decoder = decoder_i = tf.keras.layers.Input(shape=encoder.shape[1:])
        decoder = tf.keras.layers.Dense(np.prod(last_conv_shape))(decoder)
        decoder = tf.keras.layers.Reshape(last_conv_shape)(decoder)
        for i in range(self.n_resnet_blocks-1):
#             decoder = tf.keras.layers.UpSampling2D(2)(decoder)
            decoder = resnetv2_block(decoder, True, self.depths(self.n_resnet_blocks-i-1), deconv=True)

        decoder = tf.keras.layers.UpSampling2D(2)(decoder)
#         decoder = resnetv2_block(decoder, True, 8, 7, deconv=True)
        decoder = tf.keras.layers.Conv2DTranspose(kernel_size=1, strides=1, filters=1, padding="same")(decoder)
        decoder = tf.keras.activations.sigmoid(decoder)
        decoder_base = tf.keras.models.Model(decoder_i, decoder)

        encoder_t = encoder_base(inp_data)
        decoder_t = decoder_base(encoder_t)

        if self.slicer:
            svm_layer = SVMLayer()
            svm_t = svm_layer(encoder_t)

            loss = SlicerLoss(self.alpha)([inp_data, inp_label, svm_t, decoder_t])
        else:
            loss = tf.keras.layers.Lambda(standard_loss)([inp_data, decoder_t])
            
        self.encoder_model = tf.keras.models.Model(inp_data, encoder_t)
        self.ae_model = tf.keras.models.Model(inp_data, decoder_t)
        self.trainable_model = tf.keras.models.Model(
            [inp_data, inp_label],
            loss
        )
        return self
    
    def __call__(self, x):
        """
        Model application

        Computes the output of the model for a given input

        Args:
            x: input tensor
        
        Returns:
            The output of the autoencoder model (the encoding)
        """
        return self.encoder_model(x)
    
    @tf.function
    def train_step(self, x, y):
        """
        Training step

        A Tensorflow function for each forward-backward step

        Args:
            x: input images
            y: image classes
        
        Returns:
            Loss value for this batch
        """
        with tf.GradientTape() as tape:
            loss_value = self.trainable_model([x, y], training=True)
        grads = tape.gradient(loss_value, self.trainable_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_model.trainable_weights))
        return loss_value

    def validation_step(self, x, y):
        """
        Validation step

        Computes the loss value for a given input, without training the network

        Args:
            x: input images
            y: image classes
        
        Returns:
            Loss value for this batch
        """
        loss_value = np.mean(self.trainable_model.predict([x, y], batch_size=8))
        return loss_value

    def fit(self, train_generator, epochs, optimizer, keep_best=True, val_generator=None, *args, **kwargs):
        """
        Training loop

        Args:
            train_generator: generator with training data
            epochs: desired number of passes of the training dataset
            optimizer: Keras optimizer object
            keep_best: boolean indicating whether to keep the best or the last model
            val_generator: optional generator of validation data
        
        Returns:
            This object
        """
        self.optimizer = optimizer
        steps_by_epoch = len(train_generator)
        best_weights = None
        best_loss = float("inf")
        val_batch = next(val_generator)
        
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = 0
            
            for step in range(steps_by_epoch):
                x_batch_train, y_batch_train = next(train_generator)
                train_loss = self.train_step(x_batch_train, y_batch_train)

            if val_generator is None:
                print(f"Epoch {epoch} / time taken: {time.time() - start_time}, train loss: {train_loss}")
                if train_loss < best_loss:
                    best_weights = self.trainable_model.get_weights()
                    best_loss = train_loss
            else:
                val_loss = self.validation_step(*val_batch)
                print(f"Epoch {epoch} / time taken: {time.time() - start_time}, train loss: {train_loss}, val loss: {val_loss}")
                if val_loss < best_loss:
                    best_weights = self.trainable_model.get_weights()
                    best_loss = val_loss

        
        timestamp = str(time.time()).split(".")[0]
        self.last_weights = self.trainable_model.get_weights()
        self.trainable_model.save(f"{'slicer' if self.slicer else 'autoencoder'}_last_{timestamp}.h5")
        self.best_weights = best_weights
        if keep_best:
            self.trainable_model.set_weights(best_weights)
            self.trainable_model.save(f"{'slicer' if self.slicer else 'autoencoder'}_best_{timestamp}.h5")
            print(f"Restored weights with loss {best_loss}")
        
        return self
        
    def predict(self, *args, **kwargs):
        """
        Obtain the output of the model for a given input

        Args:
            args: typical arguments for the `predict` method of a Keras model (input data)
            kwargs: same

        Returns:
            The output of the model (the encoding) for the given inputs
        """
        return self.encoder_model.predict(*args, **kwargs)
    
    def predict_last(self, *args, **kwargs):
        """
        Obtain the output of the last layer for a given input

        Args:
            args: typical arguments for the `predict` method of a Keras model (input data)
            kwargs: same

        Returns:
            The output of the model (the reconstruction) for the given inputs
        """
        self.trainable_model.set_weights(self.last_weights)
        return self.encoder_model.predict(*args, **kwargs)
    
import os
import glob
    
class DataLoader(object):
    """
    Data loader

    This class is prepared to load partitioned COVIDGR-1.0 data with the following file hierarchy:
      - COVIDGR1.05fcvX
          - partitionY
              - train
                  - N
                  - P
              - val
                  - N
                  - P
              - test
                  - N
                  - P
    where X is the "repetition" number and Y is the partition index, the resulting validation
    being a X-times Y-fold validation.
    """
    def __init__(self, cv, partition, instance_shape):
        """
        Initialization
        
        Args:
            cv: repetition number (1 through 5)
            partition: partition number (0 through 4)
            instance_shape: 
        """

        self.random_rotation = 5
        self.random_shift = 0
        self.random_zoom = 0
        self.horizontal_flip = True
        self.image_dir = f"COVIDGR1.05fcv{cv}/partition{partition}"
        self.imgs_cols = instance_shape[0]
        self.imgs_rows = instance_shape[1]
        self.batch_size = 8
        self.classes = next(os.walk(self.image_dir + '/train'))[1]
        self.classes = sorted(self.classes)
        self.total_train = len(glob.glob(self.image_dir + '/train/*/*.jpg'))

        self.datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range = self.random_rotation,
            rescale=1/255.,
            width_shift_range = self.random_shift,
            height_shift_range = self.random_shift,
            zoom_range = self.random_zoom,
            horizontal_flip = self.horizontal_flip)

        self.datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

    def train_generator(self, predict=False):
        """
        Training generator

        Args:
            predict: whether to return the whole dataset in the first batch (prediction mode) or split by mini-batches
        
        Returns:
            A data generator of (x, y) batch pairs for training
        """
        return self.datagen_train.flow_from_directory(self.image_dir + '/train',
                            target_size = (self.imgs_cols, self.imgs_rows),
                            batch_size = self.total_train if predict else self.batch_size, shuffle = True,
                            color_mode="grayscale",class_mode="binary",
                            classes = self.classes)
    
    def val_generator(self):
        """
        Validation generator

        Returns:
            A data generator of (x, y) batch pairs for validation
        """
        return self.datagen_val.flow_from_directory(self.image_dir + '/val',
                            target_size = (self.imgs_cols, self.imgs_rows),
                            batch_size = 100, shuffle = False,
                            color_mode="grayscale",class_mode="binary",
                            classes = self.classes)
    def test_generator(self):
        """
        Test generator

        Returns:
            A data generator of (x, y) batch pairs for testing
        """
        return self.datagen_val.flow_from_directory(self.image_dir + '/test',
                            target_size = (self.imgs_cols, self.imgs_rows),
                            batch_size = 200, shuffle = False,
                            color_mode="grayscale",class_mode="binary",
                            classes = self.classes)
    
def train_predict(datagen):
    """
    Full train-and-predict experiment

    This function receives a data loader object and performs a full experiment with
    a basic convolutional autoencoder and a Slicer autoencoder.

    Args:
        datagen: `DataLoader` object for the current repetition and partition
    
    Returns:
        A dict with tuples structured as follows:
            {
                "train": (training inputs, training labels, autoencoder encodings, slicer encodings),
                "test": (test inputs, test labels, autoencoder encodings, slicer encodings),
            }
    """
    train_generator = datagen.train_generator()
    predict_generator = datagen.train_generator(predict=True)
    val_generator = datagen.val_generator()
    test_generator = datagen.test_generator()
    
    tf.keras.backend.clear_session()
    instance_shape = (512,512,1)
    n_resnet_blocks = 6
    autoencoder = Autoencoder(slicer=False, n_resnet_blocks=n_resnet_blocks).build(instance_shape)
    slicer = Autoencoder(slicer=True, n_resnet_blocks=n_resnet_blocks, alpha=0.1).build(instance_shape)
    epochs = 50
    steps_by_epoch = len(train_generator)
    optimizer = tf.keras.optimizers.Adam()
    autoencoder.fit(train_generator, epochs, optimizer, val_generator=val_generator)
    slicer.fit(train_generator, epochs, optimizer, val_generator=val_generator)
    
    x_train, y_train = next(predict_generator)
    x_t_enc = slicer.predict(x_train, batch_size=1)
    x_t_ae = autoencoder.predict(x_train, batch_size=1)
    
    x_test, y_test = next(test_generator)
    x_v_enc = slicer.predict(x_test, batch_size=1)
    x_v_ae = autoencoder.predict(x_test, batch_size=1)
    
    return {"train": (x_train, y_train, x_t_ae, x_t_enc),
            "test": (x_test, y_test, x_v_ae, x_v_enc)}
