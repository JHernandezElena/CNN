import json
import keras
from keras import layers
from keras import models
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sys import platform

from results import Results

import matplotlib

if platform == "darwin":
    matplotlib.use("TkAgg")  # Prevents matplotlib from crashing in macOS

    # Fix macOS error "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from matplotlib import pyplot as plt


class CNN:
    """Class to classify images using a transfer learning or fine-tuning on a pre-trained CNN.

        Examples:
            1. Training and evaluating the CNN. Optionally, save the model.
                cnn = CNN()
                cnn.train(training_dir, validation_dir, base_model='ResNet50')
                cnn.predict(validation_dir)
                cnn.save(filename)

            2. Loading a trained CNN to evaluate against a previously unseen test set.
                cnn = CNN()
                cnn.load(filename)
                cnn.predict(test_dir)

    """

    def __init__(self):
        """CNN transfer learning class initializer."""
        self._model_name = ""
        self._model = None
        self._target_size = None
        self._preprocessing_function = None

    def train(self, training_dir: str, validation_dir: str, base_model: str, epochs: int = 1,
              unfreezed_convolutional_layers: int = 0, training_batch_size: int = 32, validation_batch_size: int = 32,
              learning_rate: float = 1e-4): ##BATCH 32,64
        """Use transfer learning or fine-tuning to train a base network to classify new categories.

        Args:
            training_dir: Relative path to the training directory (e.g., 'dataset/training').
            validation_dir: Relative path to the validation directory (e.g., 'dataset/validation').
            base_model: Pre-trained CNN { DenseNet121, DenseNet169, DenseNet201, InceptionResNetV2, InceptionV3,
                                          MobileNet, MobileNetV2, NASNetLarge, NASNetMobile, ResNet50, VGG16, VGG19,
                                          Xception }.
            epochs: Number of times the entire dataset is passed forward and backward through the neural network.
            unfreezed_convolutional_layers: Starting from the end, number of trainable convolutional layers.
            training_batch_size: Number of training examples used in one iteration.
            validation_batch_size: Number of validation examples used in one iteration.
            learning_rate: Optimizer learning rate.

        """
        # Initialize a base pre-trained CNN without the classification layer
        self._initialize_base_model(base_model, unfreezed_convolutional_layers, include_top=False) #el false quita la parte de clasificacion

        # Configure loading and pre-processing/data augmentation functions
        print('\n\nReading training and validation data...')
        training_datagen = ImageDataGenerator(
            preprocessing_function=self._preprocessing_function,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,  # Randomly flip half of the images horizontally
            fill_mode='nearest'  # Strategy used for filling in new pixels that appear after transforming images
        )

        validation_datagen = ImageDataGenerator(preprocessing_function=self._preprocessing_function)

        training_generator = training_datagen.flow_from_directory(
            training_dir,
            target_size=self._target_size,
            batch_size=training_batch_size,
            class_mode='categorical'
        )

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self._target_size,
            batch_size=validation_batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Add a new softmax output layer to learn the training dataset classes
        self._add_output_layers(training_generator.num_classes)
        
        # Compile the model
        optimizer = Adam(lr=learning_rate, beta_1=0.5)
        self._model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Display a summary of the model
        print('\n\nModel summary')
        self._model.summary()

        # Callbacks. Check https://keras.io/callbacks/ for more alternatives.
        # EarlyStopping and ModelCheckpoint are probably the most relevant.
         #MIRAR EARLY STOPPOING

        # To launch TensorBoard type the following in a Terminal window: tensorboard --logdir /path/to/log/folder
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=0,
                                                           batch_size=32, write_graph=True, write_grads=False,
                                                           write_images=False, embeddings_freq=0,
                                                           embeddings_layer_names=None, embeddings_metadata=None,
                                                           embeddings_data=None, update_freq='epoch')

        callbacks = [tensorboard_callback]

        # Train the network
        print("\n\nTraining CNN...")

        history = self._model.fit_generator(
            training_generator,
            epochs=epochs,
            steps_per_epoch=training_generator.samples // training_generator.batch_size,
            validation_data=validation_generator, #SOLO SE USA PARA EL EARLY STOPPING SI LO PONEMOS
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            class_weight='auto',  # Weight all classes equally to compensate for underrepresented categories.
            callbacks=callbacks
        )

        # Plot model training history
        if epochs > 1:
            self._plot_training(history)

    def predict(self, test_dir: str, dataset_name: str = "", save: bool = True):
        """Evaluates a new set of images using the trained CNN.

        Args:
            test_dir: Relative path to the validation directory (e.g., 'dataset/test').
            dataset_name: Dataset descriptive name.
            save: Save results to an Excel file.

        """
        # Configure loading and pre-processing functions
        print('Reading test data...')
        test_datagen = ImageDataGenerator(preprocessing_function=self._preprocessing_function)

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self._target_size,
            batch_size=1,  # A batch size of 1 ensures that all test images are processed
            class_mode='categorical',
            shuffle=False
        )

        # Predict categories
        predictions = self._model.predict_generator(test_generator, steps=test_generator.samples)
        predicted_labels = np.argmax(predictions, axis=1).ravel().tolist()

        # Format results and compute classification statistics
        results = Results(test_generator.class_indices, dataset_name=dataset_name)
        accuracy, confusion_matrix, classification = results.compute(test_generator.filenames, test_generator.classes,
                                                                     predicted_labels)
        # Display and save results
        results.print(accuracy, confusion_matrix)

        if save:
            results.save(confusion_matrix, classification)

    def load(self, filename: str):
        """Loads a trained CNN model and the corresponding preprocessing information.

        Args:
           filename: Relative path to the file without the extension.

        """
        # Load Keras model
        self._model = models.load_model(filename + '.h5')

        # Load base model information
        with open(filename + '.json') as f:
            self._model_name = json.load(f)

        self._initialize_attributes()

    def save(self, filename: str):
        """Saves the model to an .h5 file and the model name to a .json file.

        Args:
           filename: Relative path to the file without the extension.

        """
        # Save Keras model
        self._model.save(filename + '.h5')

        # Save base model information
        with open(filename + '.json', 'w', encoding='utf-8') as f:
            json.dump(self._model_name, f, ensure_ascii=False, indent=4, sort_keys=True)

    def _initialize_base_model(self, base_model: str, unfreezed_convolutional_layers: int, include_top: bool = True,
                               pooling: str = 'avg'):
        """Initializes the base model.

        Args:
            base_model: Pre-trained CNN { DenseNet121, DenseNet169, DenseNet201, InceptionResNetV2, InceptionV3,
                                          MobileNet, MobileNetV2, NASNetLarge, NASNetMobile, ResNet50, VGG16, VGG19,
                                          Xception }.
            unfreezed_convolutional_layers: Starting from the end, number of trainable convolutional layers.
            include_top: True to use the full base model; false to remove the last classification layers.
            pooling: Optional pooling mode for feature extraction when include_top is False
                - None: The output of the model will be the 4D tensor output of the last convolutional block.
                - 'avg': Global average pooling will be applied to the output of the last convolutional block, and thus
                         the output of the model will be a 2D tensor.
                - 'max': Global max pooling will be applied.

        Raises:
            TypeError: If the unfreezed_convolutional_layers parameter is not an integer.
            ValueError: If the unfreezed_convolutional_layers parameter is not a positive number (>= 0).
            ValueError: If the base model is not known.

        """
        self._model_name = base_model
        self._initialize_attributes()

        input_shape = self._target_size + (3,)

        # Initialize the base model. Loads the network weights from disk.
        # NOTE: If this is the first time you run this function, the weights will be downloaded from the Internet.
        self._model = getattr(keras.applications, base_model)(weights='imagenet', include_top=include_top,
                                                              input_shape=input_shape, pooling=pooling)

        # Freeze convolutional layers
        if type(unfreezed_convolutional_layers) != int:
            raise TypeError("unfreezed_convolutional_layers must be a positive integer.")

        if unfreezed_convolutional_layers == 0:
            freezed_layers = self._model.layers
        elif unfreezed_convolutional_layers > 0:
            freezed_layers = self._model.layers[:-unfreezed_convolutional_layers]
        else:
            raise ValueError("unfreezed_convolutional_layers must be a positive integer.")

        for layer in freezed_layers:
            layer.trainable = False

    def _initialize_attributes(self):
        """Initialize the input image shape along with the pre-processing function.

        Raises:
            ValueError: If the model is unknown.

        """
        if self._model_name in ('DenseNet121', 'DenseNet169', 'DenseNet201'):
            self._target_size = (224, 224)
            self._preprocessing_function = keras.applications.densenet.preprocess_input
        elif self._model_name == 'InceptionResNetV2':
            self._target_size = (299, 299)
            self._preprocessing_function = keras.applications.inception_resnet_v2.preprocess_input
        elif self._model_name == 'InceptionV3':
            self._target_size = (299, 299)
            self._preprocessing_function = keras.applications.inception_v3.preprocess_input
        elif self._model_name == 'MobileNet':
            self._target_size = (224, 224)
            self._preprocessing_function = keras.applications.mobilenet.preprocess_input
        elif self._model_name == 'MobileNetV2':
            self._target_size = (224, 224)
            self._preprocessing_function = keras.applications.mobilenet_v2.preprocess_input
        elif self._model_name == 'NASNetLarge':
            self._target_size = (331, 331)
            self._preprocessing_function = keras.applications.nasnet.preprocess_input
        elif self._model_name == 'NASNetMobile':
            self._target_size = (224, 224)
            self._preprocessing_function = keras.applications.nasnet.preprocess_input
        elif self._model_name == 'ResNet50':
            self._target_size = (224, 224)
            self._preprocessing_function = keras.applications.resnet50.preprocess_input
        elif self._model_name == 'VGG16':
            self._target_size = (224, 224)
            self._preprocessing_function = keras.applications.vgg16.preprocess_input
        elif self._model_name == 'VGG19':
            self._target_size = (224, 224)
            self._preprocessing_function = keras.applications.vgg19.preprocess_input
        elif self._model_name == 'Xception':
            self._target_size = (299, 299)
            self._preprocessing_function = keras.applications.xception.preprocess_input
        else:
            raise ValueError("Base model not supported. Possible values are 'DenseNet121', 'DenseNet169', "
                             "'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', "
                             "'NASNetLarge', 'NASNetMobile', 'ResNet50', 'VGG16', 'VGG19' and 'Xception'.")

    def _add_output_layers(self, class_count: int, fc_layer_size: int = 1024):
        """Append a fully-connected shallow neural network with softmax outputs at the end of the base model.

        Args:
          class_count: Number of classes (i.e., number of output softmax neurons).
          fc_layer_size: Number of neurons in the hidden layer.

        """
        # Create a new model
        model = models.Sequential()

        # Add the convolutional base model
        model.add(self._model)

        # Add new layers
        model.add(layers.Dense(fc_layer_size, activation='relu')) #CAPA RELU COMPLETAMENTE CONECTADA
        model.add(layers.Dropout(0.2)) #PARA EVITAR SOBRE APRENDIZAJE (ANULA EL 20% DE LAS ENTRADAS - PROBAR 10)
        model.add(layers.Dense(class_count, activation='softmax'))

        # Assign the new model to the class attribute
        self._model = model

    @staticmethod
    def _plot_training(history):
        """Plots the evolution of the accuracy and the loss of both the training and validation sets.

        Args:
            history: Training history.

        """
        training_accuracy = history.history['acc']
        validation_accuracy = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(training_accuracy))

        # Accuracy
        plt.figure()
        plt.plot(epochs, training_accuracy, 'r', label='Training accuracy')
        plt.plot(epochs, validation_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
