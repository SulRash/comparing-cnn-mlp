from tensorflow.keras.datasets import cifar10
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import render, init_notebook_plotting
import matplotlib.pyplot as plt

# Importing of service libraries
import numpy as np

np.random.seed(1671)
N_CLASSES = 10

tf.config.set_visible_devices([], 'GPU')

# Loading in input data
(input_X_train, output_y_train), (input_X_test, output_y_test) = cifar10.load_data()
print('input_X_train shape:', input_X_train.shape)
print(input_X_train.shape[0], 'train samples')
print(input_X_test.shape[0], 'test samples')

# Convert to categorical
output_Y_train = utils.to_categorical(output_y_train, N_CLASSES)
output_Y_test = utils.to_categorical(output_y_test, N_CLASSES) 


# Float and normalization
input_X_train = input_X_train.astype('float32')
input_X_test = input_X_test.astype('float32')
input_X_train /= 255
input_X_test /= 255

# Creating a multi-layer perceptron.
class MLP:
    
    # Reshaping input for use.
    def reshape(self):
        self.dimension = 32*32*3
        self.input_x_train = input_X_train.reshape(50000, self.dimension)
        self.input_x_test = input_X_test.reshape(10000, self.dimension)
        self.input_x_train = self.input_x_train.astype('float32')
        self.input_x_test = self.input_x_test.astype('float32')
        
        self.input_x_train /= 255
        self.input_x_test /= 255
        
        self.output_y_train = utils.to_categorical(output_y_train, 10)
        self.output_y_test = utils.to_categorical(output_y_test, 10)
        
        label = self.output_y_train[2]
        print(str(label))
    
    # Function used to create fresh multi-layer perceptron to optimize.
    def build_optimize(self, hidden_layers, neurons_per_layer, activation, optimizer, learning_rate, dropout_rate):
        
        model = Sequential()
        
        model.add(Dense(neurons_per_layer, input_shape=(self.dimension,)))
        model.add(Activation(activation))
        
        for i in range(hidden_layers):
            model.add(Dense(neurons_per_layer))
            model.add(Activation(activation))
            model.add(Dropout(dropout_rate))
        
        # Softmax as final activation layer with 10 neurons since this is a classification problem with 10 possible answers.
        model.add(Dense(10))
        model.add(Activation('softmax'))
        
        if optimizer == "adam":
            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        elif optimizer == "sgd":
            model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=learning_rate), metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['accuracy'])
        
        model.summary()
        
        return model

parameters=[
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [0.0001, 0.05],
        "log_scale": True,
    },
    {
        "name": "dropout_rate",
        "type": "range",
        "bounds": [0.01, 0.5],
        "log_scale": True,
    },
    {
        "name": "hidden_layers",
        "type": "range",
        "bounds": [1, 8],
        "value_type": "int"
    },
    {
        "name": "neurons_per_layer",
        "type": "range",
        "bounds": [50, 300],
        "value_type": "int"
    },
    {
        "name": "activation",
        "type": "choice",
        "values": ['tanh', 'sigmoid', 'relu'],
    },
    {
        "name": "optimizer",
        "type": "choice",
        "values": ['adam', 'rms', 'sgd'],
    },
]

# Optimizing hyperparameters in multi-layer perceptron (rudimentary implementation) using this general framework method.
def hyperparamter_optimization(parameters):
    mlp = MLP()
    mlp.reshape()
    model = mlp.build_optimize(parameters.get('hidden_layers'), parameters.get('neurons_per_layer'), parameters.get('activation'), parameters.get('optimizer'), parameters.get('learning_rate'), parameters.get('dropout_rate'))
    
    EPOCHS = 50
    
    result = model.fit(mlp.input_x_train, mlp.output_y_train, batch_size=128, epochs=EPOCHS, validation_split=0.2)
    
    last10_scores = np.array(result.history['val_loss'][-10:])
    mean = last10_scores.mean()
    std = last10_scores.std()
    
    if np.isnan(mean):
        return 9999.0, 0.0
    
    del model

    return mean, std

# Using Ax Dev for hyperparameter optimization.
init_notebook_plotting()

ax_client = AxClient()

ax_client.create_experiment(
    name="parameter_optimization",
    parameters=parameters,
    objective_name='mlp',
    minimize=True)

def evaluate(parameters):
    return {"mlp": hyperparamter_optimization(parameters)}

for i in range(100):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
    
trials = ax_client.get_trials_data_frame().sort_values('trial_index')

# Exporting optimization trials into an easy to read CSV
trials.to_csv(("outmlp.csv"),index=False,sep="&")

best_parameters, values = ax_client.get_best_parameters()

# The best set of parameters.
for parameter in best_parameters.items():
  print(parameter)

# The best score achieved.
means, covariances = values
print(means)

mlp = MLP()
mlp.reshape()
model = mlp.build_optimize(hidden_layers=2, dropout_rate=0.03759, learning_rate=0.00052, neurons_per_layer=276, activation="relu", optimizer="adam")
history = model.fit(mlp.input_x_train, mlp.output_y_train, batch_size=128, epochs=50, validation_split=0.2,  verbose=1)
score = model.evaluate(mlp.input_x_test, mlp.output_y_test, batch_size=128, verbose=1)
print("\nTest score/loss:", score[0])
print('Test accuracy:', score[1])

# List all data in history
print(history.history.keys())

# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Multi-Layer Perceptron Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Multi-Layer Perceptron Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()