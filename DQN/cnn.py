import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

class neural_kernel():
    def __init__(self,iS=(100,100,3),learning_rate=0.0005):

        self.learning_rate = learning_rate
        self.input_shape = iS
        self.n_outputs = 4    # UP, DOWN, LEFT, RIGHT
        self.model = Sequential() #modèle vide

        # Ajout de couche au modèle :
        self.model.add(Conv2D(32, (3,3), activation ='relu', input_shape = self.input_shape))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Conv2D(64, (2,2), activation = 'relu'))
        self.model.add(Flatten())
        self.model.add(Dense(units = 256, activation = 'relu'))
        self.model.add(Dense(units = self.n_outputs))

        # On fournit au modèle comment calculer l'erreur et quelle fonction pour l'optimiser
        self.model.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate = self.learning_rate))

    def loadModel(self, filepath):
        self.model = load_model(filepath)
        return self.model