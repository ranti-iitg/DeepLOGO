from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.core import Dense, Activation, TimeDistributedDense, Dropout, Reshape, Flatten
from keras.models import Model
from keras.layers import Input
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, GlobalAveragePooling2D

number_of_brands=22
inputsize=224
max_frames=10

model_rnn3=Sequential()

# create the base pre-trained model
base_model = VGG19(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(number_of_brands ,activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#here X, Y are your Logo data images and labels
h = model.fit(X,Y, verbose=1, validation_split=0.1, nb_epoch=100,shuffle=True)
model.save("my_weight.h5")
model2=Model(input=base_model.input,output=x)
# Now we are going to train RNN over top of it.

 
model_rnn3.add(TimeDistributed(model2, input_shape=(max_frames,224,224,3)))
model_rnn3.add(GRU(output_dim=100,return_sequences=True))
model_rnn3.add(GRU(output_dim=50,return_sequences=False))
model_rnn3.add(Dropout(.2))
model_rnn3.add(Dense(number_of_brands,activation='softmax'))
model_rnn3.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#model_rnn3.fit(X,y) this X,y is your frame data from video

