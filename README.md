
<h1> Example: Split a CNN Model into Two Parts </h1>

To run this example, [Google Colab](https://colab.research.google.com) is a recommended choice.

First of all, Clone!


```
!git clone https://github.com/nrasadi/split-keras-tensorflow-model
```

    Cloning into 'split-keras-tensorflow-model'...
    remote: Enumerating objects: 13, done.[K
    remote: Counting objects: 100% (13/13), done.[K
    remote: Compressing objects: 100% (13/13), done.[K
    remote: Total 13 (delta 5), reused 0 (delta 0), pack-reused 0[K
    Unpacking objects: 100% (13/13), done.



```
# Check if Colab is using gpu. 
# If not, Runtime tab -> Change runtime type -> Hardware Accelarator (GPU)
gpu_name = !nvidia-smi -L
print("GPU name: ", gpu_name[0]) if if len(gpu_name) >= 1 else print("No GPU!")
```

    GPU name:  GPU 0: Tesla P100-PCIE-16GB (UUID: GPU-0875064e-f48e-ff55-29b6-019dd05b5f02)



```
from pathlib import Path
import os

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers as KL
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import plot_model

%matplotlib inline
```


```
os.chdir("/content/split-keras-tensorflow-model")
import splitter
```

<h2> Load MNIST Dataset </h2>

It contains:
*   60000 train samples
*   10000 test samples




```
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# The data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float16") / 255
x_test = x_test.astype("float16") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 0s 0us/step
    x_train shape: (60000, 28, 28, 1)
    60000 train samples
    10000 test samples


<h2> Visualize Dataset </h2>



```
# Visualize dataset
plt.figure(figsize=(20,10))
sample_idx = np.random.randint(len(y_test), size=6)
for i in range(6):
  plt.subplot(161+i)
  plt.imshow(x_test[sample_idx[i]].reshape(28,28).astype(np.float), cmap='gray')
  # print("Label:", np.where(y_test[sample_idx] == 1)[0][0])
  # plt.axis("off")
  plt.xticks([]); plt.yticks([])
  plt.xlabel(str(np.where(y_test[sample_idx[i]] == 1)[0][0]), fontsize=25)
plt.show()
```


![png](split_model_example_files/split_model_example_9_0.png)


<h2> Design Digit Recognizer Model Architecture </h2>

<h3> Quantize the model </h3>

*   At first, I reduced the precision of weights and activations used in neural network to make the training process <b>faster</b>. 
*   It may introduce accuracy instability on real models however. Therefore it's better to use float32 or [tensorflow mixed precision policy](https://www.tensorflow.org/guide/keras/mixed_precision) instead!

<h3> Apply augmentation </h3>
 
To prevent <b>overfitting</b> and boost the accuracy of the model, I used a few -very simple- augmentation techniques before feeding images to the DNN.

*   All the following techniques are stochastic with the degree determined by 'factor'.
*   dropout: simulates noise or pixel drop.
*   zoom: applies zoom-in and zoom-out on images. (i.e. image scaling.)
*   translation, rotation: translates and rotates images.
*   contrast: modifies contrast of images.

<h3> Model architecture design </h3>

I designed a ResNet-like architecture, but pretty much simpler and with way too less parameters.

It consists of 4 blocks:

1. Conv(16)-Conv(32)-MaxPooling
2. ResBlock: Conv(32)-Conv(32)
3. ResBlock: Conv(32)-Conv(32)
4. Conv(64)-Dense(128)-Dense(10)

* I applied *ReLU6* activation on residual blocks and *ReLU* on others
* 20% dropout is applied on all conv blocks after batch normalization.
* 50% dropout is applied just before last dense layer (output).

<h3> Model construction </h3>

I used *RMSProp* as the optimzer and *categorical cross entropy* as the loss function.


```
# Quantize the network (Reduce precision of weights and activations to float16) 
K.set_floatx('float16')

# Apply some augmentation techniques on input images
factor = .1
augmentation = keras.Sequential(
    [
      KL.Dropout(0.001, name="drop_0"),
      KL.experimental.preprocessing.RandomZoom((-factor, factor), name="zoom"),
      KL.experimental.preprocessing.RandomTranslation((-factor, factor), (-factor, factor), name="translate",),
      KL.experimental.preprocessing.RandomRotation((-factor, factor), name="rotate"),
      KL.experimental.preprocessing.RandomContrast(factor, name="contrast"),
    ], name="augmentation"
)

# Define Conv Block
def Conv(tensor, filter, k=3, activation='relu', drop_rate=.2, maxpool=False, name_pos="0"):
    """Constructs Conv2D layers"""

    x = KL.Conv2D(filter, k, name="conv_"+name_pos, padding="same")(tensor)
    x = KL.BatchNormalization(name="bn_"+name_pos)(x)
    
    if activation.lower() == 'relu6':
      x = KL.Activation(tf.nn.relu6, name="relu6_"+name_pos)(x)
    elif activation.lower() == 'swish':
      x = KL.Activation(tf.nn.swish, name="swish_"+name_pos)(x)
    else:
      x = KL.Activation('relu', name="relu_"+name_pos)(x)
    
    x = KL.Dropout(drop_rate, name="drop_"+name_pos)(x)

    if maxpool:
      x = KL.MaxPooling2D(3, name="maxp_"+name_pos)(x)

    return x

act = "relu6"

# Block 0
inp = keras.Input(shape=input_shape, name="input")
x = augmentation(inp)

# Block 1
x = Conv(x, 16, name_pos="1")
x1 = Conv(x, 32, maxpool=True, name_pos="2")

# Block 2
x = Conv(x1, 32, activation=act, name_pos="3")
x = Conv(x, 32, activation=act, name_pos="4")
x2 = KL.add([x, x1], name="add_1")

# Block 3
x = Conv(x2, 32, activation=act, name_pos="5")
x = Conv(x, 32, activation=act, name_pos="6")
x3 = KL.add([x, x2], name="split") # Split point

# Block 4
x = Conv(x3, 64, activation=act, name_pos="7")
# x = KL.GlobalAveragePooling2D(name="globavgp")(x)
x = KL.Flatten(name="Flatten")(x)
x = KL.Dense(128, activation='relu', name="dense_1")(x)
x = KL.Dropout(0.5, name="last_drop")(x)
out = KL.Dense(10, name="output", activation="softmax")(x)

# Create model
model = keras.Model(inp, out, name="digit_recognizer")
model.summary()

# Construct DAG
model.compile(loss="categorical_crossentropy", optimizer="RMSProp", metrics=["accuracy"])
```

    Model: "digit_recognizer"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input (InputLayer)              [(None, 28, 28, 1)]  0                                            
    __________________________________________________________________________________________________
    augmentation (Sequential)       (None, 28, 28, 1)    0           input[0][0]                      
    __________________________________________________________________________________________________
    conv_1 (Conv2D)                 (None, 28, 28, 16)   160         augmentation[0][0]               
    __________________________________________________________________________________________________
    bn_1 (BatchNormalization)       (None, 28, 28, 16)   64          conv_1[0][0]                     
    __________________________________________________________________________________________________
    relu_1 (Activation)             (None, 28, 28, 16)   0           bn_1[0][0]                       
    __________________________________________________________________________________________________
    drop_1 (Dropout)                (None, 28, 28, 16)   0           relu_1[0][0]                     
    __________________________________________________________________________________________________
    conv_2 (Conv2D)                 (None, 28, 28, 32)   4640        drop_1[0][0]                     
    __________________________________________________________________________________________________
    bn_2 (BatchNormalization)       (None, 28, 28, 32)   128         conv_2[0][0]                     
    __________________________________________________________________________________________________
    relu_2 (Activation)             (None, 28, 28, 32)   0           bn_2[0][0]                       
    __________________________________________________________________________________________________
    drop_2 (Dropout)                (None, 28, 28, 32)   0           relu_2[0][0]                     
    __________________________________________________________________________________________________
    maxp_2 (MaxPooling2D)           (None, 9, 9, 32)     0           drop_2[0][0]                     
    __________________________________________________________________________________________________
    conv_3 (Conv2D)                 (None, 9, 9, 32)     9248        maxp_2[0][0]                     
    __________________________________________________________________________________________________
    bn_3 (BatchNormalization)       (None, 9, 9, 32)     128         conv_3[0][0]                     
    __________________________________________________________________________________________________
    relu6_3 (Activation)            (None, 9, 9, 32)     0           bn_3[0][0]                       
    __________________________________________________________________________________________________
    drop_3 (Dropout)                (None, 9, 9, 32)     0           relu6_3[0][0]                    
    __________________________________________________________________________________________________
    conv_4 (Conv2D)                 (None, 9, 9, 32)     9248        drop_3[0][0]                     
    __________________________________________________________________________________________________
    bn_4 (BatchNormalization)       (None, 9, 9, 32)     128         conv_4[0][0]                     
    __________________________________________________________________________________________________
    relu6_4 (Activation)            (None, 9, 9, 32)     0           bn_4[0][0]                       
    __________________________________________________________________________________________________
    drop_4 (Dropout)                (None, 9, 9, 32)     0           relu6_4[0][0]                    
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, 9, 9, 32)     0           drop_4[0][0]                     
                                                                     maxp_2[0][0]                     
    __________________________________________________________________________________________________
    conv_5 (Conv2D)                 (None, 9, 9, 32)     9248        add_1[0][0]                      
    __________________________________________________________________________________________________
    bn_5 (BatchNormalization)       (None, 9, 9, 32)     128         conv_5[0][0]                     
    __________________________________________________________________________________________________
    relu6_5 (Activation)            (None, 9, 9, 32)     0           bn_5[0][0]                       
    __________________________________________________________________________________________________
    drop_5 (Dropout)                (None, 9, 9, 32)     0           relu6_5[0][0]                    
    __________________________________________________________________________________________________
    conv_6 (Conv2D)                 (None, 9, 9, 32)     9248        drop_5[0][0]                     
    __________________________________________________________________________________________________
    bn_6 (BatchNormalization)       (None, 9, 9, 32)     128         conv_6[0][0]                     
    __________________________________________________________________________________________________
    relu6_6 (Activation)            (None, 9, 9, 32)     0           bn_6[0][0]                       
    __________________________________________________________________________________________________
    drop_6 (Dropout)                (None, 9, 9, 32)     0           relu6_6[0][0]                    
    __________________________________________________________________________________________________
    split (Add)                     (None, 9, 9, 32)     0           drop_6[0][0]                     
                                                                     add_1[0][0]                      
    __________________________________________________________________________________________________
    conv_7 (Conv2D)                 (None, 9, 9, 64)     18496       split[0][0]                      
    __________________________________________________________________________________________________
    bn_7 (BatchNormalization)       (None, 9, 9, 64)     256         conv_7[0][0]                     
    __________________________________________________________________________________________________
    relu6_7 (Activation)            (None, 9, 9, 64)     0           bn_7[0][0]                       
    __________________________________________________________________________________________________
    drop_7 (Dropout)                (None, 9, 9, 64)     0           relu6_7[0][0]                    
    __________________________________________________________________________________________________
    Flatten (Flatten)               (None, 5184)         0           drop_7[0][0]                     
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 128)          663680      Flatten[0][0]                    
    __________________________________________________________________________________________________
    last_drop (Dropout)             (None, 128)          0           dense_1[0][0]                    
    __________________________________________________________________________________________________
    output (Dense)                  (None, 10)           1290        last_drop[0][0]                  
    ==================================================================================================
    Total params: 726,218
    Trainable params: 725,738
    Non-trainable params: 480
    __________________________________________________________________________________________________


<h2> Plot Graphical Visualization of the Network Architecture </h2>


```
plot_model(model, rankdir='LR', dpi=600)
```

    dot: graph is too large for cairo-renderer bitmaps. Scaling by 0.647032 to fit
    





![png](split_model_example_files/split_model_example_13_1.png)



<h2> Train the Network </h2>

* Change batch size and number of epochs to fit best with the resources you have.


```
# Train the model
batch_size = 256
epochs = 200

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
```

    Epoch 1/200
    211/211 [==============================] - 4s 17ms/step - loss: inf - accuracy: 0.1503 - val_loss: inf - val_accuracy: 0.0737
    Epoch 2/200
    211/211 [==============================] - 3s 16ms/step - loss: inf - accuracy: 0.1971 - val_loss: inf - val_accuracy: 0.1145
    Epoch 3/200
    211/211 [==============================] - 3s 16ms/step - loss: inf - accuracy: 0.2585 - val_loss: 4.8047 - val_accuracy: 0.2532
    Epoch 4/200
    211/211 [==============================] - 3s 16ms/step - loss: inf - accuracy: 0.3223 - val_loss: 1.4385 - val_accuracy: 0.5479
    Epoch 5/200
    211/211 [==============================] - 3s 16ms/step - loss: inf - accuracy: 0.3804 - val_loss: 1.9453 - val_accuracy: 0.5161
    Epoch 6/200
    211/211 [==============================] - 3s 16ms/step - loss: inf - accuracy: 0.4395 - val_loss: 1.2207 - val_accuracy: 0.6436
    Epoch 7/200
    211/211 [==============================] - 3s 16ms/step - loss: inf - accuracy: 0.4954 - val_loss: 2.2734 - val_accuracy: 0.7373
    Epoch 8/200
    211/211 [==============================] - 3s 16ms/step - loss: inf - accuracy: 0.5439 - val_loss: 0.9976 - val_accuracy: 0.7554
    Epoch 9/200
    211/211 [==============================] - 3s 16ms/step - loss: 1.1602 - accuracy: 0.5884 - val_loss: 0.8174 - val_accuracy: 0.7954
    Epoch 10/200
    211/211 [==============================] - 4s 17ms/step - loss: 1.0449 - accuracy: 0.6362 - val_loss: 2.0156 - val_accuracy: 0.7666
    Epoch 11/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.9229 - accuracy: 0.6836 - val_loss: 2.6523 - val_accuracy: 0.6660
    Epoch 12/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.8521 - accuracy: 0.7129 - val_loss: 6.8359 - val_accuracy: 0.7354
    Epoch 13/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.7803 - accuracy: 0.7383 - val_loss: 8.5703 - val_accuracy: 0.7871
    Epoch 14/200
    211/211 [==============================] - 3s 15ms/step - loss: 0.7222 - accuracy: 0.7554 - val_loss: 2.3047 - val_accuracy: 0.8589
    Epoch 15/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.6812 - accuracy: 0.7690 - val_loss: 7.8789 - val_accuracy: 0.8052
    Epoch 16/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.6299 - accuracy: 0.7896 - val_loss: 5.8359 - val_accuracy: 0.7368
    Epoch 17/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.5894 - accuracy: 0.8081 - val_loss: 2.8086 - val_accuracy: 0.8555
    Epoch 18/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.5698 - accuracy: 0.8149 - val_loss: 0.7026 - val_accuracy: 0.9038
    Epoch 19/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.5386 - accuracy: 0.8325 - val_loss: 1.9121 - val_accuracy: 0.9009
    Epoch 20/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.5073 - accuracy: 0.8442 - val_loss: 3.0000 - val_accuracy: 0.8691
    Epoch 21/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.4768 - accuracy: 0.8545 - val_loss: 1.0449 - val_accuracy: 0.8589
    Epoch 22/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.4543 - accuracy: 0.8647 - val_loss: 3.6133 - val_accuracy: 0.8398
    Epoch 23/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.4294 - accuracy: 0.8672 - val_loss: 2.2637 - val_accuracy: 0.8799
    Epoch 24/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.3975 - accuracy: 0.8750 - val_loss: 3.2539 - val_accuracy: 0.7021
    Epoch 25/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.3904 - accuracy: 0.8770 - val_loss: 0.2781 - val_accuracy: 0.9414
    Epoch 26/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.3845 - accuracy: 0.8823 - val_loss: 0.5952 - val_accuracy: 0.9492
    Epoch 27/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.3628 - accuracy: 0.8838 - val_loss: 1.3418 - val_accuracy: 0.8687
    Epoch 28/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.3481 - accuracy: 0.8853 - val_loss: 0.1714 - val_accuracy: 0.9453
    Epoch 29/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.3367 - accuracy: 0.8882 - val_loss: 0.4204 - val_accuracy: 0.8813
    Epoch 30/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.3225 - accuracy: 0.8940 - val_loss: 0.2268 - val_accuracy: 0.9326
    Epoch 31/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.3220 - accuracy: 0.8945 - val_loss: 0.2029 - val_accuracy: 0.9531
    Epoch 32/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.3115 - accuracy: 0.9004 - val_loss: 0.1755 - val_accuracy: 0.9546
    Epoch 33/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2998 - accuracy: 0.9009 - val_loss: 0.2435 - val_accuracy: 0.9399
    Epoch 34/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2859 - accuracy: 0.9033 - val_loss: 0.5122 - val_accuracy: 0.9014
    Epoch 35/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2754 - accuracy: 0.9092 - val_loss: 0.2549 - val_accuracy: 0.9355
    Epoch 36/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2737 - accuracy: 0.9131 - val_loss: 0.1138 - val_accuracy: 0.9702
    Epoch 37/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.7036 - accuracy: 0.8945 - val_loss: 0.0951 - val_accuracy: 0.9731
    Epoch 38/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2683 - accuracy: 0.9146 - val_loss: 0.3257 - val_accuracy: 0.9399
    Epoch 39/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2551 - accuracy: 0.9199 - val_loss: 1.4844 - val_accuracy: 0.7451
    Epoch 40/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2546 - accuracy: 0.9199 - val_loss: 0.3594 - val_accuracy: 0.9321
    Epoch 41/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2443 - accuracy: 0.9287 - val_loss: 0.1398 - val_accuracy: 0.9648
    Epoch 42/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2419 - accuracy: 0.9268 - val_loss: 0.0864 - val_accuracy: 0.9785
    Epoch 43/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2358 - accuracy: 0.9302 - val_loss: 0.1259 - val_accuracy: 0.9678
    Epoch 44/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2280 - accuracy: 0.9312 - val_loss: 0.2786 - val_accuracy: 0.9346
    Epoch 45/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.2235 - accuracy: 0.9380 - val_loss: 0.0641 - val_accuracy: 0.9849
    Epoch 46/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2245 - accuracy: 0.9355 - val_loss: 0.1312 - val_accuracy: 0.9653
    Epoch 47/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2128 - accuracy: 0.9443 - val_loss: 0.2231 - val_accuracy: 0.9507
    Epoch 48/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.2037 - accuracy: 0.9453 - val_loss: 0.6299 - val_accuracy: 0.8945
    Epoch 49/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1976 - accuracy: 0.9448 - val_loss: 0.3269 - val_accuracy: 0.9414
    Epoch 50/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1979 - accuracy: 0.9448 - val_loss: 0.0764 - val_accuracy: 0.9814
    Epoch 51/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.2015 - accuracy: 0.9463 - val_loss: 0.0762 - val_accuracy: 0.9800
    Epoch 52/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.1952 - accuracy: 0.9502 - val_loss: 0.0729 - val_accuracy: 0.9800
    Epoch 53/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1890 - accuracy: 0.9463 - val_loss: 0.0707 - val_accuracy: 0.9839
    Epoch 54/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1836 - accuracy: 0.9521 - val_loss: 0.1235 - val_accuracy: 0.9673
    Epoch 55/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1842 - accuracy: 0.9531 - val_loss: 0.1173 - val_accuracy: 0.9673
    Epoch 56/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1877 - accuracy: 0.9531 - val_loss: 0.1429 - val_accuracy: 0.9658
    Epoch 57/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1749 - accuracy: 0.9551 - val_loss: 0.1072 - val_accuracy: 0.9727
    Epoch 58/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1738 - accuracy: 0.9556 - val_loss: 0.0934 - val_accuracy: 0.9780
    Epoch 59/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1714 - accuracy: 0.9590 - val_loss: 0.0945 - val_accuracy: 0.9746
    Epoch 60/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1744 - accuracy: 0.9556 - val_loss: 0.0776 - val_accuracy: 0.9829
    Epoch 61/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1664 - accuracy: 0.9629 - val_loss: 0.0710 - val_accuracy: 0.9829
    Epoch 62/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.1700 - accuracy: 0.9590 - val_loss: 0.1229 - val_accuracy: 0.9707
    Epoch 63/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.1653 - accuracy: 0.9658 - val_loss: 0.0493 - val_accuracy: 0.9873
    Epoch 64/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1655 - accuracy: 0.9595 - val_loss: 0.0564 - val_accuracy: 0.9858
    Epoch 65/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1567 - accuracy: 0.9663 - val_loss: 0.0635 - val_accuracy: 0.9854
    Epoch 66/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1571 - accuracy: 0.9648 - val_loss: 0.1680 - val_accuracy: 0.9600
    Epoch 67/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.1577 - accuracy: 0.9663 - val_loss: 0.1139 - val_accuracy: 0.9731
    Epoch 68/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.1498 - accuracy: 0.9712 - val_loss: 0.0869 - val_accuracy: 0.9795
    Epoch 69/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1501 - accuracy: 0.9673 - val_loss: 0.1953 - val_accuracy: 0.9556
    Epoch 70/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.1473 - accuracy: 0.9688 - val_loss: 0.1388 - val_accuracy: 0.9668
    Epoch 71/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.1456 - accuracy: 0.9692 - val_loss: 0.0661 - val_accuracy: 0.9829
    Epoch 72/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1454 - accuracy: 0.9692 - val_loss: 0.1254 - val_accuracy: 0.9707
    Epoch 73/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1417 - accuracy: 0.9697 - val_loss: 0.0712 - val_accuracy: 0.9839
    Epoch 74/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1428 - accuracy: 0.9697 - val_loss: 0.0884 - val_accuracy: 0.9775
    Epoch 75/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1437 - accuracy: 0.9702 - val_loss: 0.2098 - val_accuracy: 0.9521
    Epoch 76/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1375 - accuracy: 0.9731 - val_loss: 0.1747 - val_accuracy: 0.9614
    Epoch 77/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1359 - accuracy: 0.9712 - val_loss: 0.0651 - val_accuracy: 0.9829
    Epoch 78/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1318 - accuracy: 0.9751 - val_loss: 0.0816 - val_accuracy: 0.9819
    Epoch 79/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1350 - accuracy: 0.9731 - val_loss: 0.0674 - val_accuracy: 0.9829
    Epoch 80/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.1268 - accuracy: 0.9756 - val_loss: 0.0671 - val_accuracy: 0.9819
    Epoch 81/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1304 - accuracy: 0.9727 - val_loss: 0.0389 - val_accuracy: 0.9907
    Epoch 82/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1266 - accuracy: 0.9756 - val_loss: 0.1327 - val_accuracy: 0.9678
    Epoch 83/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1309 - accuracy: 0.9756 - val_loss: 0.0610 - val_accuracy: 0.9868
    Epoch 84/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1265 - accuracy: 0.9771 - val_loss: 0.0895 - val_accuracy: 0.9746
    Epoch 85/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1246 - accuracy: 0.9756 - val_loss: 0.0714 - val_accuracy: 0.9805
    Epoch 86/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.1212 - accuracy: 0.9795 - val_loss: 0.0567 - val_accuracy: 0.9868
    Epoch 87/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1237 - accuracy: 0.9785 - val_loss: 0.0724 - val_accuracy: 0.9814
    Epoch 88/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1201 - accuracy: 0.9775 - val_loss: 0.0659 - val_accuracy: 0.9854
    Epoch 89/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1225 - accuracy: 0.9780 - val_loss: 0.2939 - val_accuracy: 0.9414
    Epoch 90/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1202 - accuracy: 0.9771 - val_loss: 0.0428 - val_accuracy: 0.9888
    Epoch 91/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1172 - accuracy: 0.9795 - val_loss: 0.0800 - val_accuracy: 0.9780
    Epoch 92/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.1228 - accuracy: 0.9775 - val_loss: 0.0418 - val_accuracy: 0.9893
    Epoch 93/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.1169 - accuracy: 0.9780 - val_loss: 0.0622 - val_accuracy: 0.9829
    Epoch 94/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1156 - accuracy: 0.9785 - val_loss: 0.0602 - val_accuracy: 0.9839
    Epoch 95/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1153 - accuracy: 0.9780 - val_loss: 0.0472 - val_accuracy: 0.9878
    Epoch 96/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1139 - accuracy: 0.9795 - val_loss: 0.0745 - val_accuracy: 0.9814
    Epoch 97/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.1136 - accuracy: 0.9795 - val_loss: 0.0613 - val_accuracy: 0.9854
    Epoch 98/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.1146 - accuracy: 0.9785 - val_loss: 0.0897 - val_accuracy: 0.9766
    Epoch 99/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.1154 - accuracy: 0.9810 - val_loss: 0.0594 - val_accuracy: 0.9839
    Epoch 100/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.1118 - accuracy: 0.9810 - val_loss: 0.0484 - val_accuracy: 0.9868
    Epoch 101/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1105 - accuracy: 0.9795 - val_loss: 0.0492 - val_accuracy: 0.9878
    Epoch 102/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1083 - accuracy: 0.9814 - val_loss: 0.1024 - val_accuracy: 0.9746
    Epoch 103/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1097 - accuracy: 0.9810 - val_loss: 0.0500 - val_accuracy: 0.9893
    Epoch 104/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1125 - accuracy: 0.9829 - val_loss: 0.0525 - val_accuracy: 0.9858
    Epoch 105/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1061 - accuracy: 0.9839 - val_loss: 0.0428 - val_accuracy: 0.9893
    Epoch 106/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1058 - accuracy: 0.9824 - val_loss: 0.0961 - val_accuracy: 0.9780
    Epoch 107/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1083 - accuracy: 0.9814 - val_loss: 0.0431 - val_accuracy: 0.9893
    Epoch 108/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1027 - accuracy: 0.9810 - val_loss: 0.0356 - val_accuracy: 0.9922
    Epoch 109/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1055 - accuracy: 0.9814 - val_loss: 0.0856 - val_accuracy: 0.9829
    Epoch 110/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1061 - accuracy: 0.9834 - val_loss: 0.0715 - val_accuracy: 0.9829
    Epoch 111/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1087 - accuracy: 0.9829 - val_loss: 0.0529 - val_accuracy: 0.9868
    Epoch 112/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1043 - accuracy: 0.9834 - val_loss: 0.0689 - val_accuracy: 0.9834
    Epoch 113/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1065 - accuracy: 0.9814 - val_loss: 0.0445 - val_accuracy: 0.9873
    Epoch 114/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1000 - accuracy: 0.9858 - val_loss: 0.0406 - val_accuracy: 0.9907
    Epoch 115/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1013 - accuracy: 0.9854 - val_loss: 0.0462 - val_accuracy: 0.9902
    Epoch 116/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0994 - accuracy: 0.9854 - val_loss: 0.0453 - val_accuracy: 0.9902
    Epoch 117/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1004 - accuracy: 0.9854 - val_loss: 0.0591 - val_accuracy: 0.9839
    Epoch 118/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0982 - accuracy: 0.9844 - val_loss: 0.0666 - val_accuracy: 0.9819
    Epoch 119/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1003 - accuracy: 0.9839 - val_loss: 0.0589 - val_accuracy: 0.9854
    Epoch 120/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1010 - accuracy: 0.9844 - val_loss: 0.1052 - val_accuracy: 0.9727
    Epoch 121/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1027 - accuracy: 0.9844 - val_loss: 0.0871 - val_accuracy: 0.9775
    Epoch 122/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0989 - accuracy: 0.9834 - val_loss: 0.0420 - val_accuracy: 0.9878
    Epoch 123/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0983 - accuracy: 0.9839 - val_loss: 0.0665 - val_accuracy: 0.9829
    Epoch 124/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.1021 - accuracy: 0.9858 - val_loss: 0.0922 - val_accuracy: 0.9775
    Epoch 125/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0980 - accuracy: 0.9863 - val_loss: 0.0728 - val_accuracy: 0.9829
    Epoch 126/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.0931 - accuracy: 0.9883 - val_loss: 0.0586 - val_accuracy: 0.9868
    Epoch 127/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0983 - accuracy: 0.9868 - val_loss: 0.0681 - val_accuracy: 0.9819
    Epoch 128/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0977 - accuracy: 0.9854 - val_loss: 0.0676 - val_accuracy: 0.9839
    Epoch 129/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0942 - accuracy: 0.9858 - val_loss: 0.0552 - val_accuracy: 0.9868
    Epoch 130/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0905 - accuracy: 0.9883 - val_loss: 0.0417 - val_accuracy: 0.9912
    Epoch 131/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0909 - accuracy: 0.9863 - val_loss: 0.0726 - val_accuracy: 0.9829
    Epoch 132/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0961 - accuracy: 0.9829 - val_loss: 0.1077 - val_accuracy: 0.9722
    Epoch 133/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0903 - accuracy: 0.9858 - val_loss: 0.2468 - val_accuracy: 0.9429
    Epoch 134/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0920 - accuracy: 0.9883 - val_loss: 0.0690 - val_accuracy: 0.9805
    Epoch 135/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.0893 - accuracy: 0.9868 - val_loss: 0.0425 - val_accuracy: 0.9888
    Epoch 136/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0895 - accuracy: 0.9878 - val_loss: 0.0891 - val_accuracy: 0.9805
    Epoch 137/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0898 - accuracy: 0.9868 - val_loss: 0.0919 - val_accuracy: 0.9775
    Epoch 138/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0883 - accuracy: 0.9888 - val_loss: 0.1011 - val_accuracy: 0.9727
    Epoch 139/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0942 - accuracy: 0.9868 - val_loss: 0.0585 - val_accuracy: 0.9839
    Epoch 140/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0934 - accuracy: 0.9858 - val_loss: 0.0363 - val_accuracy: 0.9893
    Epoch 141/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0916 - accuracy: 0.9878 - val_loss: 0.0486 - val_accuracy: 0.9868
    Epoch 142/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0858 - accuracy: 0.9893 - val_loss: 0.0602 - val_accuracy: 0.9849
    Epoch 143/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0873 - accuracy: 0.9888 - val_loss: 0.0640 - val_accuracy: 0.9819
    Epoch 144/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0856 - accuracy: 0.9883 - val_loss: 0.0693 - val_accuracy: 0.9834
    Epoch 145/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0882 - accuracy: 0.9893 - val_loss: 0.0411 - val_accuracy: 0.9907
    Epoch 146/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0901 - accuracy: 0.9893 - val_loss: 0.0585 - val_accuracy: 0.9858
    Epoch 147/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0879 - accuracy: 0.9858 - val_loss: 0.0751 - val_accuracy: 0.9800
    Epoch 148/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0852 - accuracy: 0.9868 - val_loss: 0.0741 - val_accuracy: 0.9829
    Epoch 149/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0867 - accuracy: 0.9893 - val_loss: 0.0663 - val_accuracy: 0.9829
    Epoch 150/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0895 - accuracy: 0.9878 - val_loss: 0.0420 - val_accuracy: 0.9888
    Epoch 151/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0834 - accuracy: 0.9883 - val_loss: 0.0622 - val_accuracy: 0.9829
    Epoch 152/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.0854 - accuracy: 0.9897 - val_loss: 0.0764 - val_accuracy: 0.9814
    Epoch 153/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0870 - accuracy: 0.9897 - val_loss: 0.0466 - val_accuracy: 0.9878
    Epoch 154/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0858 - accuracy: 0.9883 - val_loss: 0.0455 - val_accuracy: 0.9888
    Epoch 155/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0825 - accuracy: 0.9907 - val_loss: 0.0529 - val_accuracy: 0.9868
    Epoch 156/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0867 - accuracy: 0.9893 - val_loss: 0.0386 - val_accuracy: 0.9888
    Epoch 157/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0840 - accuracy: 0.9888 - val_loss: 0.0981 - val_accuracy: 0.9766
    Epoch 158/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0842 - accuracy: 0.9893 - val_loss: 0.0884 - val_accuracy: 0.9775
    Epoch 159/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0854 - accuracy: 0.9883 - val_loss: 0.0492 - val_accuracy: 0.9868
    Epoch 160/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0820 - accuracy: 0.9897 - val_loss: 0.0524 - val_accuracy: 0.9868
    Epoch 161/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.0819 - accuracy: 0.9883 - val_loss: 0.0482 - val_accuracy: 0.9878
    Epoch 162/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0806 - accuracy: 0.9897 - val_loss: 0.0459 - val_accuracy: 0.9902
    Epoch 163/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0799 - accuracy: 0.9888 - val_loss: 0.0526 - val_accuracy: 0.9854
    Epoch 164/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.0824 - accuracy: 0.9888 - val_loss: 0.0438 - val_accuracy: 0.9893
    Epoch 165/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.0798 - accuracy: 0.9922 - val_loss: 0.0508 - val_accuracy: 0.9878
    Epoch 166/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.0803 - accuracy: 0.9912 - val_loss: 0.0590 - val_accuracy: 0.9854
    Epoch 167/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0800 - accuracy: 0.9888 - val_loss: 0.0434 - val_accuracy: 0.9878
    Epoch 168/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0854 - accuracy: 0.9883 - val_loss: 0.0659 - val_accuracy: 0.9858
    Epoch 169/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0792 - accuracy: 0.9888 - val_loss: 0.0880 - val_accuracy: 0.9785
    Epoch 170/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.0817 - accuracy: 0.9893 - val_loss: 0.0378 - val_accuracy: 0.9912
    Epoch 171/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0780 - accuracy: 0.9907 - val_loss: 0.0333 - val_accuracy: 0.9932
    Epoch 172/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0804 - accuracy: 0.9912 - val_loss: 0.0579 - val_accuracy: 0.9858
    Epoch 173/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0834 - accuracy: 0.9893 - val_loss: 0.0360 - val_accuracy: 0.9907
    Epoch 174/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0801 - accuracy: 0.9912 - val_loss: 0.0522 - val_accuracy: 0.9858
    Epoch 175/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0803 - accuracy: 0.9893 - val_loss: 0.0392 - val_accuracy: 0.9907
    Epoch 176/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0806 - accuracy: 0.9912 - val_loss: 0.0479 - val_accuracy: 0.9868
    Epoch 177/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0764 - accuracy: 0.9897 - val_loss: 0.1615 - val_accuracy: 0.9604
    Epoch 178/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0756 - accuracy: 0.9897 - val_loss: 0.0560 - val_accuracy: 0.9854
    Epoch 179/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0810 - accuracy: 0.9912 - val_loss: 0.0604 - val_accuracy: 0.9829
    Epoch 180/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0819 - accuracy: 0.9907 - val_loss: 0.0416 - val_accuracy: 0.9888
    Epoch 181/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0760 - accuracy: 0.9907 - val_loss: 0.0545 - val_accuracy: 0.9868
    Epoch 182/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0789 - accuracy: 0.9912 - val_loss: 0.0495 - val_accuracy: 0.9878
    Epoch 183/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0801 - accuracy: 0.9897 - val_loss: 0.0476 - val_accuracy: 0.9878
    Epoch 184/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0757 - accuracy: 0.9907 - val_loss: 0.0815 - val_accuracy: 0.9814
    Epoch 185/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0764 - accuracy: 0.9897 - val_loss: 0.0598 - val_accuracy: 0.9868
    Epoch 186/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0754 - accuracy: 0.9912 - val_loss: 0.0441 - val_accuracy: 0.9878
    Epoch 187/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0767 - accuracy: 0.9917 - val_loss: 0.0746 - val_accuracy: 0.9800
    Epoch 188/200
    211/211 [==============================] - 4s 18ms/step - loss: 0.0768 - accuracy: 0.9917 - val_loss: 0.0426 - val_accuracy: 0.9907
    Epoch 189/200
    211/211 [==============================] - 4s 18ms/step - loss: 0.0771 - accuracy: 0.9912 - val_loss: 0.0435 - val_accuracy: 0.9888
    Epoch 190/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0761 - accuracy: 0.9917 - val_loss: 0.0504 - val_accuracy: 0.9878
    Epoch 191/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0751 - accuracy: 0.9912 - val_loss: 0.0540 - val_accuracy: 0.9854
    Epoch 192/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0766 - accuracy: 0.9897 - val_loss: 0.0405 - val_accuracy: 0.9912
    Epoch 193/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0741 - accuracy: 0.9907 - val_loss: 0.0518 - val_accuracy: 0.9868
    Epoch 194/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0768 - accuracy: 0.9907 - val_loss: 0.0494 - val_accuracy: 0.9858
    Epoch 195/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0760 - accuracy: 0.9888 - val_loss: 0.0398 - val_accuracy: 0.9888
    Epoch 196/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0761 - accuracy: 0.9897 - val_loss: 0.0312 - val_accuracy: 0.9922
    Epoch 197/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0712 - accuracy: 0.9912 - val_loss: 0.0319 - val_accuracy: 0.9932
    Epoch 198/200
    211/211 [==============================] - 4s 17ms/step - loss: 0.0749 - accuracy: 0.9917 - val_loss: 0.0335 - val_accuracy: 0.9927
    Epoch 199/200
    211/211 [==============================] - 3s 17ms/step - loss: 0.0765 - accuracy: 0.9893 - val_loss: 0.0735 - val_accuracy: 0.9814
    Epoch 200/200
    211/211 [==============================] - 3s 16ms/step - loss: 0.0770 - accuracy: 0.9912 - val_loss: 0.0471 - val_accuracy: 0.9868





    <tensorflow.python.keras.callbacks.History at 0x7f484139fd68>



<h2> Model Evaluation </h2>

Evaluate the model on the test set to see how good the network has been trained.


*  Having *batch_size=256* and *epochs=200*, the model should achieve ~99.2-99.5% accuracy on the test set




```
score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss: {:.4f}".format(score[0]))
print("Test accuracy: {:.2f}%".format(score[1]*100))
```

    313/313 [==============================] - 1s 4ms/step - loss: 0.0470 - accuracy: 0.9922
    Test loss: 0.0470
    Test accuracy: 99.22%


<h2> Split The Network </2>

After all, we want to split the network from a point called <b> split point </b>.
In the *Model architecture design* section, I named a specific layer (the second Add layer) as <b>'split'</b>. I want to split the model from that point now. Of course, you can choose a different point. It's on you!

* Make sure you have read the split_network document to have enough understanding about its arguments.
* After applying split_network, we have two models, head and tail, each one works separately.
* The split point itself belongs to the head model. You can move it to the tail model by turning the *on_head* argument into False.


```
# Find custom layers that are not pure Keras layers/objects, but brought directly by Tensorflow backend.
custom_layers = {}
for l in model.layers:
  name = l.name
  if "relu6" in name or "swish" in name:
    custom_layers[name] = l

# See what these custom layers are.
print(custom_layers)

# Get head and tail models after apllying split_network. Arguments are explained. Make sure you have read its document.
head, tail = splitter.split_network(model=model, 
                                    split_layer_name="split",
                                    on_head=True, 
                                    names=("head_model", "tail_model"),
                                    custom_objects=custom_layers)
```

    {'relu6_3': <tensorflow.python.keras.layers.core.Activation object at 0x7f484557a588>, 'relu6_4': <tensorflow.python.keras.layers.core.Activation object at 0x7f484022be80>, 'relu6_5': <tensorflow.python.keras.layers.core.Activation object at 0x7f4842a26cf8>, 'relu6_6': <tensorflow.python.keras.layers.core.Activation object at 0x7f48415fdc88>, 'relu6_7': <tensorflow.python.keras.layers.core.Activation object at 0x7f48425ef128>}


<h3> Let's see how these head and tail models are </h3>


```
_ = head.summary(line_length=100), tail.summary(line_length=100)
```

    Model: "head_model"
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    input (InputLayer)               [(None, 28, 28, 1)]   0                                            
    ____________________________________________________________________________________________________
    augmentation (Sequential)        (None, 28, None, 1)   0           input[0][0]                      
    ____________________________________________________________________________________________________
    conv_1 (Conv2D)                  (None, 28, 28, 16)    160         augmentation[0][0]               
    ____________________________________________________________________________________________________
    bn_1 (BatchNormalization)        (None, 28, 28, 16)    64          conv_1[0][0]                     
    ____________________________________________________________________________________________________
    relu_1 (Activation)              (None, 28, 28, 16)    0           bn_1[0][0]                       
    ____________________________________________________________________________________________________
    drop_1 (Dropout)                 (None, 28, 28, 16)    0           relu_1[0][0]                     
    ____________________________________________________________________________________________________
    conv_2 (Conv2D)                  (None, 28, 28, 32)    4640        drop_1[0][0]                     
    ____________________________________________________________________________________________________
    bn_2 (BatchNormalization)        (None, 28, 28, 32)    128         conv_2[0][0]                     
    ____________________________________________________________________________________________________
    relu_2 (Activation)              (None, 28, 28, 32)    0           bn_2[0][0]                       
    ____________________________________________________________________________________________________
    drop_2 (Dropout)                 (None, 28, 28, 32)    0           relu_2[0][0]                     
    ____________________________________________________________________________________________________
    maxp_2 (MaxPooling2D)            (None, 9, 9, 32)      0           drop_2[0][0]                     
    ____________________________________________________________________________________________________
    conv_3 (Conv2D)                  (None, 9, 9, 32)      9248        maxp_2[0][0]                     
    ____________________________________________________________________________________________________
    bn_3 (BatchNormalization)        (None, 9, 9, 32)      128         conv_3[0][0]                     
    ____________________________________________________________________________________________________
    relu6_3 (Activation)             (None, 9, 9, 32)      0           bn_3[0][0]                       
    ____________________________________________________________________________________________________
    drop_3 (Dropout)                 (None, 9, 9, 32)      0           relu6_3[0][0]                    
    ____________________________________________________________________________________________________
    conv_4 (Conv2D)                  (None, 9, 9, 32)      9248        drop_3[0][0]                     
    ____________________________________________________________________________________________________
    bn_4 (BatchNormalization)        (None, 9, 9, 32)      128         conv_4[0][0]                     
    ____________________________________________________________________________________________________
    relu6_4 (Activation)             (None, 9, 9, 32)      0           bn_4[0][0]                       
    ____________________________________________________________________________________________________
    drop_4 (Dropout)                 (None, 9, 9, 32)      0           relu6_4[0][0]                    
    ____________________________________________________________________________________________________
    add_1 (Add)                      (None, 9, 9, 32)      0           drop_4[0][0]                     
                                                                       maxp_2[0][0]                     
    ____________________________________________________________________________________________________
    conv_5 (Conv2D)                  (None, 9, 9, 32)      9248        add_1[0][0]                      
    ____________________________________________________________________________________________________
    bn_5 (BatchNormalization)        (None, 9, 9, 32)      128         conv_5[0][0]                     
    ____________________________________________________________________________________________________
    relu6_5 (Activation)             (None, 9, 9, 32)      0           bn_5[0][0]                       
    ____________________________________________________________________________________________________
    drop_5 (Dropout)                 (None, 9, 9, 32)      0           relu6_5[0][0]                    
    ____________________________________________________________________________________________________
    conv_6 (Conv2D)                  (None, 9, 9, 32)      9248        drop_5[0][0]                     
    ____________________________________________________________________________________________________
    bn_6 (BatchNormalization)        (None, 9, 9, 32)      128         conv_6[0][0]                     
    ____________________________________________________________________________________________________
    relu6_6 (Activation)             (None, 9, 9, 32)      0           bn_6[0][0]                       
    ____________________________________________________________________________________________________
    drop_6 (Dropout)                 (None, 9, 9, 32)      0           relu6_6[0][0]                    
    ____________________________________________________________________________________________________
    split (Add)                      (None, 9, 9, 32)      0           drop_6[0][0]                     
                                                                       add_1[0][0]                      
    ====================================================================================================
    Total params: 42,496
    Trainable params: 42,144
    Non-trainable params: 352
    ____________________________________________________________________________________________________
    Model: "tail_model"
    ____________________________________________________________________________________________________
    Layer (type)                                 Output Shape                            Param #        
    ====================================================================================================
    input_3 (InputLayer)                         [(None, 9, 9, 32)]                      0              
    ____________________________________________________________________________________________________
    conv_7 (Conv2D)                              (None, 9, 9, 64)                        18496          
    ____________________________________________________________________________________________________
    bn_7 (BatchNormalization)                    (None, 9, 9, 64)                        256            
    ____________________________________________________________________________________________________
    relu6_7 (Activation)                         (None, 9, 9, 64)                        0              
    ____________________________________________________________________________________________________
    drop_7 (Dropout)                             (None, 9, 9, 64)                        0              
    ____________________________________________________________________________________________________
    Flatten (Flatten)                            (None, 5184)                            0              
    ____________________________________________________________________________________________________
    dense_1 (Dense)                              (None, 128)                             663680         
    ____________________________________________________________________________________________________
    last_drop (Dropout)                          (None, 128)                             0              
    ____________________________________________________________________________________________________
    output (Dense)                               (None, 10)                              1290           
    ====================================================================================================
    Total params: 683,722
    Trainable params: 683,594
    Non-trainable params: 128
    ____________________________________________________________________________________________________


<h2> Do Inference by both models </h2>



<h3> Get an image for test </h3>


```
plt.imshow(x_test[0].reshape(28,28).astype(np.float), cmap='gray');plt.axis("off");plt.show()
```


![png](split_model_example_files/split_model_example_24_0.png)


<h3> Firstly, feed image to the <b>head</b> model. Then feed the output prediction of the head into the <b>tail</b> model.

<h3> At last, Let's see whether the final prediction is correct?! </h3>


```
# Create batch size 1
tst = np.expand_dims(x_test[0], 0)

# Do inference by head model
head_pred = head.predict(tst)
print("Output shape of the head model: ",head_pred.shape)

# Do inference by tail model
tail_pred = tail.predict(head_pred)
print("Output shape of the tail model: ",tail_pred.shape)

# Check whether the output is correct or not.
print("Predicted Label:", np.argmax(tail_pred))
```

    Output shape of the head model:  (1, 9, 9, 32)
    Output shape of the tail model:  (1, 10)
    Predicted Label: 7


<h1 dir='rtl' align='center'>YUP :D</h1>

<h3> Save splitted models </h3>



```
head.save("head_model", save_format='tf')
tail.save("tail_model", save_format='tf')
```

    INFO:tensorflow:Assets written to: head_model/assets
    INFO:tensorflow:Assets written to: tail_model/assets


<h2> Merge Head and Tail <h2>

<h3> Merging these two seperated models is too simple! </h3>


```
merged_model = keras.Sequential([head, tail], name="merged_model")
merged_model.summary()

```

    Model: "merged_model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    head_model (Functional)      (None, 9, 9, 32)          42496     
    _________________________________________________________________
    tail_model (Functional)      (None, 10)                683722    
    =================================================================
    Total params: 726,218
    Trainable params: 725,738
    Non-trainable params: 480
    _________________________________________________________________


<h2> Practice: </h2>



1.   The model below is a VGG-style model designed for MNIST dataset. Split it into two parts.
2.   Split a model into more than two parts.




```

# Design a VGG-like digit recognition model
model = keras.Sequential(
    [
      keras.Input(shape=input_shape, name="input"),
      KL.Dropout(0.001, name="drop_0"),
      KL.experimental.preprocessing.RandomZoom((-factor, factor), name="zoom"),
      KL.experimental.preprocessing.RandomTranslation((-factor, factor), (-factor, factor), name="translate"),
      KL.experimental.preprocessing.RandomRotation((-factor, factor), name="rotate"),
      KL.experimental.preprocessing.RandomContrast(factor, name="contrast"),
     
      KL.Conv2D(32, kernel_size=(3, 3), strides=(1,1), name="conv_1"),
      KL.BatchNormalization(name="bn_1"),
      KL.Activation(tf.nn.relu6, name="act_1"),
      # KL.Lambda(relu6, name="act_1"),
      KL.Dropout(0.2, name="drop_1"),
      KL.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), name="conv_2"),
      KL.BatchNormalization(name="bn_2"),
      KL.Activation(tf.nn.relu6, name="act_2"),
      KL.MaxPooling2D(pool_size=(2, 2), name="split"),

      KL.Flatten(name="flatten"),
      KL.Dropout(0.25, name="drop_3"),
      KL.Dense(64, activation="relu", name="dense_1"),
      KL.Dropout(.2, name="drop_4"),
      KL.Dense(num_classes, activation="softmax", name="output"),
    ]
)

# Print a summary of the model
# model.summary(line_length=150)

```

<h1> END! </h1>

Any kind of contribution is appreciated.

Github: [github.com/nrasadi/split-keras-tensorflow-model](https://github.com/nrasadi/split-keras-tensorflow-model)
