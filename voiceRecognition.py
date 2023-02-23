from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU , Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Input
#import numpy as np
#from playsound import playsound
import sounddevice as sound
#from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import wave
#import sys
import random
from enum import Enum
from tensorflow.keras.callbacks import TensorBoard
from scipy.io.wavfile import read
from preprocessData import LoadAndPreProcess
from scipy.fft import fft, fftfreq
#function find the first non zero index
#function divide into index precentage
#normalive dataset
#from skrf.plotting import plot


def createDatasets(signal1,signal2,interval):
    startIndex = 8500
    Index1 = 0
    Index2 = 0
    length = len(signal1)
    NOF_BATCHES =  round(length / interval)
    labelSignal = np.array([],dtype=int)
    #fixed size block with single label signal[i].shape = 1000 labelSignal[i] = 1
    # add to list by random
    unifiedSignal = np.array([])
    i1 = 0
    i2 = 0
    for i in range(NOF_BATCHES*2):
        classifier = np.array([random.randint(0, 1)],dtype=int)

        if (classifier == 0 and (i1+1)*interval< len(signal1)):
            unifiedSignal = np.concatenate((unifiedSignal , signal1[i1*interval:(i1+1)*interval]),axis=0)
            i1 = i1 + 1
            labelSignal = np.concatenate((labelSignal, classifier), axis=0)
        if (classifier == 1 and (i2+1)*interval< len(signal2)):
            unifiedSignal = np.concatenate((unifiedSignal , signal2[i2*interval:(i2+1)*interval]),axis=0)
            i2 = i2 + 1
            labelSignal = np.concatenate((labelSignal, classifier), axis=0)

    return [unifiedSignal, labelSignal]

def categorize(signal1, signal2 , category1,category2, BATCH_LEN,Fs,loadData):
    length1 = (int(len(signal1)/BATCH_LEN))*BATCH_LEN;
    x_train1 = np.split(signal1[0:length1],int(len(signal1)/BATCH_LEN));
    #y_train1 = np.full(len(x_train1),category1)

    len2 = (int(len(signal2)/BATCH_LEN))*BATCH_LEN;
    x_train2 = np.split(signal2[0:len2],int(len(signal2)/BATCH_LEN))
    #y_train2 = np.full(len(x_train2),category2)
    # list to np.array : array = np.array(lst)
    ONLY_POSITIVE_FREQS = 1
    TRAINING_SET_LEN = 190
    x_train1_Filtered ,x_train2_Filtered =loadData.signalEnergyPlot(x_train1,x_train2)  # filter only high energy signals to remove noise
    
    x_train =  np.empty((0,int(BATCH_LEN/ONLY_POSITIVE_FREQS)))
    x_train_phase =  np.empty((0,int(BATCH_LEN/ONLY_POSITIVE_FREQS)))
    x_label =  np.empty((0,1))
    c1_idx = 0
    c2_idx = 0
    for indx in range(length1+len2):
        classifier = np.array([random.randint(0, 1)],dtype=int)
        
        if (classifier==0 and c1_idx<len(x_train1_Filtered)):
            fftSignal = abs(fft(x_train1_Filtered[c1_idx]))
            fftphase = np.unwrap(np.angle(fft(x_train1_Filtered[c1_idx])))
            fftSignal = fftSignal[-int(len(fftSignal)/ONLY_POSITIVE_FREQS):]
            fftphase = fftphase[-int(len(fftphase)/ONLY_POSITIVE_FREQS):]
            x_train = np.vstack((x_train,(np.array(fftSignal))))
            x_train_phase = np.vstack((x_train_phase,np.array(fftphase)))
            x_label = np.vstack((x_label, np.array([category1])))
            c1_idx = c1_idx +1
        if (classifier ==1 and c2_idx<len(x_train2_Filtered)):
            fftSignal =  abs(fft(x_train2_Filtered[c2_idx]))
            fftphase = np.unwrap(np.angle(fft(x_train2_Filtered[c2_idx])))
            fftSignal = fftSignal[-int(len(fftSignal)/ONLY_POSITIVE_FREQS):]
            fftphase = fftphase[-int(len(fftphase)/ONLY_POSITIVE_FREQS):]
            x_train = np.vstack((x_train,(np.array(fftSignal))))
            x_train_phase = np.vstack((x_train_phase,np.array(fftphase)))   
            x_label = np.vstack((x_label, np.array([category2])))
            c2_idx = c2_idx +1

    #x_train = np.concatenate(((np.array(x_train1_Filtered)),(np.array(x_train2))),axis=0)
    #y_train = np.concatenate(((y_train1.reshape(len(y_train1),1)),(y_train2.reshape(len(y_train2),1))),axis=0)
   
    loadData.plotDataset(x_train,x_train_phase,Fs)

    return x_train[0:TRAINING_SET_LEN], x_train_phase[0:TRAINING_SET_LEN] , x_label[0:TRAINING_SET_LEN],x_train[TRAINING_SET_LEN:],x_train_phase[TRAINING_SET_LEN:] , x_label[TRAINING_SET_LEN:]

def openFile():
    spf = read("G:\\Old_Disk\\Documents\\Develpos\\keras\\PythonDL\\VoiceRecognition\\Untitled.wav", "r")
# Extract Raw Audio from Wav File
    signal = np.array(spf[1],dtype=float)
    signal = spf.readframes(1200000)
    #signal = np.fromstring(signal, "Int16")
    fs = spf.getframerate()
    Time = np.linspace(0, len(signal) / fs, num=len(signal))
    plt.figure(1)
    plt.title("Signal Wave...")
    plt.plot(Time, signal)
    plt.show()
    return signal,fs

def showSignals(signal1,signal2 , fs):
    signal_len = min(len ( signal1 ),len ( signal2 ))
    Time = np.linspace ( 0, signal_len / fs, num=signal_len )
    plt.figure ( 1 )
    plt.title ( "Signal Wave..." )
    plt.subplot(2,1,1)
    plt.title('Pshisha')
    plt.plot ( Time, signal1[0:signal_len] ,'C1' )
    plt.subplot(2,1,2)
    plt.title ( 'Me' )
    plt.plot ( Time, signal2[0:signal_len], 'C2' )
    plt.show ()

def NormalizeSignal(signal):
    mean = signal.mean(axis=0)
    Nsignal = signal - mean
    std = Nsignal.std(axis=0)
    Psignal = Nsignal/std
    return Psignal

def plotLoss(history):
    plt.figure()
    plt.subplot(211)
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.subplot(212)
    acc = history.history['accuracy']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def build_LSTM_network(input_len):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(input_len, 1)))
    model.add(Dropout(0.5))
    #model.add ( LSTM (units= 30, activation='relu' ) )
    #model.add ( Dropout ( 0.5 ) )
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer= 'rmsprop', metrics=['accuracy'])
    return model


def create_model(input_length):
    print ('Creating model...')
    model = Sequential()
    model.add(Embedding(input_dim = 188, output_dim = 50, input_length = input_length))
    model.add(LSTM(units= 30,output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units= 30,output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1, activation='sigmoid'))

    print ('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def build_GRU_network(input_len):
    model = Sequential()
    model.add(GRU(50,  dropout=0.3,  recurrent_dropout=0.3,     input_shape=(input_len, 1))) # (None, float_data.shape[-1]), return_sequences=True,
    model.add(Dense(1))
    model.compile(loss='sparse_categorical_crossentropy', optimizer= 'rmsprop', metrics=['accuracy'])
    return model

def build_FullyConnected_network(input_len): #dense not convnet
    model = Sequential()
  #  model.add(GRU(100, activation='sigmoid', recurrent_activation='sigmoid',dropout=0.2,recurrent_dropout=0.2, input_shape=(input_len, 1)))

    model.add(Dense(units = 800, activation='relu',input_shape=(input_len, )))
    model.add(Dropout(0.1))
    model.add(Dense(units =500, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units =128, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer= 'rmsprop', metrics=['accuracy'])
    return model

def build_Phase_FullyConnected(input_len):
    model = Sequential()
    model.add(Dense(units = 400, activation='relu',input_shape=(input_len, )))
    model.add(Dropout(0.1))
    model.add(Dense(units =300, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units =128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer= 'rmsprop', metrics=['accuracy'])
    return model

def build_Fully_connected_network_Keras_API(input_len):
    model = Sequential()
    inputs = keras.Input(shape=(input_len,),name="my_input")
    layer1 = layer.Dense(units = 800,activation = 'relu')(inputs) 
    #layer2 = layer.De
    return model
#loadData = LoadAndPreProcess("G:\\Old_Disk\\Documents\\Develpos\\keras\\PythonDL\\VoiceRecognition\\Untitled.wav");
#[Fs , Signal] = loadData.openWavFile()
loadData = LoadAndPreProcess("VoiceSamples.txt");
[fs , signal] = loadData.openFile()
loadData.plotFFtSignal(signal,fs);
signal_d5 , Fs_d5= loadData.downSampleData(5)  # downSample by factor 5
loadData.plotFFtSignal(signal_d5 , Fs_d5)
#sound.play(signal_d5,Fs_d5)   # sound OK with sampled signal
[C, R] = loadData.getSections(signal_d5,Fs_d5)
#time domail signals
[x_train, x_train_phase, x_label, x_test, x_test_phase, x_label_test] = categorize(C, R, 0, 1, 1000 , Fs_d5,loadData);

##showSignals(signal1,signal2 , fs)
##sound.play(signal1,fs*2)
##sound.play(signal2,fs*2)
#


class Mode(Enum):
    LSTM = 1
    GRU = 2
    FULLY_CONNECTED = 3
    FULLY_CONNECTED_PHASE = 4

class Classification(Enum):
    KULFA = 0
    ME  = 1


callbacks = [
TensorBoard(
log_dir='log_dir',
histogram_freq=1,
embeddings_freq=1,
)
]

#x_label = x_label[:-1]
mode = Mode.FULLY_CONNECTED

if(mode == Mode.FULLY_CONNECTED):
    input_len = len(x_train[0])
    model = build_FullyConnected_network(input_len)
    history = model.fit(x_train, x_label, epochs=85, validation_split=0.2, verbose=1)
    test_metrics = model.evaluate(x_test, x_label_test)
    result = model.predict(x_test)

if(mode== Mode.FULLY_CONNECTED_PHASE):
    input_len = len(x_train_phase[0,500:])
    model = build_Phase_FullyConnected(input_len)
    history = model.fit(NormalizeSignal(x_train_phase[:,500:]), x_label, epochs=85, validation_split=0.2, verbose=1)
    model.predict(NormalizeSignal(x_test_phase[:,500:]))

if(mode == Mode.GRU):
     model = build_GRU_network(input_len)
     history = model.fit(x_train, x_label, epochs=5, validation_split=0.2, verbose=1)

#mode = Mode.LSTM
if(mode == Mode.LSTM):
     model = create_model(input_len)
     history = model.fit(x_train, x_label, epochs=5, validation_split=0.2, verbose=1)



plotLoss(history)
#test Signal


#TestNormalizedSignal = NormalizeSignal(TestunifiedSignal)

#x_test = TestNormalizedSignal.reshape(interval,len(TestunifiedLabel))
#x_testNorm = x_test.T

#result = model.predict((x_testNorm.reshape(len(x_testNorm),len(x_testNorm[0]),1)))



