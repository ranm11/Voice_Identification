import numpy as np
#from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import wave
import sounddevice as sound

class LoadAndPreProcess:
    
    wavFilePath='';
    signalStartindex='';

    def __init__(self,filepath):
        self.wavFilePath=filepath;
        self.signalStartindex = 63408;
        self.Fs=0;
        self.signal=np.array([0]); # whole signal interleave 
        self.t=0;
        self.Energy_threshold = 5;

    def openFile(self):
        with open(self.wavFilePath,'r') as file:
            signal = file.read().replace('\n',',')
        signal = signal.split(",")
        signal = signal[1:-1]
        self.signal = [float(i) for i in signal]
        self.Fs = 48000
        return  self.Fs , self.signal    

    def openWavFile(self):
        file = wave.open(self.wavFilePath)
        self.Fs = file.getframerate()
        self.NOF_Samples = file.getnframes()
        data = file.readframes(-1);
        channels = file.getnchannels();
        signal = np.frombuffer(data,np.int8)
        return self.Fs , self.signal

    def plotFFtSignal(self, Signal , Fs):
        t = np.linspace(0, len(Signal)/Fs, len(Signal));
        fftSignal = fft(Signal)
        fftSignal = abs(fftSignal)
        #f = np.linspace(0,(self.Fs), len(self.signal))
        f=fftfreq(len(Signal),1/Fs)
        plt.figure(1)
        #plt.title("Signal Wave")
        plt.subplot(311)
        plt.plot(t,Signal);
        plt.subplot(312)
        plt.plot(f,fftSignal)
        plt.subplot(313)
        plt.plot(f,np.unwrap(np.angle(fft(Signal))))


    def downSampleData(self, downSampleFactor):
        samples_decimated = int(len(self.signal)/downSampleFactor)
        t = np.linspace(0, len(self.signal)/self.Fs, len(self.signal));
        t_dwn = np.linspace(0,t[-1],samples_decimated,endpoint=False);
        sig_dwn = signal.decimate(self.signal,downSampleFactor)
        return sig_dwn , self.Fs/downSampleFactor

    def getSections(self, Signal_d5 , Fs_d5):
        #sound.play(Signal_d5,Fs_d5)
        #start = 12150;
        C1 = Signal_d5[12150:105000];
        R1 = Signal_d5[105000: 198000];
        C2 = Signal_d5[203000:322000];
        R2 = Signal_d5[325000:442000];
        C3 = Signal_d5[443000:602000];
        R3 = Signal_d5[606000:-100];
        C12 = np.concatenate((C1, C2), axis=0)
        C = np.concatenate( (C12  , C3),axis=0);
        R12 = np.concatenate((R1,R2),axis=0)
        R = np.concatenate((R12, R3),axis=0);
        #C  = [C1,C2,C3];
        #R = [R1,R2,R3];
        return C , R

    def plotDataset(self,train_images,train_phase,Fs):
        f=fftfreq(len(train_images[0]*2),1/Fs)
        #f = np.linspace(0, Fs/2, len(train_images[0]));
        #f=f[:-int((len(f)/2))]

        plt.figure(1)
        plt.subplot(441)
        plt.plot(f,train_images[0])
        plt.subplot(442)
        plt.plot(f,train_phase[0])
        plt.subplot(443)
        plt.plot(f,train_images[3])
        plt.subplot(444)
        plt.plot(f,train_phase[3])
        plt.subplot(445)
        plt.plot(f,train_images[5])
        plt.subplot(446)
        plt.plot(f,train_phase[5])
        plt.subplot(447)
        plt.plot(f,train_images[7])
        plt.subplot(448)
        plt.plot(f,train_phase[7])
        plt.subplot(449)
        plt.plot(f,train_images[9])

    def signalEnergyPlot(self,x_train1,x_train2):
        
        signal1_energy = np.array((0))
        signal2_energy = np.array((0))
        signal3_energy = np.array((0))
        for index_batch in range(len(x_train1)):
            signal1_energy = np.vstack((signal1_energy,sum(abs(x_train1[index_batch]))))  
        for index_batch in range(len(x_train2)):
            signal2_energy = np.vstack((signal2_energy,sum(abs(x_train2[index_batch]))))      
        return ((np.array(x_train1))[np.where(signal1_energy> self.Energy_threshold)[0]]) , ((np.array(x_train2))[np.where(signal2_energy>self.Energy_threshold)[0]])
        