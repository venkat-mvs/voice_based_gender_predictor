import pyaudio
import numpy as np
import pandas as pd
from scipy.stats import skew
from statistics import mode
import csv
import tkinter as tk
#from training import *
from sklearn.linear_model import LogisticRegression

def gen_trained_model(train_data_file='voice.csv'):
    data = pd.read_csv(train_data_file)
    """['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew',
    	'kurt', 'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun',
    	'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx']"""
    feature_labels=['mode','minfun','maxdom','Q25','Q75','IQR','meanfun','median','skew']
    x_data=data.loc[:,feature_labels]
    y=data.loc[:,'label']
    y.replace(["male","female"],[0,1],inplace=True)

    model=LogisticRegression(solver='liblinear').fit(x_data,y)
    return model



def get_noice_details(duration=6):
    # loop through stream and look for dominant peaks while also subtracting noise
    global chunk,samp_rate,stream,noise_fft,noise_amp
    global statusL,recogniseB
    recogniseB.configure(state="disabled")
    noise_fft_vec,noise_amp_vec = [],[]
    #print("RECORDING BACKGROUND\nPlease stay quiet")
    for _ in range(duration*samp_rate//chunk):

        # read stream and convert data from bits to Pascals
        #print(".",end='')
        stream.start_stream()
        data = np.frombuffer(stream.read(chunk,exception_on_overflow=False),dtype=np.int16)
        stream.stop_stream()

        # compute FFT
        fft_data = (np.abs(np.fft.fft(data))[0:chunk//2])/chunk
        fft_data[1:] = 2*fft_data[1:]

        # calculate and subtract average spectral noise
        noise_fft_vec.append(fft_data)
        noise_amp_vec.extend(data)

    noise_fft = np.max(noise_fft_vec,axis=0)
    noise_amp = np.mean(noise_amp_vec)
    noise_status=1
    statusL.configure(text='Click "Recognise me" to recognise your gender')
    recogniseB.configure(state="normal")


def predict_gender_from_voice(model,duration=6,peak_shift=5):
    global samp_rate,chunk,stream,f_vec,low_freq_loc,high_freq_loc,noise_amp,noise_fft
    results=[]
    #print("\nDONE\nPlease speak something continuously")
    for i in range(duration):
        freqs=[]
        amps=[]
        fun_freqs=[]
        dom_freqs=[]
        for _ in range(samp_rate//chunk):

            # read stream and convert data from bits to Pascals
            #print('.',end='')
            stream.start_stream()
            data = np.frombuffer(stream.read(chunk,exception_on_overflow=False),dtype=np.int16)
            data = data-noise_amp
            stream.stop_stream()

            # compute FFT
            fft_data = (np.abs(np.fft.fft(data))[0:chunk//2])/chunk
            fft_data[1:] = 2*fft_data[1:]

            
            fft_data = np.subtract(fft_data,noise_fft) # subtract average spectral noise


            peak_data = 1.0*fft_data
            chunk_peaks=[]
            for jj in range(6):
                max_loc = np.argmax(peak_data[low_freq_loc:high_freq_loc])
                if peak_data[max_loc+low_freq_loc]>10*np.mean(noise_amp):
                    chunk_peaks.append([f_vec[max_loc+low_freq_loc]/1000,fft_data[max_loc+low_freq_loc]])
                    freqs.append(chunk_peaks[-1][0])
                    amps.append(chunk_peaks[-1][1])
                    # zero-out old peaks so we dont find them again
                    peak_data[max_loc+low_freq_loc-peak_shift:max_loc+low_freq_loc+peak_shift] = np.repeat(0,peak_shift*2)
            if(len(chunk_peaks)):
                chunk_peaks=np.array(chunk_peaks)
                fun_freqs.append(chunk_peaks[:,0].min())
                dom_freqs.append(chunk_peaks[np.argmax(chunk_peaks[:,1]),0])

        if(len(freqs)):
            #mode
            mode=pd.DataFrame(freqs)[0].mode()[0]
            #minfun
            minfun=min(fun_freqs)
            #maxdom
            maxdom=max(dom_freqs)
            #Q25
            Energy_sort_indices=np.argsort([freqs[i]*amps[i] for i in range(len(freqs))])
            Q25=(freqs[Energy_sort_indices[len(freqs)//4]]+freqs[Energy_sort_indices[(len(freqs)-2)//4]])/2
            #Q75
            Q75=(freqs[Energy_sort_indices[-1-(len(freqs)//4)]]+freqs[Energy_sort_indices[-1-(len(freqs)-2)//4]])/2
            #IQR
            IQR=Q75-Q25
            #meanfun
            meanfun=sum(fun_freqs)/len(fun_freqs)
            #median
            median=(freqs[Energy_sort_indices[len(freqs)//2]]+freqs[Energy_sort_indices[(len(freqs)-1)//2]])/2
            #skew
            skw=skew(freqs)

            x_test=[mode,minfun,maxdom,Q25,Q75,IQR,meanfun,median,skw]
            x_test=pd.DataFrame(x_test)
            y_prediction_test=model.predict(x_test.T)
            x_test=list(x_test[0])
            if(y_prediction_test[0]==1):
                results.append('female')
                x_test.append('female')
                #print("female")
            else:
                results.append('male')
                x_test.append('male')
                #print("male")
            csv_file=open('myvalues.csv','a')
            csv.writer(csv_file).writerow(x_test)
            csv_file.close()
    if(len(results)):
        try:
            statusL.configure(text=f'\n {mode(results)} ')
        except:
            statusL.configure(text=f"\n{results[len(results)//2]}")
    else:
        statusL.configure(text=f'\nSorry no voice detected\nTry speaking louder')


def config():
    global statusL
    statusL.after(1,lambda :statusL.configure(text='Recording Background sounds \nPlease stay quiet'))
    statusL.after(3000,lambda : get_noice_details())

def recognise(model):
    global statusL
    statusL.after(1,lambda :statusL.configure(text='Recording your sound \nPlease speak continuously'))
    statusL.after(3000,lambda : predict_gender_from_voice(model))

def clear():
    global statusL
    statusL.configure(text='Press "Configure" to record background noice again\n"Recognise me" to recognise your gender')

if __name__ == '__main__':
    model=gen_trained_model('voice.csv')

    form_1 = pyaudio.paInt16 # 16-bit resolution
    chans = 1 # 1 channel
    samp_rate = 44100 # 44.1kHz sampling rate
    chunk = 8192 # samples for buffer (more samples = better freq resolution)

    audio = pyaudio.PyAudio() # create pyaudio instantiation

    f_vec = samp_rate*np.arange(chunk/2)/chunk # frequency vector based on window size and sample rate
    mic_low_freq = 70 # low frequency response of the mic (mine in this case is 100 Hz)
    low_freq_loc = np.argmin(np.abs(f_vec-mic_low_freq))
    mic_high_freq=300
    high_freq_loc=np.argmin(np.abs(f_vec-mic_high_freq))


    # create pyaudio stream
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                        input = True, \
                        frames_per_buffer=chunk)

    noise_status=0
    # some peak-finding and noise preallocations
    peak_shift = 5

    win=tk.Tk()
    win.title('Gender Recognisition From Voice')

    titleL=tk.Label(win,text='Gender Recognisition From voice')
    titleL.grid(row=1,column=1,padx=20,pady=20,columnspan=2)

    configB=tk.Button(win,text='CONFIGURE',command=config)
    configB.grid(row=2,column=1,padx=20,pady=20)

    recogniseB=tk.Button(win,text='RECOGNISE ME',command=lambda :recognise(model),state="disabled")
    recogniseB.grid(row=2,column=2,padx=20,pady=20)

    statusL=tk.Label(win,text='Press "Configure" to record background sounds')
    statusL.grid(row=3,column=1,padx=20,pady=20,columnspan=2)

    clearB=tk.Button(win,text='CLEAR',command=clear)
    clearB.grid(row=4,column=1,padx=20,pady=20,columnspan=2)

    tk.mainloop()
    stream.close()
    audio.terminate()
