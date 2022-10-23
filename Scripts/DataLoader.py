# Code to read the files
# The files are stores in the eeg-data folder
import numpy as np
import pandas as pd
from scipy import stats
import os
import csv
from scipy import stats
from BCI2kReader import BCI2kReader as b2k 
    
def get_data(src_dir, bad_channels_file, subject_name) :

    #out_dir = 'Desktop/eeg-data/output/'+ subject_name +'_results1.pdf'
    list_dir = os.listdir(src_dir)
    print (list_dir)
    subject_to_signals =  {}  #dictionary; key= subject, value = data

    for subject_dir in os.listdir(src_dir)[0:85]:

        if not subject_dir.startswith("."):
                if (subject_dir.endswith(".dat")):
                    with b2k.BCI2kReader(src_dir+subject_dir, "training1.dat") as file: 
                        subject_to_signals[subject_dir] = file.read()

    f=open(bad_channels_file, 'r')
    csv_reader = csv.DictReader(f)
    Subject_Channels = csv_reader
    in_the_list = 0
    bad_channels = []
    for row in Subject_Channels :
        if (row['Subject']==subject_name.lower()) :
            in_the_list = 1
            if row['c1'] != "" :
                bad_channels.append(int(row['c1']))
            if row['c2'] != "" :
                bad_channels.append(int(row['c2']))
            if row['c3'] != "" :
                bad_channels.append(int(row['c3']))
            if row['c4'] != "" :
                bad_channels.append(int(row['c4']))
            if row['c5'] != "" :
                bad_channels.append(int(row['c5']))
            if row['c6'] != "" :
                bad_channels.append(int(row['c6']))
    print(len(bad_channels))    

                        
# Prepare the data for training
# Train data is for EEGNet trainng & testing as well as RG+xDawn evaluation
#192 temporal data points are used for EEGNet 
#Iterative work pointed that 128 temporal points for ech event produced gives best LDA performance
    window = 192   #Signal time series window
    sht_window = 128 # shoretened window for LDA analysis
    subject_name = []
    z_score = []
    zerozscore = []
    #print (subject_to_signals)
    num_channels = 32 - len(bad_channels)
    array_elements = sht_window * num_channels

    Train_data = np.array(np.empty((0,num_channels,window), float))  # Training data array initialization
    Train_data_inception = np.array(np.empty((0,window,num_channels), float))  # Training data array initialization

    Train_labels = []   # Training label data initialization
    Train_data_flattened = np.array(np.empty((0,array_elements), float))  # Training data array initialization



    for subject, my_signals in subject_to_signals.items():
       # print (my_signals)
        Train_data_temp = np.array(np.empty((0,num_channels,window), float))  # Training data array initialization
        Train_data_temp_inception = np.array(np.empty((0,window,num_channels), float))  # Training data array initialization

        Train_labels_temp = []   # Training label data initialization
        Train_data_flattened_temp = np.array(np.empty((0,array_elements), float))  # Training data array initialization

        stimuluscode = (my_signals[1]['StimulusCode']) #ndarrays ; target 
        stimulustype = (my_signals[1]['StimulusType']) #p3label
        data = stimuluscode[0] #target numbers
        data_label = stimulustype[0] #p3label numbers
        rownum = len(data)
        labelrow = len(data_label)
        zerozscore = (-1*np.mean(my_signals[0])/(my_signals[0]).std())

        change = [] #indexes of 0 where next index is nonzero for target
        for x in range(rownum-1):    
            if bool(data.item(x) == 0) and bool(data.item(x + 1) != 0): 
                    change.append(x)



        attended = [] #indexes where p3label value is 1
        nonattended = [] #indexes where p3label value is 0


        nonattended_count =1

        for x in range(len(change)): 

            attended_values = np.empty((0,window), float) #2D array with train data each row in same a time frame
            nonattended_values = np.empty((0,window), float)
            attended_values_sht = np.empty((0,sht_window), float) #2D array with train data each row in same a time frame
            nonattended_values_sht = np.empty((0,sht_window), float)

            if bool(data_label.item(change[x] + 1) == 0): 
                nonattended_count =nonattended_count +1    
                nonattended.append(change[x])
                Train_labels_temp.append(int(0))
                for num in range(len(my_signals[0])):
                    if num not in bad_channels :  # exclude bad channels data
                        electrode = my_signals[0][num]
                        zelectrode = stats.zscore(my_signals[0][num])
                        nonattendedtemp = []
                        nonattendedtemp_sht =[]
                        if (change[x] + window) > len(zelectrode):
                            addtime = len(zelectrode) - (change[x])            
                            for y in range(addtime):
                                nonattendedtemp.append(zelectrode[change[x] + y])
                            for y in range(sht_window):
                                nonattendedtemp_sht.append(zelectrode[change[x] + y])
                            nonattended_values_sht = np.append(nonattended_values_sht, np.array([nonattendedtemp_sht]), axis = 0)               
                            nonattended_values = np.append(nonattended_values, np.array([nonattendedtemp]), axis = 0)
                        else:
                            for y in range(window):
                                nonattendedtemp.append(zelectrode[change[x] + y])
                            for y in range(sht_window):
                                nonattendedtemp_sht.append(zelectrode[change[x] + y])
                        nonattended_values = np.append(nonattended_values, np.array([nonattendedtemp]), axis = 0)
                        nonattended_values_sht = np.append(nonattended_values_sht, np.array([nonattendedtemp_sht]), axis = 0)               
                Train_data_flattened_temp = np.append(Train_data_flattened_temp, np.array([nonattended_values_sht.flatten()]), axis=0)
                Train_data_temp_inception = np.append(Train_data_temp_inception,np.array([nonattended_values.T]), axis=0)
                Train_data_temp = np.append(Train_data_temp,np.array([nonattended_values]), axis=0)

            else: 
                attended.append(change[x] + 1) 
                Train_labels_temp.append(int(1))

                for num in range(len(my_signals[0])):
                    if num not in bad_channels :  # exclude bad channels data
                    
                        electrode = my_signals[0][num]
                        zelectrode = stats.zscore(my_signals[0][num])
                        attendedtemp = []
                        attendedtemp_sht = []

                        for y in range(window): 
                            attendedtemp.append(zelectrode[change[x] + y])
                        for y in range(sht_window): 
                            attendedtemp_sht.append(zelectrode[change[x] + y])

                        attended_values = np.append(attended_values, np.array([attendedtemp]), axis = 0)
                        attended_values_sht = np.append(attended_values_sht, np.array([attendedtemp_sht]), axis = 0)

                Train_data_flattened_temp = np.append( Train_data_flattened_temp, np.array([attended_values_sht.flatten()]), axis=0)
                Train_data_temp_inception = np.append( Train_data_temp_inception,np.array([attended_values.T]), axis=0)        
                Train_data_temp = np.append( Train_data_temp,np.array([attended_values]), axis=0)        

        Train_data_flattened = np.concatenate((Train_data_flattened, Train_data_flattened_temp),axis=0) # format is in (trials, channels, samples)
        Train_data = np.concatenate((Train_data, Train_data_temp),axis=0) # format is in (trials, channels, samples)
        Train_data_inception = np.concatenate((Train_data_inception, Train_data_temp_inception),axis=0) # format is in (trials, channels, samples)
        Train_labels = np.concatenate((Train_labels, Train_labels_temp),axis=0) 

        print("no of labled P300 events:"+str(len(attended)))
        print("no of non_attended P300 events:"+str(len(nonattended)))

    print(np.shape(Train_data))
    print(np.shape(Train_labels))
    print(np.shape(Train_data_flattened)) 
    return Train_data, Train_data_inception, Train_data_flattened, Train_labels, num_channels
