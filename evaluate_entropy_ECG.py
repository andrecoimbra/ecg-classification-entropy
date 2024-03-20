import numpy as np 
import csv
import time

start_time = time.time()


with open ('filename.dat', 'r') as myfile:
    fn = myfile.read().splitlines()

with open ('diagnostics.dat', 'r') as myfile:
    di = myfile.read().splitlines()

with open ('block_list.dat', 'r') as myfile:
    block_list = myfile.read().splitlines()

samples = 10000

def Max_Entropy(x_rand,y_rand,Serie,StatsBlock):
    Threshold=0.
    Frac=10
    Frac2=4
    Increase_Thr=1./Frac
    Max_Threshold=0.0
    S_Max=0
    S=0
    StatsM = np.zeros((pow(2,StatsBlock*StatsBlock)))
    pow_vec=np.zeros((StatsBlock*StatsBlock),np.int64)

    for K in range(0,(StatsBlock*StatsBlock)):
        pow_vec[K] = int(pow(2,K))
    
    for i in range(0,Frac2):
        
        if (i > 0):
            Threshold=Max_Threshold-1.0*Increase_Thr
            Increase_Thr=((2.0*Increase_Thr)/(1.0*Frac))
        
        for j in range(0,Frac):
            
            Stats = np.zeros((pow(2,StatsBlock*StatsBlock)))
            for count in range(len(x_rand)):
                Add=0
                for count_y in range(0,StatsBlock):
                    for count_x in range(0,StatsBlock):
                        a = int(abs(Serie[(x_rand[count]+count_x)]-Serie[(y_rand[count]+count_y)]) <= Threshold)
                        Add += a*pow_vec[count_x+count_y*StatsBlock]
                Stats[Add]+=1
            S=0
            for Hist_S in Stats:
                if (Hist_S > 0): 
                    S-= (float(Hist_S)/(1.0*samples))*(np.log((float(Hist_S)/(1.0*samples))))

            if (S > S_Max):
                S_Max=S; Max_Threshold=Threshold;
                StatsM=Stats
            
            Threshold=Threshold+Increase_Thr
    return Max_Threshold,S_Max,StatsM 

StatsBlock=3
pow_vec=np.zeros((StatsBlock*StatsBlock),np.int64)

diag_list = np.array(['SR','SB','AFIB','ST','SVT','AF','SI','AT','AVNRT','AVRT','SAAWR'])
aaa = np.zeros(len(diag_list))
X = []
Y = []

for i in range(len(fn)):
    if fn[i] not in block_list:
        file = open('ECGDataDenoised/%s.csv' % fn[i])
        csvreader = csv.reader(file)
        j = 0
        for row in csvreader:
            j+=1 
        file = open('ECGDataDenoised/%s.csv' % fn[i])
        csvreader = csv.reader(file) 
        data = []

        print(i,di[i],np.where(diag_list == di[i])[0][0])
        aaa[np.where(diag_list == di[i])[0][0]]+=1
        for row in csvreader:
            data.append(row)
        data = np.array(data)
        data = data.astype(np.float64)
        Aux = []

        for k in range(data.shape[1]):
            Serie = data[:,k]
            Size = len(Serie)
            Serie=(Serie-np.min(Serie))/(np.max(Serie)-np.min(Serie))                            
            Stats=np.zeros((pow(2,StatsBlock*StatsBlock)))
            for K in range(0,(StatsBlock*StatsBlock)):
                pow_vec[K]=int(pow(2,K))

            x_rand = np.random.choice(Size-StatsBlock-1,samples)
            y_rand = np.random.choice(Size-StatsBlock-1,samples)

            Eps,S_max, Stats=Max_Entropy(x_rand,y_rand,Serie,StatsBlock)
            Aux.append(S_max/(StatsBlock*StatsBlock*np.log(2)))
            Aux.append(Eps)

        X.append(Aux)
        Y.append(np.where(diag_list == di[i])[0][0])


        file.close()

X=np.array(X)
Y=np.array(Y)
print(aaa)
np.save('10000_Data_S.npy',X)
np.save('10000_Data_L.npy',Y)
print("--- %s minutes ---" % ((time.time() - start_time)/60))


#for i in range(len(fn)):'''
