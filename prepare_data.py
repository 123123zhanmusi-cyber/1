import wfdb
import numpy as np
import os

data = []
labels = []

mapping = {
'N':0,
'L':0,
'R':0,
'A':1,
'a':1,
'V':2,
'F':3
}

files = os.listdir("mitdb")

records = list(set([f.split(".")[0] for f in files]))

for record_name in records:

    try:

        record = wfdb.rdrecord("mitdb/"+record_name)
        annotation = wfdb.rdann("mitdb/"+record_name,'atr')

        signal = record.p_signal[:,0]

        for i,peak in enumerate(annotation.sample):

            symbol = annotation.symbol[i]

            if symbol in mapping:

                start = peak-100
                end = peak+200

                if start>0 and end<len(signal):

                    beat = signal[start:end]

                    data.append(beat)
                    labels.append(mapping[symbol])

    except:
        continue

X = np.array(data)
y = np.array(labels)

np.save("X.npy",X)
np.save("y.npy",y)

print("数据处理完成")
print("样本数:",len(X))