X_text['1'].apply(lambda x: x.lower())


text = {}
for i in range(1,len(X_text.columns)+1):
    count_vectorize.fit(X_text[str(i)].values.astype('U'))
    text['x{0}'.format(i)] = count_vectorize.transform(X_text[str(i)].values.astype('U'))  



a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}

[df for df in a_dict.items()]

import numpy as np
import pandas as pd

a = np.array([0,1,2])
np.tile(a, (1,3))

x = ['1','2','3','4']
y = ['joe','joe','joe','joe']

data = pd.DataFrame([x,y])

data.iloc[::-1]

data = data.transpose()

data[2] = data[[0,1]].apply(lambda x: ' '.join(x), axis=1)

data.iloc[:,0:].apply(lambda x: ' '.join(x), axis=1)
