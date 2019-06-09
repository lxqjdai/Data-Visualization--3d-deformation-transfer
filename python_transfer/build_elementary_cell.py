import numpy as np
import numpy.linalg
def build_elementary_cell(ts,l):
    E=[]
    import numpy as np
    for i in range(l):
        temp2=numpy.linalg.inv(ts[i])
        temp1=-np.sum(temp2,0)
        E.append(np.transpose(np.vstack((temp1,temp2))))
    return E