from scipy.sparse import csr_matrix
from scipy import array
import matlab
import matlab.engine
def matlabsolver():
    eng = matlab.engine.start_matlab()
    # get the data, shape and indices
    vsp1 = eng.solver()
    return vsp1