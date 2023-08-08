import numpy as np
cimport numpy as np
cimport cython
import pyximport
pyximport.install()

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

def dim(np.ndarray[DTYPE_t, ndim=6] cols,
                            np.ndarray[DTYPE_t, ndim=4] x_padded,
                            int N, int C, int H, int W, int HH, int WW,
                            int stride):
    cdef int out_h=(H-HH)//stride+1
    cdef int out_w=(W-WW)//stride+1
    cdef int c, hh, ww, n, h, w
    for n in range(N):
        for c in range(C):
            for hh in range(HH):
                for ww in range(WW):
                    for h in range(out_h):
                        for w in range(out_w):
                            x_padded[n, c, stride * h + hh, stride * w + ww] += cols[c, hh, ww, n, h, w]

    return x_padded