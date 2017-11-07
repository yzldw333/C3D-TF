import numpy as np  # Import Python functions, attributes, submodules of numpy
cimport numpy as np  # Import numpy C/C++ API
from cython.operator cimport dereference
def cudaTVL1OpticalFlowWrapper(np.ndarray[np.uint8_t, ndim=2] prsv,
                               np.ndarray[np.uint8_t, ndim=2] next):
    np.import_array()
    cdef Mat prevMat

    pyopencv_to(<PyObject*> prsv, prevMat)
    cdef GpuMat prevMat_GPU
    prevMat_GPU.upload(prevMat)
    cdef Mat nextMat

    pyopencv_to(<PyObject*> next, nextMat)
    cdef GpuMat nextMat_GPU
    nextMat_GPU.upload(nextMat)
    cdef Ptr[OpticalFlowDual_TVL1] pFlow= OpticalFlowDual_TVL1.create()


    cdef GpuMat flowMat_GPU
    cdef Mat flowMat
    #pyopencv_to(<PyObject*> flow, flowMat)
    dereference(pFlow).calc(prevMat_GPU,nextMat_GPU,flowMat_GPU)

    flowMat_GPU.download(flowMat)
    flow = <np.ndarray>pyopencv_from(flowMat)
    return flow
