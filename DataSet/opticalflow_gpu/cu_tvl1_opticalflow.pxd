cimport numpy as np
cimport cython
from libcpp cimport bool
from cpython.ref cimport PyObject

# References PyObject to OpenCV object conversion code borrowed from OpenCV's own conversion file, cv2.cpp
cdef extern from 'pyopencv_converter.cpp':
    void import_array()
    cdef PyObject* pyopencv_from(const Mat& m)
    cdef bool pyopencv_to(PyObject* o, Mat& m)

cdef extern from 'opencv2/core.hpp' namespace 'cv':
    cdef cppclass Mat:
        Mat() except +
        void create(int, int, int) except +
        void* data
        int rows
        int cols

cdef extern from 'opencv2/core/cvstd.hpp': 
    cdef cppclass Ptr[T]:
        Ptr() except + 
        Ptr(Ptr*) except +
        T& operator* () # probably no exceptions

cdef extern from 'opencv2/core/cuda.hpp' namespace 'cv::cuda':
    cdef cppclass GpuMat:
        GpuMat() except +
        void upload(Mat arr) except +
        void download(Mat dst) const
    cdef cppclass Stream:
        Stream() except +

cdef extern from 'opencv2/cudaoptflow.hpp' namespace 'cv::cuda':
    cdef cppclass OpticalFlowDual_TVL1:
        OpticalFlowDual_TVL1() except +
        # Function using defualt values
        void calc(GpuMat I0, GpuMat I1, GpuMat flow)
        @staticmethod
        Ptr[OpticalFlowDual_TVL1] create()
