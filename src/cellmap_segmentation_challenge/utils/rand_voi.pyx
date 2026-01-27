# Code originally from funlib.evaluate@git+//github.com/funkelab/funlib.evaluate
from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.map cimport map as cpp_map
import numpy as np
cimport numpy as np

def rand_voi(truth, test, return_cluster_scores=False):

    for d in range(truth.ndim):
        assert truth.shape[d] == test.shape[d], (
                "shapes between truth and test don't match")

    return rand_voi_wrapper(
        np.ravel(truth, order='A'),
        np.ravel(test, order='A'),
        return_cluster_scores)

def rand_voi_wrapper(
        np.ndarray[uint64_t] truth,
        np.ndarray[uint64_t] test,
        return_cluster_scores):

    # the C++ part assumes contiguous memory, make sure we have it (and do
    # nothing, if we do)
    if truth is not None and not truth.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous ground-truth arrray (avoid this by passing C_CONTIGUOUS arrays)")
        truth = np.ascontiguousarray(truth)
    if not test.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous test arrray (avoid this by passing C_CONTIGUOUS arrays)")
        test = np.ascontiguousarray(test)

    cdef uint64_t* test_data
    cdef uint64_t* truth_data

    test_data = <uint64_t*>test.data
    truth_data = <uint64_t*>truth.data

    return rand_voi_arrays(
        test.size,
        truth_data,
        test_data,
        return_cluster_scores)

cdef extern from "impl/rand_voi.hpp":

    struct Metrics:
        double rand_split
        double rand_merge
        double voi_split
        double voi_merge
        double nvi_split
        double nvi_merge
        double nid
        cpp_map[uint64_t, double] voi_split_i
        cpp_map[uint64_t, double] voi_merge_j

    Metrics rand_voi_arrays(
            size_t          size,
            const uint64_t* truth_data,
            const uint64_t* test_data,
            bool            return_cluster_scores);