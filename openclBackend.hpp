#ifndef __OPENCLBACKEND_H_
#define __OPENCLBACKEND_H_

#include <memory>
#include <vector>
#include "opencl.hpp"

using namespace std;

class openclBackend{
    private:
        int verbosity;
        unsigned int platformID, deviceID;
        unsigned int block_size = 3;

        vector<int> colIndices, rowPointers, csr2csc_mapping;
        vector<double> nnzValues, nnzValuesCSC;
        vector<double> result;

        vector<cl::Device> devices;
        cl::Program program;
        unique_ptr<cl::Context> context;
        unique_ptr<cl::CommandQueue> queue;

        cl::Buffer d_nnzValues, d_satFrobenius, d_colIndices;
        cl::Buffer d_rowPointers, d_mapping, d_maxvals;

        unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, const unsigned int, const unsigned int, cl::LocalSpaceArg> > sat_block_frobenius_k;
        unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer, cl::LocalSpaceArg> > find_max_k;

        unsigned int ceilDivision(const unsigned int A, const unsigned int B);
        void sat_block_frobenius_w(cl::Buffer in, cl::Buffer out, const unsigned int bs, const unsigned int nnzb);
        void find_max_w(cl::Buffer vals, cl::Buffer rind, cl::Buffer cptr, cl::Buffer map, cl::Buffer max, const unsigned int N);
        void initialize();
        void copy_data_to_gpu();
        void read_data_from_gpu(cl::Buffer buf, int len);
        void set_csr2csc_mapping();

    public:
        openclBackend(int verbosity_, unsigned int platformID_, unsigned int deviceID_);
        void run();
};

#endif // __OPENCLBACKEND_H_
