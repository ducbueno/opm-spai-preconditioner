#include <memory>
#include <fstream>
#include <iostream>
#include <iterator>
#include "kernel.hpp"
#include "opencl.hpp"

using namespace std;

template<typename T>
void read_vec(string fname, vector<T> &temp){
    T value;
    ifstream input(fname.c_str());

    while(input >> value){
        temp.push_back(value);
    }
    input.close();
}

unsigned int ceilDivision(const unsigned int A, const unsigned int B){
    return A / B + (A % B > 0);
}

int main(){
    vector<int> colIndices;
    vector<int> rowPointers;
    vector<double> nnzValues;
    vector<double> result;

    read_vec<int>("/hdd/mysrc/convert_bsr_coo/data/norne_bsr/colIndices.txt", colIndices);
    read_vec<int>("/hdd/mysrc/convert_bsr_coo/data/norne_bsr/rowPointers.txt", rowPointers);
    read_vec<double>("/hdd/mysrc/convert_bsr_coo/data/norne_bsr/nnzValues.txt", nnzValues);
    result.resize(colIndices.size());

    int platformID = 0;
    int deviceID = 0;

    cl_int err = CL_SUCCESS;
    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    unique_ptr<cl::Context> context;
    unique_ptr<cl::CommandQueue> queue;
    unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, const unsigned int, const unsigned int, cl::LocalSpaceArg> > sat_block_frobenius_k;
    cl::Program::Sources source(1, make_pair(sat_block_frobenius_s, strlen(sat_block_frobenius_s)));
    cl::Program program;
    cl::Buffer d_nnzValues, d_result;

    try{
        cl::Platform::get(&platforms);
        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[platformID])(), 0};
        context.reset(new cl::Context(CL_DEVICE_TYPE_GPU, properties));
        devices = context->getInfo<CL_CONTEXT_DEVICES>();
        program = cl::Program(*context, source);
        program.build(devices);
        queue.reset(new cl::CommandQueue(*context, devices[deviceID], 0, &err));

        //cl_ulong size;
        //cl_device_id tmp_id = devices[deviceID]();
        //clGetDeviceInfo(tmp_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &size, 0);
        //cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE  : " << size << endl;

        d_nnzValues = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * nnzValues.size());
        d_result = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * colIndices.size());
        queue->enqueueWriteBuffer(d_nnzValues, CL_TRUE, 0, sizeof(double) * nnzValues.size(), nnzValues.data());

        sat_block_frobenius_k.reset(new cl::make_kernel<cl::Buffer&, cl::Buffer&, const unsigned int, const unsigned int, cl::LocalSpaceArg>(cl::Kernel(program, "sat_block_frobenius")));

        const unsigned int block_size = 3;
        const unsigned int num_blocks = colIndices.size();
        const unsigned int work_group_size = 256;
        const unsigned int num_work_groups = ceilDivision(num_blocks, work_group_size);
        const unsigned int total_work_items = num_work_groups * work_group_size;
        const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;

        cl::Event event = (*sat_block_frobenius_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
                                                d_nnzValues, d_result, block_size, num_blocks, cl::Local(lmem_per_work_group));

        queue->enqueueReadBuffer(d_result, CL_TRUE, 0, sizeof(double) * result.size(), result.data());
    }

    catch (const cl::Error& error) {
        cout << "OpenCL Error: " << error.what() << "(" << error.err() << ")" << endl;
        cout << getErrorString(error.err()) << endl;
        exit(0);
    }

    ofstream output_file("/hdd/mysrc/spai/data/frobenius.txt");
    ostream_iterator<double> output_iterator(output_file, "\n");
    copy(result.begin(), result.end(), output_iterator);
   
    return 0;
}
