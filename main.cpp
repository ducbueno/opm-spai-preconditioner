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
    int platformID = 0;
    int deviceID = 0;

    cl_int err = CL_SUCCESS;
    unique_ptr<cl::Context> context;
    unique_ptr<cl::CommandQueue> queue;
    unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, const unsigned int, const unsigned int, cl::LocalSpaceArg> > sat_block_frobenius_k;

    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[platformID])(), 0};
    context.reset(new cl::Context(CL_DEVICE_TYPE_GPU, properties));

    vector<cl::Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();

    cout << "Platform, context and device set succesfully!" << endl;

    cl::Program::Sources source(1, make_pair(sat_block_frobenius_s, strlen(sat_block_frobenius_s)));
    cl::Program program = cl::Program(*context, source);
    program.build(devices);

    cout << "Program built succesfully!" << endl;

    queue.reset(new cl::CommandQueue(*context, devices[deviceID], 0, &err));

    cout << "Queue set succesfully!" << endl;

    vector<int> colIndices;
    vector<int> rowPointers;
    vector<double> nnzValues;
    vector<double> result;

    read_vec<int>("/hdd/mysrc/convert_bsr_coo/data/norne_bsr/colIndices.txt", colIndices);
    read_vec<int>("/hdd/mysrc/convert_bsr_coo/data/norne_bsr/rowPointers.txt", rowPointers);
    read_vec<double>("/hdd/mysrc/convert_bsr_coo/data/norne_bsr/nnzValues.txt", nnzValues);
    result.resize(colIndices.size());

    cl::Buffer d_nnzValues = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * nnzValues.size());
    cl::Buffer d_result = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * colIndices.size());
    queue->enqueueWriteBuffer(d_nnzValues, CL_TRUE, 0, sizeof(double) * nnzValues.size(), nnzValues.data());

    cout << "Data succesfully written to buffers!" << endl;

    sat_block_frobenius_k.reset(new cl::make_kernel<cl::Buffer&, cl::Buffer&, const unsigned int, const unsigned int, cl::LocalSpaceArg>(cl::Kernel(program, "sat_block_frobenius")));

    cout << "Kernel set succesfully!" << endl;

    const unsigned int block_size = 3;
    const unsigned int num_blocks = colIndices.size();
    const unsigned int work_group_size = 32;
    const unsigned int num_work_groups = ceilDivision(num_blocks, work_group_size);
    const unsigned int total_work_items = num_work_groups * work_group_size;
    const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;

    cl::Event event = (*sat_block_frobenius_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(num_work_groups)),
                                               d_nnzValues, d_result, block_size, num_blocks, cl::Local(lmem_per_work_group));

    cout << "Kernel executed succesfully!!!" << endl;

    queue->enqueueReadBuffer(d_result, CL_TRUE, 0, sizeof(double) * result.size(), result.data());

    cout << "Data succesfully read from device!" << endl;
   
    for(double r: result) cout << r << endl;

    /*
    string v_opath = fpath + "v_-opencl.txt";
    string t_opath = fpath + "t_-opencl.txt";
    ofstream v_output_file(v_opath.c_str());
    ofstream t_output_file(t_opath.c_str());
    ostream_iterator<double> v_output_iterator(v_output_file, "\n");
    ostream_iterator<double> t_output_iterator(t_output_file, "\n");
    copy(h_v.begin(), h_v.end(), v_output_iterator);
    copy(h_t.begin(), h_t.end(), t_output_iterator);
    */

    return 0;
}
