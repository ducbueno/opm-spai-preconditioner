#include <fstream>
#include <iostream>

#include "kernel.hpp"
#include "openclBackend.hpp"

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

openclBackend::openclBackend(int verbosity_, unsigned int platformID_, unsigned int deviceID_): verbosity(verbosity_), platformID(platformID_), deviceID(deviceID_){
    cl_int err = CL_SUCCESS;

    try{
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cout << "Found " << platforms.size() << " OpenCL platforms" << endl;

        if (verbosity > 1) {
            std::string platform_info;
            for (unsigned int i = 0; i < platforms.size(); ++i) {
                platforms[i].getInfo(CL_PLATFORM_NAME, &platform_info);
                cout << "Platform name      : " << platform_info << endl;
                platforms[i].getInfo(CL_PLATFORM_VENDOR, &platform_info);
                cout << "Platform vendor    : " << platform_info << endl;
                platforms[i].getInfo(CL_PLATFORM_VERSION, &platform_info);
                cout << "Platform version   : " << platform_info << endl;
                platforms[i].getInfo(CL_PLATFORM_PROFILE, &platform_info);
                cout << "Platform profile   : " << platform_info << endl;
                platforms[i].getInfo(CL_PLATFORM_EXTENSIONS, &platform_info);
                cout << "Platform extensions: " << platform_info << endl << endl;
            }
        }

        std::string platform_info;
        cout << "Chosen:" << endl;
        platforms[platformID].getInfo(CL_PLATFORM_NAME, &platform_info);
        cout << "Platform name      : " << platform_info << endl;
        platforms[platformID].getInfo(CL_PLATFORM_VERSION, &platform_info);
        cout << "Platform version   : " << platform_info << endl << endl;

        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[platformID])(), 0};
        context.reset(new cl::Context(CL_DEVICE_TYPE_GPU, properties));

        devices = context->getInfo<CL_CONTEXT_DEVICES>();
        cout << "Found " << devices.size() << " OpenCL devices" << endl;

        if (verbosity > 1) {
            for (unsigned int i = 0; i < devices.size(); ++i) {
                std::string device_info;
                std::vector<size_t> work_sizes;
                std::vector<cl_device_partition_property> partitions;

                devices[i].getInfo(CL_DEVICE_NAME, &device_info);
                cout << "CL_DEVICE_NAME            : " << device_info << endl;
                devices[i].getInfo(CL_DEVICE_VENDOR, &device_info);
                cout << "CL_DEVICE_VENDOR          : " << device_info << endl;
                devices[i].getInfo(CL_DRIVER_VERSION, &device_info);
                cout << "CL_DRIVER_VERSION         : " << device_info << endl;
                devices[i].getInfo(CL_DEVICE_BUILT_IN_KERNELS, &device_info);
                cout << "CL_DEVICE_BUILT_IN_KERNELS: " << device_info << endl;
                devices[i].getInfo(CL_DEVICE_PROFILE, &device_info);
                cout << "CL_DEVICE_PROFILE         : " << device_info << endl;
                devices[i].getInfo(CL_DEVICE_OPENCL_C_VERSION, &device_info);
                cout << "CL_DEVICE_OPENCL_C_VERSION: " << device_info << endl;
                devices[i].getInfo(CL_DEVICE_EXTENSIONS, &device_info);
                cout << "CL_DEVICE_EXTENSIONS      : " << device_info << endl;

                devices[i].getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &work_sizes);
                for (unsigned int j = 0; j < work_sizes.size(); ++j) {
                    cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES[" << j << "]: " << work_sizes[j] << endl;
                }
                devices[i].getInfo(CL_DEVICE_PARTITION_PROPERTIES, &partitions);
                for (unsigned int j = 0; j < partitions.size(); ++j) {
                    cout << "CL_DEVICE_PARTITION_PROPERTIES[" << j << "]: " << partitions[j] << endl;
                }
                partitions.clear();
                devices[i].getInfo(CL_DEVICE_PARTITION_TYPE, &partitions);
                for (unsigned int j = 0; j < partitions.size(); ++j) {
                    cout << "CL_DEVICE_PARTITION_PROPERTIES[" << j << "]: " << partitions[j] << endl;
                }

                // C-style properties
                cl_device_id tmp_id = devices[i]();
                cl_ulong size;
                clGetDeviceInfo(tmp_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
                cout << "CL_DEVICE_LOCAL_MEM_SIZE       : " << size / 1024 << " KB" << endl;
                clGetDeviceInfo(tmp_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
                cout << "CL_DEVICE_GLOBAL_MEM_SIZE      : " << size / 1024 / 1024 / 1024 << " GB" << endl;
                clGetDeviceInfo(tmp_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_ulong), &size, 0);
                cout << "CL_DEVICE_MAX_COMPUTE_UNITS    : " << size << endl;
                clGetDeviceInfo(tmp_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &size, 0);
                cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE   : " << size / 1024 / 1024 << " MB" << endl;
                clGetDeviceInfo(tmp_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &size, 0);
                cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE  : " << size << endl;
                clGetDeviceInfo(tmp_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
                cout << "CL_DEVICE_GLOBAL_MEM_SIZE      : " << size / 1024 / 1024 / 1024 << " GB" << endl << endl;
            }
        }

        std::string device_info;
        cout << "Chosen:" << endl;
        devices[deviceID].getInfo(CL_DEVICE_NAME, &device_info);
        cout << "CL_DEVICE_NAME            : " << device_info << endl;
        devices[deviceID].getInfo(CL_DEVICE_VERSION, &device_info);
        cout << "CL_DEVICE_VERSION         : " << device_info << endl << endl;

        cl::Event event;
        queue.reset(new cl::CommandQueue(*context, devices[deviceID], 0, &err));
    } catch (const cl::Error& error) {
        cout << "OpenCL Error: " << error.what() << "(" << error.err() << ")" << endl;
    } catch (const std::logic_error& error) {
        throw error;
    }
}

unsigned int openclBackend::ceilDivision(const unsigned int A, const unsigned int B){
    return A / B + (A % B > 0);
}

void openclBackend::sat_block_frobenius_w(cl::Buffer in, cl::Buffer out, const unsigned int bs, const unsigned int nnzb){
    const unsigned int work_group_size = 256;
    const unsigned int num_work_groups = ceilDivision(nnzb, work_group_size);
    const unsigned int total_work_items = 4 * num_work_groups * work_group_size;
    const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;

    cl::Event event = (*sat_block_frobenius_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
                                                in, out, bs, nnzb, cl::Local(lmem_per_work_group));
}

void openclBackend::initialize(){
    read_vec<int>("/hdd/mysrc/convert_bsr_coo/data/norne_bsr/colIndices.txt", colIndices);
    read_vec<int>("/hdd/mysrc/convert_bsr_coo/data/norne_bsr/rowPointers.txt", rowPointers);
    read_vec<double>("/hdd/mysrc/convert_bsr_coo/data/norne_bsr/nnzValues.txt", nnzValues);

    try{
        cl::Program::Sources source(1, make_pair(sat_block_frobenius_s, strlen(sat_block_frobenius_s)));
        program = cl::Program(*context, source);
        program.build(devices);

        d_nnzValues = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * nnzValues.size());
        d_satFrobenius = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * colIndices.size());

        sat_block_frobenius_k.reset(new cl::make_kernel<cl::Buffer&, cl::Buffer&, const unsigned int, const unsigned int, cl::LocalSpaceArg>(cl::Kernel(program, "sat_block_frobenius")));
    } catch (const cl::Error& error) {
        cout << "OpenCL Error: " << error.what() << "(" << error.err() << ")\n";
    } catch (const std::logic_error& error) {
        throw error;
    }
}

void openclBackend::copy_data_to_gpu(){
    try{
        queue->enqueueWriteBuffer(d_nnzValues, CL_TRUE, 0, sizeof(double) * nnzValues.size(), nnzValues.data());
    } catch (const cl::Error& error) {
        cout << "OpenCL Error: " << error.what() << "(" << error.err() << ")\n";
    } catch (const std::logic_error& error) {
        throw error;
    }
}

void openclBackend::run(){
    if(verbosity >= 1){
        cout << "Initializing..." << endl;
    }
    initialize();

    if(verbosity >= 1){
        cout << "Copying data to GPU..." << endl;
    }
    copy_data_to_gpu();

    if(verbosity >= 1){
        cout << "Calculating Frobenius norm of saturation blocks..." << endl;
    }
    sat_block_frobenius_w(d_nnzValues, d_satFrobenius, block_size, colIndices.size());
}
