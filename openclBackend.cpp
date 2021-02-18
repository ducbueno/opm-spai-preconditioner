#include <fstream>
#include <iostream>
#include <chrono>
#include <iterator>

#include "kernel.hpp"
#include "openclBackend.hpp"

using namespace std;
using namespace std::chrono;

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
        std::vector<cl::Platform> platforus;
        cl::Platform::get(&platforus);
        cout << "Found " << platforus.size() << " OpenCL platforus" << endl;

        if (verbosity > 1) {
            std::string platform_info;
            for (unsigned int i = 0; i < platforus.size(); ++i) {
                platforus[i].getInfo(CL_PLATFORM_NAME, &platform_info);
                cout << "Platform name      : " << platform_info << endl;
                platforus[i].getInfo(CL_PLATFORM_VENDOR, &platform_info);
                cout << "Platform vendor    : " << platform_info << endl;
                platforus[i].getInfo(CL_PLATFORM_VERSION, &platform_info);
                cout << "Platform version   : " << platform_info << endl;
                platforus[i].getInfo(CL_PLATFORM_PROFILE, &platform_info);
                cout << "Platform profile   : " << platform_info << endl;
                platforus[i].getInfo(CL_PLATFORM_EXTENSIONS, &platform_info);
                cout << "Platform extensions: " << platform_info << endl << endl;
            }
        }

        std::string platform_info;
        cout << "Chosen:" << endl;
        platforus[platformID].getInfo(CL_PLATFORM_NAME, &platform_info);
        cout << "Platform name      : " << platform_info << endl;
        platforus[platformID].getInfo(CL_PLATFORM_VERSION, &platform_info);
        cout << "Platform version   : " << platform_info << endl << endl;

        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforus[platformID])(), 0};
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
        exit(0);
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
    const unsigned int total_work_iteus = 4 * num_work_groups * work_group_size;
    const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;

    cl::Event event = (*sat_block_frobenius_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_iteus), cl::NDRange(work_group_size)),
                                               in, out, bs, nnzb, cl::Local(lmem_per_work_group));
}

void openclBackend::find_max_w(cl::Buffer vals, cl::Buffer cind, cl::Buffer rptr, cl::Buffer map, cl::Buffer max, const unsigned int N){
    const unsigned int work_group_size = 32;
    const unsigned int num_work_groups = ceilDivision(N, work_group_size);
    const unsigned int total_work_iteus = num_work_groups * work_group_size;
    const unsigned int row_offset = 4;
    const unsigned int lmem_per_work_group = sizeof(double) * row_offset;

    cl::Event event = (*find_max_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_iteus), cl::NDRange(work_group_size)),
                                    vals, cind, rptr, map, max, cl::Local(lmem_per_work_group), row_offset, N);
    event.wait();
}

void openclBackend::initialize(){
    try{
        cout << "    - Building kernels..." << endl;
        cl::Program::Sources source(1, make_pair(sat_block_frobenius_s, strlen(sat_block_frobenius_s)));
        source.emplace_back(make_pair(find_max_s, strlen(find_max_s)));
        program = cl::Program(*context, source);
        program.build(devices);

        cout << "    - Setting buffers..." << endl;
        d_nnzValues = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * nnzValues.size());
        d_colIndices = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * colIndices.size());
        d_rowPointers = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * rowPointers.size());
        d_mapping = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * csr2csc_mapping.size());
        d_satFrobenius = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * colIndices.size());
        d_maxvals = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * (rowPointers.size() - 1));

        cout << "    - Setting kernels..." << endl << endl;
        sat_block_frobenius_k.reset(new cl::make_kernel<cl::Buffer&, cl::Buffer&, const unsigned int, const unsigned int, cl::LocalSpaceArg>(cl::Kernel(program, "sat_block_frobenius")));
        find_max_k.reset(new cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer,
                                             cl::LocalSpaceArg, const unsigned int, const unsigned int>(cl::Kernel(program, "find_max")));
    } catch (const cl::Error& error) {
        cout << "OpenCL Error: " << error.what() << "(" << error.err() << ")\n";

        if(error.err() == CL_BUILD_PROGRAM_FAILURE){
            for(cl::Device dev: devices){
                cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                if(status != CL_BUILD_ERROR) continue;

                string name = dev.getInfo<CL_DEVICE_NAME>();
                string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                cerr << "Build log for " << name << ":" << endl << buildlog << endl;
            }
        }
       
        exit(0);
    } catch (const std::logic_error& error) {
        throw error;
    }
}

void openclBackend::copy_data_to_gpu(){
    try{
        queue->enqueueWriteBuffer(d_nnzValues, CL_TRUE, 0, sizeof(double) * nnzValues.size(), nnzValues.data());
        queue->enqueueWriteBuffer(d_colIndices, CL_TRUE, 0, sizeof(int) * colIndices.size(), colIndices.data());
        queue->enqueueWriteBuffer(d_rowPointers, CL_TRUE, 0, sizeof(int) * rowPointers.size(), rowPointers.data());
        queue->enqueueWriteBuffer(d_mapping, CL_TRUE, 0, sizeof(int) * csr2csc_mapping.size(), csr2csc_mapping.data());
    } catch (const cl::Error& error) {
        cout << "OpenCL Error: " << error.what() << "(" << error.err() << ")\n";
        exit(0);
    } catch (const std::logic_error& error) {
        throw error;
    }
}

void openclBackend::read_data_from_gpu(cl::Buffer buf, int len){
    result.resize(len);

    try{
        queue->enqueueReadBuffer(buf, CL_TRUE, 0, sizeof(double) * result.size(), result.data());
    } catch (const cl::Error& error) {
        cout << "OpenCL Error: " << error.what() << "(" << error.err() << ")\n";
        exit(0);
    } catch (const std::logic_error& error) {
        throw error;
    }
}

void openclBackend::set_csr2csc_mapping(){
    vector<int> aux(rowPointers);
    csr2csc_mapping.resize(colIndices.size());

    for(int row = 0; row < int(rowPointers.size()) - 1; row++){
        for(int jj = rowPointers[row]; jj < rowPointers[row+1]; jj++){
            int col = colIndices[jj];
            int dest = aux[col];

            csr2csc_mapping[dest] = jj;

            aux[col]++;
        }
    }
}

void openclBackend::run(){
    read_vec<int>("/hdd/mysrc/convert_bsr_coo/data/norne_bsr/colIndices.txt", colIndices);
    read_vec<int>("/hdd/mysrc/convert_bsr_coo/data/norne_bsr/rowPointers.txt", rowPointers);
    read_vec<double>("/hdd/mysrc/convert_bsr_coo/data/norne_bsr/nnzValues.txt", nnzValues);

    cout << "Setting CSR to CSC conversion mapping..." << endl;
    auto start = high_resolution_clock::now();
    set_csr2csc_mapping();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "    -> took " << duration.count() << " us" << endl << endl;

    cout << "Initializing OpenCL..." << endl;
    initialize();

    cout << "Copying data to GPU..." << endl;
    copy_data_to_gpu();

    cout << "Calculating Frobenius norm of saturation blocks..." << endl;
    start = high_resolution_clock::now();
    sat_block_frobenius_w(d_nnzValues, d_satFrobenius, block_size, colIndices.size());
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "    -> took " << duration.count() << " us" << endl << endl;

    cout << "Calculating maximum values per column..." << endl;
    start = high_resolution_clock::now();
    find_max_w(d_satFrobenius, d_colIndices, d_rowPointers, d_mapping, d_maxvals, rowPointers.size() - 1);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "    -> took " << duration.count() << " us" << endl << endl;

    cout << "Reading result from GPU..." << endl;
    read_data_from_gpu(d_maxvals, rowPointers.size() - 1);

    cout << "Writing result to file..." << endl;
    ofstream output_file("/hdd/mysrc/spai/data/maxvals.txt");
    ostream_iterator<double> output_iterator(output_file, "\n");
    copy(result.begin(), result.end(), output_iterator);
}
