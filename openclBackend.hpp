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
        unsigned int nnz, nblocks, N, max_nspai;
        double tau;

        vector<int> colIndices, rowPointers, csr2csc_mapping;
        vector<double> nnzValues;

        vector<cl::Device> devices;
        cl::Program program;
        unique_ptr<cl::Context> context;
        unique_ptr<cl::CommandQueue> queue;

        cl::Buffer d_nnzValues, d_satFrobenius, d_colIndices;
        cl::Buffer d_rowPointers, d_mapping, d_maxvals;
        cl::Buffer d_J, d_I, d_row_norm;
        //cl::Buffer d_J, d_I, d_A_hat, d_R;
        //cl::Buffer d_spaiSolutions, d_input, d_output;

        unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, const unsigned int, const unsigned int, cl::LocalSpaceArg> > sat_block_frobenius_k;
        unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg> > find_max_k;
        unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, const unsigned int, const double, cl::Buffer&> > findJ_k;
        unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, const unsigned int> > findI_k;
        unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg, const unsigned int> > get_spai_vals_k;
        //unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, const unsigned int> > construct_A_hat_k;
        //unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg, const unsigned int, const unsigned int> > qr_decomp_iter_k;
        //unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, const unsigned int> > solve_qr_subsystems_k;
        //unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, const unsigned int> > apply_k;

        unsigned int ceilDivision(const unsigned int A, const unsigned int B);
        void sat_block_frobenius_w(cl::Buffer in, cl::Buffer out);
        void find_max_w(cl::Buffer vals, cl::Buffer rind, cl::Buffer cptr, cl::Buffer map, cl::Buffer max);
        void findJ_w(cl::Buffer vals, cl::Buffer cind, cl::Buffer rptr, cl::Buffer map, cl::Buffer max, cl::Buffer j);
        void findI_w(cl::Buffer j, cl::Buffer i, cl::Buffer map, cl::Buffer rptr);
        void get_spai_vals_w(cl::Buffer i, cl::Buffer vals, cl::Buffer cind, cl::Buffer rptr, cl::Buffer norm);
        //void construct_A_hat_w(cl::Buffer j, cl::Buffer i, cl::Buffer vals, cl::Buffer cind, cl::Buffer rptr, cl::Buffer A_hat);
        //void qr_decomp_iter_w(cl::Buffer q, cl::Buffer r);
        //void solve_qr_subsystems_w(cl::Buffer j, cl::Buffer q, cl::Buffer r, cl::Buffer b);
        //void apply_w(cl::Buffer j, cl::Buffer spai, cl::Buffer vals, cl::Buffer rind, cl::Buffer cptr, cl::Buffer in, cl::Buffer out);
        void initialize();
        void copy_data_to_gpu();
        template<typename T> void read_data_from_gpu(cl::Buffer buf, int len, string const& fname);
        void set_csr2csc_mapping();
        void set_sizes();

    public:
        openclBackend(int verbosity_, unsigned int platformID_, unsigned int deviceID_, double tau_);
        void run();
};

#endif // __OPENCLBACKEND_H_
