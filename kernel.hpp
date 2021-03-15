#ifndef __KERNEL_H_
#define __KERNEL_H_

const char* find_max_s = R"(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void find_max(__global const double *vals,
                       __global const unsigned int *rowIndex,
                       __global const unsigned int *colPtr,
                       __global const unsigned int *mapping,
                       __global double *maxvals,
                       __local double *tmp)
{
    const unsigned int wiId = get_local_id(0);
    const unsigned int wgId = get_group_id(0);
    const unsigned int wgSz = get_local_size(0);

    unsigned int first_row = colPtr[wgId];
    unsigned int last_row = colPtr[wgId + 1];
    unsigned int row = first_row + wiId;
  
    double local_max = -MAXFLOAT;
    for(; row < last_row; row += wgSz){
        if(fabs(vals[mapping[row]]) > local_max){
            local_max = fabs(vals[mapping[row]]);
        }
    }

    tmp[wiId] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int offset = wgSz/2; offset > 0; offset >>= 1){
        if(tmp[wiId + offset] > tmp[wiId]){
            tmp[wiId] = tmp[wiId + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(wiId == 0) maxvals[wgId] = tmp[0];
}
)";

const char* findJ_s = R"(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void findJ(__global const double *vals,
                    __global const unsigned int *rowIndex,
                    __global const unsigned int *colPtr,
                    __global const unsigned int *mapping,
                    __global const double *maxvals,
                    const unsigned int n2max,
                    const double tau,
                    __global unsigned int *Jind)
{
    const unsigned int wiId = get_local_id(0);
    const unsigned int wgId = get_group_id(0);
    const unsigned int wgSz = get_local_size(0);

    __local unsigned int offDiagCount;
    offDiagCount = 0;

    unsigned int first_row = colPtr[wgId];
    unsigned int last_row = colPtr[wgId + 1];
    unsigned int row = first_row + wiId;

    if(wiId < n2max){
        for(; row < last_row; row += wgSz){
            if(rowIndex[row] == wgId){
                Jind[n2max * wgId + wiId] = wgId;
            }

            else if(fabs(vals[mapping[row]]) > (1 - tau)*maxvals[row] && offDiagCount < n2max - 1){
                Jind[n2max * wgId + wiId] = rowIndex[row];
                atomic_inc((__local int *)&offDiagCount);
            }
        }
    }
}
)";

const char* findI_s = R"(
__kernel void findI(__global const unsigned int *J,
                    __global unsigned int *I,
                    const unsigned int nmax)
{
    const unsigned int wiId = get_local_id(0);
    const unsigned int wgId = get_group_id(0);

    if(wiId < nmax){
        unsigned int col = wgId;
        unsigned int row = J[nmax * col + wiId];

        if(row != -1){
            I[nmax * row + wiId] = col;
        }
    }
}
)";

const char* construct_A_hat_s = R"(
__kernel void construct_A_hat(__global const unsigned int *J,
                              __global const unsigned int *I,
                              __global const double *nnzValues,
                              __global const unsigned int *colIndex,
                              __global const unsigned int *rowPtr,
                              __global double *A_hat,
                              const unsigned int nmax)
{
    const unsigned int wiId = get_local_id(0);
    const unsigned int wgId = get_group_id(0);
    const unsigned int i_it = wiId / nmax; // work_group_size must be nmax*nmax (=64)
    const unsigned int j_it = wiId % nmax;
    const unsigned int i = I[nmax * wgId + i_it];
    const unsigned int j = J[nmax * wgId + j_it];

    if(i != -1){
        if(j != -1){
            for(unsigned int k = rowPtr[j]; k < rowPtr[j + 1]; k++){
                if(colIndex[k] == i){
                    A_hat[wgId * nmax * nmax + i_it * nmax + j_it] = nnzValues[k];
                    break;
                }
            }
        }
    }
}
)";

const char* sat_block_frobenius_s = R"(
__kernel void sat_block_frobenius(__global const double *vals,
                                  __global double *result,
                                  const unsigned int block_size,
                                  const unsigned int num_blocks,
                                  __local double *tmp)
{
    const unsigned int idx_t = get_local_id(0);
    const unsigned int bs = block_size;
    const unsigned int sat_bs = 2;
    const unsigned int c = (idx_t / sat_bs) % sat_bs + 1;
    const unsigned int r = idx_t % sat_bs + 1;
    unsigned int block = get_global_id(0) / (sat_bs*sat_bs);

    if(block < num_blocks){
        double A_elem = vals[block*bs*bs + c + r*bs];
        tmp[idx_t] = A_elem * A_elem;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int offset = 3; offset > 0; offset--){
        if (idx_t % 4 == 0){
            tmp[idx_t] += tmp[idx_t + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(block < num_blocks){
        if(idx_t % 4 == 0){
            result[block] = tmp[idx_t];
        }
    }
}
)";

const char* qr_decomp_iter_s = R"(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void qr_decomp_iter(__global double *Q,
                             __global double *R,
                             __local double *tmp,
                             const unsigned int nmax,
                             const unsigned int qcol)
{
    const unsigned int wiId = get_local_id(0);
    const unsigned int wgId = get_group_id(0);
    const unsigned int row_it = wiId % nmax;
    const unsigned int col_it = wiId / nmax;
    const unsigned int target_col = (qcol + col_it) % nmax;
    const double val = Q[wgId * nmax * nmax + row_it * nmax + qcol];
    const double valn = Q[wgId * nmax * nmax + row_it * nmax + target_col];
    double colnorm, dot;

    tmp[wiId] = val * valn;

    for(unsigned int offset = 1; offset < nmax; offset <<= 1){
        if(wiId % (2 * offset) == 0){
            tmp[wiId] += tmp[wiId + offset];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    colnorm = sqrt(tmp[0]);
    R[wgId * nmax * nmax + qcol * nmax + qcol] = colnorm;

    if(col_it == qcol){
        if(colnorm != 0){
            Q[wgId * nmax * nmax + row_it * nmax + col_it] /= colnorm;
        }
    }

    if(col_it > qcol){
        if(colnorm != 0){
            dot = tmp[(col_it - qcol) * nmax] / colnorm;
        }
        else{
            dot = 0.0;
        }

        R[wgId * nmax * nmax + qcol * nmax + col_it] = dot;
        Q[wgId * nmax * nmax + row_it * nmax + col_it] -= Q[wgId * nmax * nmax + row_it * nmax + qcol] * dot;
    }
}
)";

#endif // __KERNEL_H_
