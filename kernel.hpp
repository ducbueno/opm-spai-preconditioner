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

__kernel void findJ(__global const double *nnzValues,
                    __global const unsigned int *rowIndex,
                    __global const unsigned int *colPtr,
                    __global const unsigned int *mapping,
                    __global const double *maxvals,
                    const unsigned int ncols,
                    const double tau,
                    __global unsigned int *J)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_b = get_group_id(0);
    const unsigned int idx_t = get_local_id(0);
    const unsigned int idx = get_global_id(0);
    const unsigned int num_threads = get_global_size(0);
    const unsigned int num_warps_in_grid = num_threads / warpsize;
    const unsigned int num_vals_per_warp = warpsize / 8;
    const unsigned int lane = idx_t % warpsize;
    unsigned int target_col = idx / warpsize;

    while(target_col < ncols){
        unsigned int first = colPtr[target_col];
        unsigned int last = colPtr[target_col + 1];
        unsigned int current = colPtr[target_col] + lane;

        for(; current < last; current += num_vals_per_warp){
            if(fabs(nnzValues[mapping[current]]) > (1 - tau) * maxvals[target_col]){
                J[current] = rowIndex[current];
            }
        }

        target_col += num_warps_in_grid;
    }
}
)";

const char* findI_s = R"(
__kernel void findI(__global const unsigned int *J,
                    __global unsigned int *I,
                    __global const unsigned int *mapping,
                    __global const unsigned int *colPtr,
                    const unsigned int ncols)
{

    const unsigned int warpsize = 32;
    const unsigned int idx_b = get_group_id(0);
    const unsigned int idx_t = get_local_id(0);
    const unsigned int idx = get_global_id(0);
    const unsigned int num_threads = get_global_size(0);
    const unsigned int num_warps_in_grid = num_threads / warpsize;
    const unsigned int num_vals_per_warp = warpsize / 8;
    const unsigned int lane = idx_t % warpsize;
    unsigned int target_col = idx / warpsize;

    while(target_col < ncols){
        unsigned int first = colPtr[target_col];
        unsigned int last = colPtr[target_col + 1];
        unsigned int current = colPtr[target_col] + lane;

        for(; current < last; current += num_vals_per_warp){
            if(J[current] != -1){
                unsigned int ind = mapping[J[current]];
                I[ind] = J[current];
            }
        }

        target_col += num_warps_in_grid;
    }
}
)";

const char *get_spai_vals_s = R"(
__kernel void get_spai_vals(__global const unsigned int *I,
                            __global const double *nnzValues,
                            __global const unsigned int *colIndex,
                            __global const unsigned int *rowPtr,
                            __global double *norm,
                            __local double *tmp,
                            const unsigned int nrows)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_b = get_group_id(0);
    const unsigned int idx_t = get_local_id(0);
    const unsigned int idx = get_global_id(0);
    const unsigned int num_threads = get_global_size(0);
    const unsigned int num_warps_in_grid = num_threads / warpsize;
    const unsigned int num_vals_per_warp = warpsize / 8;
    const unsigned int lane = idx_t % warpsize;
    unsigned int target_row = idx / warpsize;

    while(target_row < nrows){
        unsigned int first = rowPtr[target_row];
        unsigned int last = rowPtr[target_row + 1];
        unsigned int current = first + lane;
        double local_out = 0.0;

        for(; current < last; current += num_vals_per_warp){
            if(colIndex[current] == I[current]){
                local_out = nnzValues[current] * nnzValues[current];
            }
        }

        tmp[lane] = local_out;
        barrier(CLK_LOCAL_MEM_FENCE);

        for(unsigned int offset = warpsize / 2; offset > 0; offset >>= 1){
            if(lane + offset < warpsize){
                tmp[lane] += tmp[lane + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        norm[target_row] = tmp[0];

        target_row += num_warps_in_grid;
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
    const unsigned int i = wiId % nmax;
    const unsigned int j = wiId / nmax;

    const unsigned int targetCol = get_group_id(0);
    const unsigned int row = J[nmax * targetCol + j];

    if(row == targetCol){
        const unsigned int col = I[nmax * row + i];
        if(col != -1){
            for(unsigned int r = rowPtr[row]; r < rowPtr[row + 1]; r++){
                if(colIndex[r] == col){
                    norm[i] = nnzValues[r] * nnzValues[r];
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

const char *qr_decomp_iter_s = R"(
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

const char *solve_qr_subsystems_s = R"(
__kernel void solve_qr_subsystems(__global const unsigned int *J,
                                  __global const double *Q,
                                  __global double *R,
                                  __global double *b,
                                  const unsigned int nmax)
{
    const unsigned int wiId = get_local_id(0);
    const unsigned int wgId = get_group_id(0);
    const unsigned int row_it = wiId / nmax;
    const unsigned int col_it = wiId % nmax;
    const double tol = 1E-08;
    double dot;

    if(J[nmax * wgId + col_it] == wgId){
        b[wgId * nmax + row_it] = Q[wgId * nmax * nmax + row_it * nmax + col_it];
    }

    if(R[wgId * nmax * nmax + row_it * nmax + row_it] > tol){
        b[wgId * nmax + row_it] /= R[wgId * nmax * nmax + row_it * nmax + row_it];
        R[wgId * nmax * nmax + row_it * nmax + col_it] /= R[wgId * nmax * nmax + row_it * nmax + row_it];
        R[wgId * nmax * nmax + row_it * nmax + row_it] = -R[wgId * nmax * nmax + row_it * nmax + row_it];
    }
    else{
        b[wgId * nmax + row_it] = 0;
    }

    if(wiId == 0){
        for(unsigned int i = nmax - 1; i > 0; i--){
            dot = 0.0;
            for(unsigned int j = nmax; j > i - 1; j--){
                dot += R[wgId * nmax * nmax + (i - 1) * nmax + (j - 1)] * b[wgId * nmax + (j - 1)];
            }
            b[wgId * nmax + (i - 1)] = -dot;
        }
    }
}
)";

const char *apply_s = R"(
__kernel void apply(__global const unsigned int *J,
                    __global const double *spai,
                    __global const double *nnzValues,
                    __global const unsigned int *rowIndex,
                    __global const unsigned int *colPtr,
                    __global const double *input,
                    __global double *output,
                    const unsigned int nmax)
{
    const unsigned int wiId = get_local_id(0);
    const unsigned int wgId = get_group_id(0);
    const unsigned int bs = 3;
    double precond[3];

    if(wiId < nmax){
        const unsigned int j = J[wgId * nmax + wiId];

        if(j != -1){
            if(j == wgId){
                for(unsigned int i = colPtr[wgId]; i < colPtr[wgId + 1]; i++){
                    if(rowIndex[i] == wgId){
                        precond[0] = 1 / nnzValues[i * bs * bs]; // invp
                        break;
                    }
                }
            }
            else{
                precond[0] = 0.0;
            }

            precond[1] = spai[nmax * wgId + wiId];
            precond[2] = spai[nmax * wgId + wiId];

          for(unsigned int i = 0; i < bs; i++){
              output[j * bs + i] += precond[i] * input[j * bs + i];
          }
        }
    }
}
)";

#endif // __KERNEL_H_
