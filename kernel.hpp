#ifndef __KERNEL_H_
#define __KERNEL_H_

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
            result[block] = sqrt(tmp[idx_t]);
        }
    }
}
)";

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

const char* assemble_I_J_s = R"(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void assemble_I_J(__global const double *nnzValues,
                           __global const unsigned int *colIndex,
                           __global const unsigned int *rowPtr,
                           __global const unsigned int *mapping,
                           __global const double *maxvals,
                           const unsigned int nrows,
                           const double tau,
                           __global int *I,
                           __global int *J)
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
        unsigned int current = rowPtr[target_row] + lane;

        for(; current < last; current += num_vals_per_warp){
            if(fabs(nnzValues[current]) > (1 - tau) * maxvals[colIndex[current]]){
                I[current] = colIndex[current];
            }

            if(fabs(nnzValues[mapping[current]]) > (1 - tau) * maxvals[target_row]){
                J[current] = colIndex[current];
            }
        }

        target_row += num_warps_in_grid;
    }
}
)";

const char *get_spai_s = R"(
__kernel void get_spai(__global const unsigned int *I,
                       __global const unsigned int *J,
                       __global const double *nnzValues,
                       __global const unsigned int *rowIndex,
                       __global const unsigned int *colPtr,
                       __global const unsigned int *mapping,
                       __global double *x)
{
    const unsigned int idx_b = get_group_id(0);
    const unsigned int idx_t = get_local_id(0);
    const unsigned int first = colPtr[idx_b];
    const unsigned int last = colPtr[idx_b + 1];

    unsigned int valid_rows_idx[8];
    unsigned int one_row_idx;
    unsigned int n_valid_rows = 0;
    unsigned int row_idx, col_idx;
    unsigned int first_col_in_row, last_col_in_row, current_col_in_row;

    double normalize_residual;
    __local double submat[8 * 8];
    __local double residual[8];
    __local double conjugate[8];

    double alpha = 0.0;
    double normalize_x = 0.0;
    __local double x0[8];

    for(unsigned int current = first; current < last; current++){
        if(rowIndex[current] == J[current]){
            valid_rows_idx[n_valid_rows] = J[current];

            if(J[current] == idx_b){
                one_row_idx = n_valid_rows;
            }

            n_valid_rows++;
        }
    }

    row_idx = idx_t % n_valid_rows;
    col_idx = idx_t / n_valid_rows;
    first_col_in_row = colPtr[valid_rows_idx[row_idx]];
    last_col_in_row = colPtr[valid_rows_idx[row_idx] + 1];
    current_col_in_row = first_col_in_row + col_idx;

    if(current_col_in_row < last_col_in_row){
        for(unsigned int i = 0; i < n_valid_rows; i++){
            submat[n_valid_rows * row_idx + i] = 0.0;
            if(I[current_col_in_row] == valid_rows_idx[i]){
                submat[n_valid_rows * row_idx + i] = nnzValues[current_col_in_row];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(idx_t < n_valid_rows){
        residual[idx_t] = 0.0;
        conjugate[idx_t] = 0.0;

        for(unsigned int i = 0; i < n_valid_rows; i++){
            residual[idx_t] += submat[n_valid_rows * idx_t + i] * submat[n_valid_rows * idx_t + i];
        }

        normalize_residual = (residual[one_row_idx] != 0) ? residual[one_row_idx] : 1.0;
        residual[one_row_idx] = 0.0;

        for(unsigned int i = 0; i < n_valid_rows; i++){
            conjugate[idx_t] += submat[n_valid_rows * idx_t + i] * residual[i];
        }

        residual[idx_t] /= normalize_residual;
        conjugate[idx_t] *= residual[idx_t];
        residual[idx_t] *= residual[idx_t];

        for(unsigned int i = 1; i < n_valid_rows; i++){
            residual[0] += residual[i];
            conjugate[0] += conjugate[i];
        }

        if(conjugate[0] != 0){
            alpha = residual[0] / conjugate[0];
        }

        x0[idx_t] = (submat[n_valid_rows * one_row_idx + idx_t] / normalize_residual) + alpha * residual[idx_t];

        for(unsigned int i = 0; i < n_valid_rows; i++){
            normalize_x += submat[n_valid_rows * one_row_idx + i] * x0[i];
        }

        if(normalize_x != 0){
            x0[idx_t] /= normalize_x;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(idx_t == 0){
        unsigned int p = 0;
        for(unsigned int current = first; current < last; current++){
            if(rowIndex[current] == J[current]){
                x[current] = x0[p];
                p++;
            }
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
