#ifndef __KERNEL_H_
#define __KERNEL_H_

const char* find_max_s = R"(
__kernel void find_max(__global const double *vals,
                       __global const int *rowIndex,
                       __global const int *colPtr,
                       __global const double *maxvals,
                       __local double *tmp,
                       const int row_offset,
                       const int N){
    const unsigned int warpsize = 32;
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);
    const unsigned int NUM_THREADS = get_global_size(0);
    const unsigned int num_warps_in_grid = NUM_THREADS / warpsize;
    const unsigned int target_col = gid / row_offset;
    const unsigned int lane = lid % row_offset;

    while(target_col < N){
        unsigned int first_row = colPtr[target_col];
        unsigned int last_row = colPtr[target_col + 1];
        unsigned int row = first_row + lane;

        double local_max = 0;
        for(; row < last_row; row += row_offset){
            if(abs(vals[row]) > local_max){
                local_max = abs(vals[row]);
            }
        }

        tmp[lane] = local_max;
        barrier(CLK_LOCAL_MEM_FENCE);

        for(unsigned int offset = row_offset - 1; offset > 0; offset--){
            if(tmp[lane + offset] > temp[lane]){
                tmp[lane] = tmp[lane + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(lane == 0){
            maxvals[target_col] = tmp[lane];
        }

        target_col += num_warps_in_grid;
    }
}
)";

const char* findJ_s = R"(
__kernel void findJ(__global const double *vals,
                    __global const int *rowIndex,
                    __global const int *colPtr,
                    __global const double *maxvals,
                    const int n2max,
                    const int n,
                    const int warpSize,
                    const double tau,
                    __global int *J,
                    __global int *Jptr,
                    __local int *sJ){
    int gid = get_global_id(0);
    int warpId = gid/warpSize;
    int offset = get_local_size(0)/(warpSize * get_num_groups(0));
    int lane = gid & (warpSize - 1);
    int tid = get_local_id(0)/warpSize;

    for(int col = warpId; col < n; col += offset){
        int colBegin = colPtr[col];
        int colEnd = colPtr[col + 1];
        int nnzCol = colEnd - colBegin;

        for(int j = lane; j < nnzCol - 1; j++){
            if(rowIndex[colBegin + j] == col){
                sJ[tid * n2max + j] = col;
            }
            else if(abs(vals[colBegin + j]) > (1 - tau)*maxvals[col]){
                sJ[tid * n2max + j] = rowIndex[colBegin + j];
            }
            else{
                sJ[tid * n2max + j] = -1;
            }
        }

        if(lane == 0){
            Jptr[col] = 0;
            for(int j = 0; j < valSize - 1; j++){
                if(sJ[tid * n2max + j] != -1){
                    J[col * n2max + Jptr[col]] = sJ[tid * n2max + j];
                    Jptr[col]++;
                }
            }
        }
    }
}
)";

const char* sat_block_frobenius_s = R"(
__kernel void sat_block_frobenius(
    __global const double *vals,
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

#endif // __KERNEL_H_
