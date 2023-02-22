#pragma once
#include <cuda.h>
#include "nvshmem.h"
#include "nvshmemx.h"


__device__ __forceinline__
float atomic_add(float* const __restrict__ address,
                 const float val,
                 const int pe){

  int* address_as_int = (int*)address;
  int assumed; 
  int old = nvshmem_int_g(address_as_int, pe);
  do
  {
    assumed = old;
    old = nvshmem_int_atomic_compare_swap(address_as_int, assumed,
                                          __float_as_int(val +
                                                         __int_as_float(assumed)),
                                          pe);
  } while (assumed !=old);
  return __int_as_float(old);
}


__device__ __forceinline__ 
double atomic_add(double* const __restrict__ address,
                  const double val,
                  const int pe){

  long long int* address_as_ll = (long long int*)address;
  long long int assumed; 
  long long int old = nvshmem_longlong_g(address_as_ll, pe);
  do
  {
    assumed = old;
    old = nvshmem_longlong_atomic_compare_swap(address_as_ll, assumed,
                                               __double_as_longlong(val +
                                                                    __longlong_as_double(assumed)),
                                               pe);
  }while(assumed != old);
  return __longlong_as_double(old);
}


 __global__
void scatter(
  const float* __restrict__ values,
  const int* __restrict__ indices,
  float* __restrict__  output,
  const int mini_batch_size,
  const int num_values_rows, 
  const int num_cols,
  const int num_output_rows){
  
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;

  const size_t nthreadsx = gridDim.x * blockDim.x;
  const size_t nthreadsy = gridDim.y * blockDim.y;
  const size_t nthreadsz = gridDim.z * blockDim.z;

  for(size_t mb_i = gidz; mb_i < mini_batch_size; mb_i += nthreadsz){
    const auto values_offset = mb_i * num_cols * num_values_rows;
    const auto output_offset = mb_i * num_cols * num_output_rows;
    const auto ind_offset = mb_i * num_values_rows; 
    
    for(size_t row = gidy; row < num_values_rows; row += nthreadsy){
      const int ind = indices[ind_offset + row];
      
      for(size_t i = gidx; i < num_cols; i+= nthreadsx){  
        if (ind > 0 && ind < num_output_rows){
          const auto val = values[values_offset + row * num_cols + i];
          atomicAdd(&output[output_offset + ind * num_cols + i], val);
        }
      }
    }
  }
}

 __global__
void scatter(
  const float* __restrict__ values,
  const float* __restrict__ indices,
  float* __restrict__  output,
  const int mini_batch_size,
  const int num_values_rows, 
  const int num_cols,
  const int num_output_rows){
  
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;

  const size_t nthreadsx = gridDim.x * blockDim.x;
  const size_t nthreadsy = gridDim.y * blockDim.y;
  const size_t nthreadsz = gridDim.z * blockDim.z;

  for(size_t mb_i = gidz; mb_i < mini_batch_size; mb_i += nthreadsz){
    const auto values_offset = mb_i * num_cols * num_values_rows;
    const auto output_offset = mb_i * num_cols * num_output_rows;
    const auto ind_offset = mb_i * num_values_rows; 
    
    for(size_t row = gidy; row < num_values_rows; row += nthreadsy){
      const int ind = __float2int_rd(indices[ind_offset + row]);
      
      for(size_t i = gidx; i < num_cols; i+= nthreadsx){  
        if (ind > 0 && ind < num_output_rows){
          const auto val = values[values_offset + row * num_cols + i];
          atomicAdd(&output[output_offset + ind * num_cols + i], val);
        }
      }
    }
  }
}


__global__
void gather(
  const float* __restrict__ values,
  const int* __restrict__ indices,
  float* __restrict__  output,
  const int mini_batch_size,
  const int num_values_rows, 
  const int num_cols,
  const int num_output_rows){
  
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;

  const size_t nthreadsx = gridDim.x * blockDim.x;
  const size_t nthreadsy = gridDim.y * blockDim.y;
  const size_t nthreadsz = gridDim.z * blockDim.z;

  for(size_t mb_i = gidz; mb_i < mini_batch_size; mb_i += nthreadsz){
    const auto values_offset = mb_i * num_cols * num_values_rows;
    const auto output_offset = mb_i * num_cols * num_output_rows;
    const auto ind_offset = mb_i * num_output_rows;
    for(size_t row = gidy; row < num_output_rows; row += nthreadsy){
      const int ind = indices[ind_offset + row];
      if (ind > 0 && ind < num_values_rows){
        for(size_t i = gidx; i < num_cols; i+= nthreadsx){
          output[output_offset + row * num_cols + i] = values[values_offset + ind * num_cols + i];
        }
      }
    }
  }
}


template <typename DataType>
__global__ void Scatter_NVSHMEM_Kernel(
    const DataType* __restrict__ values,
    const DataType* __restrict__ indices,
    DataType* __restrict__ outputs,
    const int mini_batch_size,
    const int num_local_values_rows,
    const int num_cols,
    const int num_local_output_rows){
  // Indices
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;

  const size_t nthreadsx = gridDim.x * blockDim.x;
  const size_t nthreadsy = gridDim.y * blockDim.y;
  const size_t nthreadsz = gridDim.z * blockDim.z;

  for (size_t mb_i = gidz; mb_i < mini_batch_size; mb_i += nthreadsz){
    const auto values_offset = mb_i * num_local_values_rows * num_cols;
    const auto output_offset = mb_i * num_local_output_rows * num_cols;
    const auto indices_offset = mb_i * num_local_values_rows;

    for(size_t row = gidy; row < num_local_values_rows; row += nthreadsy){
      // Figure out which rank to send the vector
      const auto ind = __float2int_rd(indices[indices_offset + row]);
      if (ind > -1){
        const int pe = (ind) / num_local_output_rows;
        const int local_ind = ind % num_local_output_rows;
        for(size_t i = gidx; i < num_cols; i+= nthreadsx){
          const auto val = values[values_offset + row * num_cols + i];
          atomic_add(outputs + output_offset + local_ind * num_cols + i, val, pe);
        }
      }
    }
  }
}


template <typename DataType>
__global__ void Gather_NVSHMEM_Kernel(
  const DataType* __restrict__ values,
  const DataType* __restrict__ indices,
  DataType* __restrict__ shared_buffer,
  const int mini_batch_size,
  const int num_local_values_rows,
  const int num_local_cols,
  const int num_local_output_rows){

  // Indice
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  
  const size_t nthreadsy = gridDim.y * blockDim.y;
  const size_t nthreadsx = gridDim.x * blockDim.x;
  
  const int n_pes = nvshmem_n_pes();

  for (size_t mb_i = gidy; mb_i < mini_batch_size; mb_i += nthreadsy){
    // Figure out which rank to send the vector
    const auto mb_offset = mb_i*num_local_cols * num_local_output_rows;
    const auto values_offest = mb_i*num_local_cols * num_local_values_rows;
    const auto ind_offset = mb_i*num_local_output_rows;
    
    for(size_t row = gidx; row < num_local_output_rows; row += nthreadsx){
      const auto ind = __float2int_rd(indices[ind_offset + row]);
      if (ind > -1 ){ 
        const int pe = (ind) / num_local_values_rows;
        const int local_ind = ind % num_local_values_rows;
        nvshmem_getmem_nbi(shared_buffer + mb_offset + row * num_local_cols,
                           values + values_offest + local_ind * num_local_cols,
                           num_local_cols * sizeof(DataType),
                           pe);
      }
    }
  }
}


template <typename DataType>
__global__ void Gather_NVSHMEM_Kernel_Warp(
  const DataType* __restrict__ values,
  const DataType* __restrict__ indices,
  DataType* __restrict__ shared_buffer,
  const int mini_batch_size,
  const int num_local_values_rows,
  const int num_local_cols,
  const int num_local_output_rows){
  
  constexpr int warp_size = 32;
  // Indice
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  
  const size_t nthreadsy = gridDim.y * blockDim.y;
  const size_t nthreadsx = gridDim.x * blockDim.x;
  
  const int n_pes = nvshmem_n_pes();

  for (size_t mb_i = gidy; mb_i < mini_batch_size; mb_i += nthreadsy){
    // Figure out which rank to send the vector
    const auto mb_offset = mb_i*num_local_cols * num_local_output_rows;
    const auto values_offest = mb_i*num_local_cols * num_local_values_rows;
    const auto ind_offset = mb_i*num_local_output_rows;
    
    for(size_t row = gidx; row < num_local_output_rows * warp_size; row += nthreadsx){
      const auto ind = __float2int_rd(indices[ind_offset + row / warp_size]);
      if (ind > -1 ){ 
        const int pe = (ind) / num_local_values_rows;
        const int local_ind = ind % num_local_values_rows;
        nvshmemx_getmem_nbi_warp(shared_buffer + mb_offset + (row / warp_size) * num_local_cols,
                           values + values_offest + local_ind * num_local_cols,
                           num_local_cols * sizeof(DataType),
                           pe);
      }
    }
  }
}


template <typename DataType>
__global__ void generate_local_indices(
  const DataType* __restrict__ global_indices,
  DataType* local_indices,
  const int num_values){

}

template <typename DataType>
__global__ void recursive_reduce(
  const DataType* __restrict__ values,
  DataType* __restrict__ local_reduction,
  const DataType* __restrict__ indices,
  DataType* __restrict__ remapped_indices,
  const int mini_batch_size,
  const int num_values_rows,
  const int num_values_cols){
  
  // Indices
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;

  const size_t nthreadsx = gridDim.x * blockDim.x;
  const size_t nthreadsy = gridDim.y * blockDim.y;
  const size_t nthreadsz = gridDim.z * blockDim.z;

  for (size_t mb_i = gidz; mb_i < mini_batch_size; mb_i += nthreadsz){
    const auto values_offset = mb_i * num_values_rows * num_values_cols;
    const auto indices_offset = mb_i * num_values_rows;

    for(size_t row = gidy; row < num_values_rows / 2; row += nthreadsy){
      // Figure out which rank to send the vector
      const auto ind_1 = __float2int_rd(indices[indices_offset + 2*row]);
      const auto ind_2 = __float2int_rd(indices[indices_offset + 2*row + 1]);

      const auto row_offset_1 = 2 * row * num_values_cols;
      const auto row_offset_2 = (2 * row + 1) * num_values_cols;
      for(size_t i = gidx; i < num_values_cols; i+= nthreadsx){
        local_reduction[values_offset + row_offset_1 + i] = values[values_offset + row_offset_1 + i];
        remapped_indices[indices_offset + 2 * row] = __int2float_rd(ind_1);

        if (ind_1 == ind_2){
          local_reduction[values_offset + row_offset_1 + i] += values[values_offset + row_offset_2 + i];
          remapped_indices[indices_offset + 2 * row + 1] = __int2float_rd(-1);
        }else{
          local_reduction[values_offset + row_offset_2 + i] = values[values_offset + row_offset_2 + i];
          remapped_indices[indices_offset + 2 * row + 1] = __int2float_rd(ind_2);
        }
      } 
    }
  }
}
