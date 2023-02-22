#include "test_utils.hpp"
#include "NVSHMEM_Kernels.cuh"
#include <iostream>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>

int 
main(int argc, char * argv[]){
  // RVO don't fail me now
  Edges edges = Read_Arxiv_Data();

  std::cout << edges.num_edges() << std::endl;

  const int num_edges = edges.num_edges();
  constexpr int num_nodes = 169343;

  constexpr int num_cols = 64;
  constexpr int weights_size = num_cols * num_cols; 


  constexpr int mini_batch_size = 1;
  const auto input_matrix_size = num_cols * num_nodes;
  const auto neighbor_matrix_size = num_cols * num_edges;
  
  std::ofstream outfile;
  outfile.open("1_scatter_arxiv.csv");
  constexpr int count = 5;
  for (size_t i = 0; i < count; i++){
    outfile << ", Run " << i+1;
  }
  outfile<<'\n';

  std::vector<double> timing_array; 

  for (size_t i = 0; i < count; i++){    
    float* value_matrix = _rand_arr(input_matrix_size);
    float* weights_matrix = _rand_arr(weights_size);

    const int* source_indices_vec = _to_device_ptr<int>(edges.get_source_indices(), num_edges);
    const int* target_indices_vec = _to_device_ptr<int>(edges.get_target_indices(), num_edges);

    float* gather_output_matrix = _zero_arr(neighbor_matrix_size);
    float* matmul_output_matrix = _zero_arr(neighbor_matrix_size);
    float* scatter_out_matrix = _zero_arr(input_matrix_size);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetStream(handle, stream);

    /*///////////////////////////////////////
     * cublas constants
     * 
     */////////////////////////////////////
    
    constexpr float alpha = 1;
    constexpr float beta = 0;
    const int m = num_edges;
    const int k = num_cols;
    const int n = num_cols;
    /////////////////////////////////////////////
    
    auto start = std::chrono::steady_clock::now();
    dim3 grid, block;
    block.x = 8;
    block.y = 32;
    block.z = mini_batch_size; 

    grid.x = 1;
    grid.y = (num_edges + block.y - 1) / block.y;
    grid.z = 1;
    cudaProfilerStart();
    gather<<<grid, block, 0, stream>>>(value_matrix,
                                       target_indices_vec,
                                       gather_output_matrix,
                                       mini_batch_size,
                                       num_nodes,
                                       num_cols,
                                       num_edges);

    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m, n, k,
                &alpha,
                gather_output_matrix, m,
                weights_matrix, k,
                &beta,
                matmul_output_matrix, m);

    scatter<<<grid, block, 0, stream>>>(matmul_output_matrix,
                                        source_indices_vec,
                                        scatter_out_matrix,
                                        mini_batch_size,
                                        num_edges,
                                        num_cols,
                                        num_nodes);
    
    cudaStreamSynchronize(stream);
    cudaProfilerStop();
    auto elapsed_time = static_cast<double>(since(start).count());
    timing_array.push_back(elapsed_time);

    cudaStreamDestroy(stream);
    cublasDestroy(handle) ; 
    outfile<<","<<elapsed_time;
    cudaFree(value_matrix);
    cudaFree(weights_matrix);

    cudaFree(const_cast<int*>(source_indices_vec));
    cudaFree(const_cast<int*>(target_indices_vec));

    cudaFree(gather_output_matrix);
    cudaFree(matmul_output_matrix);
    cudaFree(scatter_out_matrix);

  }
  outfile<<'\n';
  std::cout << "Scatter Elapsed Time (S)=" << average(timing_array) * 1E-9 << std::endl;
  outfile.close();
  return 0; 
}
