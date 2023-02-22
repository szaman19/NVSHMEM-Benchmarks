#include "test_utils.hpp"
#include "NVSHMEM_Kernels.cuh"
#include <stdio.h>
#include <cuda.h>
#include <numeric>
#include "nvml.h"
#include "mpi.h"
#include <cublas_v2.h>


int main(int argc, char* argv[]){
  int mype_node;
  cudaStream_t stream;
  int rank, nranks;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  cudaSetDevice(mype_node);
  cudaStreamCreate(&stream);
  
  printf("[%d / %d] Cuda device set to [%d]\n", rank, nranks, mype_node);
  
  Edges edges = Read_Arxiv_Data();
  constexpr int num_nodes = 169343; 
  const int num_edges = edges.num_edges();

  const int nodes_per_rank = (num_nodes + nranks - 1) / nranks;
  const int edges_per_rank = (num_edges - 1 + nranks) / nranks;


  // const int node_offset = nodes_per_rank * rank;
  const int edge_offset = edges_per_rank * rank;

  constexpr int num_cols = 64;
  constexpr int mini_batch_size = 1;
  constexpr int weights_size = num_cols * num_cols; 

  const auto input_matrix_size = num_cols * nodes_per_rank;
  const auto gather_matrix_size = num_cols * edges_per_rank;
  const auto matmul_matrix_size = num_cols * edges_per_rank;
  // const auto scatter_matrix_size = num_cols * nodes_per_rank;
  const auto scatter_matrix_size = num_cols * edges_
  
  std::ofstream outfile;
  constexpr int count = 5;
  
  if (rank == 0){  
    std::string nranks_string = std::to_string(nranks);
    std::string _fname = nranks_string +"_GCN_arxiv.csv";
    outfile.open(_fname);
    for (size_t i = 0; i < count; i++){
      outfile << ", Run " << i+1;
    }
    outfile<<'\n';
  }

  std::vector<double> timing_array;

  const int indices_per_rank = (rank == nranks-1) ? num_edges - edge_offset : edges_per_rank;


  printf("[%d] Reading ARXIV indices : %d ([%d]-[%d]) \n", rank, indices_per_rank, edge_offset, edge_offset+indices_per_rank);

  for(size_t i = 0; i < count; ++i){
    float* value_matrix = _rand_arr_nvshmem(input_matrix_size);
    float* weights_matrix = _rand_arr(weights_size);
    // float* source_indices_vec = _int2float(&edges.get_source_indices()[edge_offset], indices_per_rank);
    // float* target_indices_vec = _int2float(&edges.get_target_indices()[edge_offset], indices_per_rank);

    float* source_indices_vec =_shift__indices_arr_nvshmem(indices_per_rank, edges_per_rank * nranks);
    float* target_indices_vec = _shift__indices_arr_nvshmem(indices_per_rank, edges_per_rank * nranks);
    float* gather_output_matrix = _zero_arr_nvshmem(gather_matrix_size);
    float* matmul_output_matrix = _zero_arr(matmul_matrix_size);
    float* scatter_output_matrix = _zero_arr_nvshmem(scatter_matrix_size);
   
    // Pad the indices_vectors
    source_indices_vec = _pad(source_indices_vec, indices_per_rank, edges_per_rank, -1.0);
    target_indices_vec = _pad(target_indices_vec, indices_per_rank, edges_per_rank, -1.0);

    
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetStream(handle, stream);

    /*///////////////////////////////////////
     * cublas constants
     * 
     */////////////////////////////////////
    
    constexpr float alpha = 1;
    constexpr float beta = 0;
    const int m = edges_per_rank;
    const int k = num_cols;
    const int n = num_cols;
    /////////////////////////////////////////////
    nvshmemx_barrier_all_on_stream(stream);
    cudaDeviceSynchronize();
     
    auto start = std::chrono::steady_clock::now();
    dim3 grid, block;
    block.x = 32;
    block.y = mini_batch_size;
    block.z = 1; 

    grid.x = (edges_per_rank + block.x - 1) / block.x;
    grid.y = 1;
    grid.z = 1; 
    Gather_NVSHMEM_Kernel<<<grid, block, 0, stream>>>(value_matrix,
                                                      target_indices_vec,
                                                      gather_output_matrix,
                                                      mini_batch_size,
                                                      nodes_per_rank,
                                                      num_cols,
                                                      edges_per_rank);

    nvshmemx_barrier_all_on_stream(stream);
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m, n, k,
                &alpha,
                gather_output_matrix, m,
                weights_matrix, k,
                &beta,
                matmul_output_matrix, m);
    block.x = 8;
    block.y = 32;
    block.z = mini_batch_size;

    grid.x = (num_cols + block.x - 1) / block.x;
    grid.y = (edges_per_rank + block.y - 1) / block.y;
    grid.z = 1;
    // cuProfilerStart();
    Scatter_NVSHMEM_Kernel<<<grid, block, 0, stream>>>(matmul_output_matrix,
                                                       source_indices_vec,
                                                       scatter_output_matrix,
                                                       mini_batch_size,
                                                       edges_per_rank,
                                                       num_cols,
                                                       nodes_per_rank);
    
    // nvshmemx_quiet_on_stream(stream);
    nvshmemx_barrier_all_on_stream(stream);
    cudaStreamSynchronize(stream);

    auto elapsed_time = static_cast<double>(since(start).count());
    if (rank == 0){
      timing_array.push_back(elapsed_time);
      outfile<<","<<elapsed_time;
      std::cout << "GCN Elapsed Time (S)=" << elapsed_time * 1E-9 << std::endl;
    }
    
    cublasDestroy(handle); 
    nvshmem_free(value_matrix);
    gpuErrchk( cudaFree(weights_matrix) );
    gpuErrchk( cudaFree(source_indices_vec) );
    gpuErrchk( cudaFree(target_indices_vec) );

    nvshmem_free(gather_output_matrix) ;
    gpuErrchk( cudaFree(matmul_output_matrix) );
    nvshmem_free(scatter_output_matrix) ;
  }

  nvshmemx_barrier_all_on_stream(stream); 
  cudaDeviceSynchronize();
  if (rank == 0){
    outfile<<'\n';
    std::cout << "Average GCN Elapsed Time (S)=" << average(timing_array) * 1E-9 << std::endl;
    outfile.close();
  }

  nvshmem_finalize();
  MPI_Finalize();
  return 0;
}

