#include "test_utils.hpp"
#include "NVSHMEM_Kernels.cuh"
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include "nvml.h"
#include "mpi.h"



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
  
  Edges edges = Read_Shift_Data();
  constexpr int num_nodes = 1166243; 
  // constexpr int num_edges = 1166243;
  const int input_stride = (edges.num_edges() + nranks - 1) / nranks;
  const int output_stride = (num_nodes - 1 + nranks) / nranks;

  const int input_start = input_stride * rank;
  const int output_start = output_stride * rank;

  const int num_in_rows = input_stride;
  const int num_out_rows = output_stride;

  constexpr int num_cols = 128;
  constexpr int mini_batch_size = 1;
  const auto input_matrix_size = num_cols * num_in_rows;
  const auto output_matrix_size = num_cols * num_out_rows;
  
  std::ofstream outfile;
  constexpr int count = 1;
  
  if (rank == 0){  
    std::string nranks_string = std::to_string(nranks);
    std::string _fname = nranks_string +"_scatter_arxiv.csv";
    outfile.open(_fname);
    for (size_t i = 0; i < count; i++){
      outfile << ", Run " << i+1;
    }
    outfile<<'\n';
  }

  std::vector<double> timing_array;

  const int num_indices = (rank == nranks-1) ? edges.num_edges() - input_start : input_stride;

  printf("[%d] Reading ARXIV indices : %d ([%d]-[%d]) \n", rank, num_indices, output_start, output_start+num_indices);

  for(size_t i = 0; i < count; ++i){
    float* value_matrix = _rand_arr_nvshmem(input_matrix_size);
    float* indices_vec = _int2float(&edges.get_source_indices()[input_start], num_indices);
    float* output_matrix = _zero_arr_nvshmem(output_matrix_size);
    
    // Pad the indices_vector so    
    
    indices_vec = _pad(indices_vec, num_indices, input_stride, -1.0);

    cudaDeviceSynchronize();
//    nvshmemx_barrier_all_on_stream(stream);

    auto start = std::chrono::steady_clock::now();
    dim3 grid, block;
    block.x = 16;
    block.y = 32;
    block.z = 1; 

    grid.x = (num_cols + block.x - 1) / block.x;
    grid.y = (num_in_rows + block.y - 1) / block.y;
    grid.z = 1; 

    Scatter_NVSHMEM_Kernel<<<grid, block, 0, stream>>>(value_matrix,
                                                       indices_vec,
                                                       output_matrix,
                                                       mini_batch_size,
                                                       num_in_rows,
                                                       num_cols,
                                                       num_out_rows);
    
    // nvshmemx_quiet_on_stream(stream);
    nvshmemx_barrier_all_on_stream(stream);
    cudaDeviceSynchronize();

    auto elapsed_time = static_cast<double>(since(start).count());
    if (rank == 0){
      timing_array.push_back(elapsed_time);
      outfile<<","<<elapsed_time;
      std::cout << "Scatter Elapsed Time (S)=" << elapsed_time * 1E-9 << std::endl;
    }

    nvshmem_free(value_matrix);
    cudaFree(indices_vec);
    nvshmem_free(output_matrix);
    
  }

  nvshmemx_barrier_all_on_stream(stream); 
  cudaDeviceSynchronize();
  if (rank == 0){
    outfile<<'\n';
    std::cout << "Average Scatter Elapsed Time (S)=" << average(timing_array) * 1E-9 << std::endl;
    outfile.close();
  }

  nvshmem_finalize();
  MPI_Finalize();
  return 0;
}

