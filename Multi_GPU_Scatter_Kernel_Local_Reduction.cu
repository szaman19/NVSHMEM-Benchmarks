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
  
  int my_rank_in_node = std::stoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  int my_device = my_rank_in_node % 4;

  CUDA_CHECK(cudaSetDevice(my_device));
  
  CUDA_CHECK(cudaStreamCreate(&stream));
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

  CUDA_CHECK(cudaDeviceSynchronize());
  nvshmemx_barrier_all_on_stream(stream);
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("[%d / %d] Cuda device set to [%d]\n", rank, nranks, my_device);
  
  Edges edges = Read_Arxiv_Data();
  constexpr int num_nodes = 169343; 

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
  constexpr int count = 10;
  
  if (rank == 0){  
    std::string nranks_string = std::to_string(nranks);
    std::string output_dir = "scatter_results/";
    std::string suffix = "_scatter_arxiv_local_reduction.csv";
    std::string _fname = output_dir + nranks_string + suffix;
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

    std::cout <<"["<<rank <<"]" << " Starting generating random matrix: \t" << input_matrix_size << std::endl;
    float* value_matrix = _rand_arr_nvshmem(input_matrix_size);
    std::cout <<"["<<rank <<"]" << " Finished generating random matrix" << std::endl;
    const int* source_indices = &edges.get_source_indices()[input_start];
    float* local_indices = new float[num_indices];
    float* global_remap = new float[num_indices];

    if (rank == 0){
      std::cout << "[0] Starting remapping" << std::endl;
    }
    auto start_remap = std::chrono::steady_clock::now();
    remap_indices(source_indices, local_indices, global_remap, num_indices);
    auto elapsed_time_remap = static_cast<double>(since(start_remap).count());
    
    if (rank == 0){
      std::cout << "[0] Finished remapping" << std::endl;
    }
    float* local_indices_vec = _int2float(source_indices, num_indices);
    float* indices_vec = _device_arr(global_remap, num_indices);
    float* output_matrix = _zero_arr_nvshmem(output_matrix_size);
    float* local_value_matrix = _zero_arr_nvshmem(input_matrix_size);
    // Pad the indices_vector so    
    
    indices_vec = _pad(indices_vec, num_indices, input_stride, -1.0);
    local_indices_vec = _pad(local_indices_vec, num_indices, input_stride, -1.0);


    cudaDeviceSynchronize();
    if (rank == 0){
      std::cout << "[0] Finished setup" << std::endl;
    }
    nvshmemx_barrier_all_on_stream(stream);
    if (rank == 0){
      std::cout << "[0] Starting local scatter" << std::endl;
    }
    auto start = std::chrono::steady_clock::now();
    dim3 grid, block;
    block.x = 16;
    block.y = 32;
    block.z = 1; 

    grid.x = (num_cols + block.x - 1) / block.x;
    grid.y = (num_in_rows + block.y - 1) / block.y;
    grid.z = 1; 

    scatter<<<grid, block, 0, stream>>>(value_matrix,
                                        local_indices_vec,
                                        local_value_matrix,
                                        mini_batch_size,
                                        num_in_rows,
                                        num_cols,
                                        num_out_rows);

    Scatter_NVSHMEM_Kernel<<<grid, block, 0, stream>>>(local_value_matrix,
                                                       indices_vec,
                                                       output_matrix,
                                                       mini_batch_size,
                                                       num_in_rows,
                                                       num_cols,
                                                       num_out_rows);
    
    nvshmemx_quiet_on_stream(stream);
    nvshmemx_barrier_all_on_stream(stream);
    cudaStreamSynchronize(stream);

    auto elapsed_time = static_cast<double>(since(start).count());
    if (rank == 0){
      timing_array.push_back(elapsed_time + elapsed_time_remap);
      outfile<<","<<elapsed_time + elapsed_time_remap;
      std::cout << "Scatter Elapsed Time (S)=" << (elapsed_time + elapsed_time_remap) * 1E-9 << std::endl;
    }

    nvshmem_free(value_matrix);
    nvshmem_free(local_value_matrix);
    cudaFree(local_indices_vec);
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

