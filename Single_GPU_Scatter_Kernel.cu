#include "test_utils.hpp"
#include "NVSHMEM_Kernels.cuh"
#include <iostream>


int 
main(int argc, char * argv[]){
  // RVO don't fail me now
  Edges edges = Read_Arxiv_Data();

  std::cout << edges.num_edges() << std::endl;

  const int num_in_rows = edges.num_edges();
  constexpr int num_out_rows = 169343;

  constexpr int num_cols = 64;
  constexpr int mini_batch_size = 1;
  const auto input_matrix_size = num_cols * num_in_rows;
  const auto output_matrix_size = num_cols * num_out_rows;
  
  std::ofstream outfile;
  outfile.open("1_scatter_arxiv.csv");
  constexpr int count = 1;
  for (size_t i = 0; i < count; i++){
    outfile << ", Run " << i+1;
  }
  outfile<<'\n';

  std::vector<double> timing_array; 

  for (size_t i = 0; i < count; i++){    
    float* value_matrix = _rand_arr(input_matrix_size);
    const int* indices_vec = _to_device_ptr<int>(edges.get_source_indices(), num_in_rows);
    float* output_matrix = _zero_arr(output_matrix_size);
    
    auto start = std::chrono::steady_clock::now();
    dim3 grid, block;
    block.x = 16;
    block.y = 32;
    block.z = mini_batch_size; 

    grid.x = 1;
    grid.y = (num_in_rows + block.y - 1) / block.y;
    grid.z = 1; 
    scatter<<<grid, block>>>(value_matrix,
                             indices_vec,
                             output_matrix,
                             mini_batch_size,
                             num_in_rows,
                             num_cols,
                             num_out_rows);

    cudaDeviceSynchronize();

    auto elapsed_time = static_cast<double>(since(start).count());
    timing_array.push_back(elapsed_time);
    outfile<<","<<elapsed_time;
    cudaFree(value_matrix);
    cudaFree(const_cast<int*>(indices_vec));
    cudaFree(output_matrix);

  }
  outfile<<'\n';
  std::cout << "Scatter Elapsed Time (S)=" << average(timing_array) * 1E-9 << std::endl;
  outfile.close();
  return 0; 
}
