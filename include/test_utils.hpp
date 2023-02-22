#pragma once
#include <vector>
#include "nvshmem.h"
#include "nvshmemx.h"
#include <random>
#include <cuda.h>
#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

template <
    class result_t   = std::chrono::nanoseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::nanoseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

template<typename T>
T* 
cudaAlloc(int elements)
{ T* device_ptr;
  gpuErrchk( cudaMalloc((void**)&device_ptr, elements * sizeof(T)) );
  return device_ptr;
}

/**
 * @brief Generates a device array of the given size filled with random floats
 * 
 * @param size 
 * @return float* 
 */
float*
_rand_arr(int size){
  float* device_ptr = cudaAlloc<float>(size);
  
  float* host_ptr =  new float[size];
  
  unsigned int seed = 42;
  std::minstd_rand0 gen(seed);
  std::uniform_real_distribution<float>dis(0, 1);

  for(auto i = 0; i < size; ++i){
    host_ptr[i] = dis(gen);
  } 

  gpuErrchk( cudaMemcpy(device_ptr, host_ptr, size * sizeof(float), cudaMemcpyHostToDevice) );
  delete[] host_ptr;
  return device_ptr;
}

/**
 * @brief Generates a device array of the given size filled with random floats
 * 
 * @param size 
 * @return float* 
 */
float*
_rand_arr_nvshmem(int size){
  float* device_ptr = (float*)nvshmem_malloc(size * sizeof(float));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "Memory allocated " << std::endl;
  float* host_ptr =  new float[size];
  
  unsigned int seed = 42;
  std::minstd_rand0 gen(seed);
  std::uniform_real_distribution<float>dis(0, 1);

  #pragma omp for
  for(auto i = 0; i < size; ++i){
    host_ptr[i] = dis(gen);
  } 

  gpuErrchk( cudaMemcpy(device_ptr, host_ptr, size * sizeof(float), cudaMemcpyHostToDevice) );
  delete[] host_ptr;
  return device_ptr;
}


/**
 * @brief Generates a device array of the given size filled with 0
 * 
 * @param size 
 * @return float* 
 */
float*
_zero_arr(int size){
  float* device_ptr = cudaAlloc<float>(size);
  gpuErrchk( cudaMemset(device_ptr, 0, sizeof(float) * size) );
  return device_ptr;
}


/**
 * @brief Generates a device array of the given size filled with 0
 * 
 * @param size 
 * @return float* 
 */
float*
_zero_arr_nvshmem(int size){
  float* device_ptr = (float*)nvshmem_malloc(size * sizeof(float));
  gpuErrchk( cudaMemset(device_ptr, 0, sizeof(float) * size) );
  return device_ptr;
}


/**
 * @brief Generates a float device array of with size ranging from [-1, max_val) 
 * 
 * @param size 
 * @param max_val 
 * @return float* 
 */
float*
_indices_arr(int size, int max_val){
  float* device_ptr = cudaAlloc<float>(size);
  float* host_ptr =  new float[size];
  unsigned int seed = 42;
  std::minstd_rand0 gen(seed);
  std::uniform_int_distribution<> distrib(-1, max_val-1);

  for(auto i = 0; i < size; ++i){
    host_ptr[i] = static_cast<float>(distrib(gen));
  } 
  gpuErrchk( cudaMemcpy(device_ptr, host_ptr, size * sizeof(float), cudaMemcpyHostToDevice) );
  delete[] host_ptr;
  return device_ptr;
}


/**
 * @brief Generates a float device array of with size ranging from [-1, max_val) 
 * 
 * @param size 
 * @param max_val 
 * @return float* 
 */
float*
_indices_arr_nvshmem(int size, int max_val){
  float* device_ptr = (float*)nvshmem_malloc(size * sizeof(float));
  float* host_ptr =  new float[size];
  unsigned int seed = 42;
  std::minstd_rand0 gen(seed);
  std::uniform_int_distribution<> distrib(-1, max_val-1);

  for(auto i = 0; i < size; ++i){
    host_ptr[i] = static_cast<float>(distrib(gen));
  } 
  gpuErrchk( cudaMemcpy(device_ptr, host_ptr, size * sizeof(float), cudaMemcpyHostToDevice) );
  delete[] host_ptr;
  return device_ptr;
}

float*
_shift__indices_arr_nvshmem(int size, int max){
  float* device_ptr = (float*) nvshmem_malloc(size * sizeof(float));
  float *host_ptr = new float(size);

  for(auto i = 0; i < size; ++i){
    host_ptr[i] = static_cast<float>( (i + 1) % max );
  }
  gpuErrchk( cudaMemcpy(device_ptr, host_ptr, size * sizeof(float), cudaMemcpyHostToDevice) );
  delete[] host_ptr;
  return device_ptr;
}

float mem_size(int rows){
  return 128 * 64 * static_cast<float>(rows) * 4 / (1024.0 * 1024); 
}

double average(std::vector<double> const& v){
    if(v.empty()){
        return 0;
    }

    auto const count = static_cast<double>(v.size());
    return std::accumulate(v.begin(), v.end(), 0.0) / count;
}

struct Edges
{ 
  Edges(std::vector<int> source,
        std::vector<int> targets):m_source_indices(std::move(source)),
                                  m_target_indices(std::move(targets)){
                                    m_num_edges = m_source_indices.size();
                                  }

  std::vector<int> m_source_indices;
  std::vector<int> m_target_indices;
  int m_num_edges;

  const int* get_source_indices() const{
    return m_source_indices.data();
  }

  const int* get_target_indices() const{
    return m_target_indices.data();
  }

  int num_edges() const{
    return m_num_edges;
  } 
};

__global__
void _int2float_copy(float* device_ptr_f, const int* device_ptr_i, int num_elements){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreadsx = gridDim.x * blockDim.x;
  for (int i = idx; i < num_elements; i+=nthreadsx){
    device_ptr_f[i] = device_ptr_i[i];
  }
}

__global__
void _fill_arr_(float* arr, int num_elements, float val){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreadsx = gridDim.x * blockDim.x;
  for (int i = idx; i < num_elements; i+=nthreadsx){
    arr[i] = val;
  }
}

template<typename T>
T* 
_to_device_ptr(const T* host_ptr, int num_elements){
  T* device_ptr = cudaAlloc<T>(num_elements);
  gpuErrchk( cudaMemcpy(device_ptr, host_ptr, num_elements * sizeof(T), cudaMemcpyHostToDevice) );
  return device_ptr;
}

float*
_int2float(const int* host_ptr, int num_elements){
  int* device_ptr_int = _to_device_ptr<int>(host_ptr, num_elements);
  float* device_ptr_float = cudaAlloc<float>(num_elements);
  
  int block_size = 256;
  int num_blocks  = (num_elements +  block_size - 1) / (block_size);

  _int2float_copy<<<num_blocks, block_size>>>(device_ptr_float, device_ptr_int, num_elements);
  
  return device_ptr_float;
}


float*
_device_arr(float* host_ptr, int num_elements){
  float* device_ptr = cudaAlloc<float>(num_elements);
  gpuErrchk( cudaMemcpy(device_ptr, host_ptr, num_elements * sizeof(float), cudaMemcpyHostToDevice) );
  return device_ptr;
}


float*
_pad(float* device_ptr_org, int num_elems, int padded_size, float fill_value = 0.0){
  
  if (num_elems == padded_size){
    return device_ptr_org;
  }

  float* device_ptr  = cudaAlloc<float>(padded_size);
  int block_size = 256;
  int num_blocks  = (padded_size +  block_size - 1) / (block_size);

  _fill_arr_<<<num_blocks, block_size>>>(device_ptr, padded_size, fill_value);
  gpuErrchk( cudaMemcpy(device_ptr, device_ptr_org, num_elems * sizeof(float), cudaMemcpyDeviceToDevice) );

  gpuErrchk( cudaFree(device_ptr_org) );
  
  return device_ptr;
}


/**
 * @brief Reads
 * 
 * @param arr 
 * @param size 
 * @param mini_batch_size 
 */


Edges Read_Arxiv_Data(){
  std::string fileName = "arxiv/raw/edge.csv";
  std::string delim = ",";
  std::vector<int> _sources;
  std::vector<int> _targets;

  std::ifstream file;
  file.open(fileName);
  std::string line;
  int counter = 0;
  while(!file.eof()){
    counter += 1;

    std::getline(file, line);
    try{
      if(!line.empty()){
        std::istringstream inp;
        inp.str(line);
        std::string source_str, target_str;

        std::getline(inp, source_str, ',');
        std::getline(inp, target_str, ',');

      _sources.push_back(std::stoi(source_str));    
      _targets.push_back(std::stoi(target_str));
      }
    }
    catch(...){
      std::cout << "Exception occured parsing line: "<<  counter <<"\n";
      std::cout << line << "\n";
    }
    
  }
  file.close();
  Edges arxiv_edges(_sources, _targets);
  return arxiv_edges;
}

/**
 * @brief Reads in a line graph that is the same size as the number of edges onr Arxiv
 * 
 * @return Edges 
 */
Edges Read_Shift_Data(){
  std::string fileName = "shift_data.csv";
  std::string delim = ",";
  std::vector<int> _sources;
  std::vector<int> _targets;

  std::ifstream file;
  file.open(fileName);
  std::string line;
  int counter = 0;
  while(!file.eof()){
    counter += 1;

    std::getline(file, line);
    try{
      if(!line.empty()){
        std::istringstream inp;
        inp.str(line);
        std::string source_str, target_str;

        std::getline(inp, source_str, ',');
        std::getline(inp, target_str, ',');

      _sources.push_back(std::stoi(source_str));    
      _targets.push_back(std::stoi(target_str));
      }
    }
    catch(...){
      std::cout << "Exception occured parsing line: "<<  counter <<"\n";
      std::cout << line << "\n";
    }
    
  }
  file.close();
  Edges shift_edges(_sources, _targets);
  return shift_edges;
}

// class NodeFT_Matrix{
//   public:
//     float *data;
//     int num_rows;
//     int num_cols;
//     NodeFT_Matrix(int rows = 169343){

//     }

//     ~NodeFT_Matrix(){
//       delete[] data;
//     }
// };

// float*
// Get_Node_Data(int num_rows, int num_cols){
//   std::string fileName = "arxiv/raw/node-feat.csv";
//   std::string delim = ",";
//   float* data = new float[num_rows * num_cols];

//   std::ifstream file;
//   file.open(fileName);
//   std::string line;
//   for(auto i = 0; i < num_rows; ++i){
//     std::getline(file, line);

//     try{
//       if(!line.empty()){
//         std::istringstream inp;
//         inp.str(line);
//         for(auto j = 0; j < num_cols; ++j){
//           std::string float_str;
//           std::getline(inp, float_str, ',');
//           data[i * num_rows + j] = std::stof(float_str);
//         }
//       }
//     catch(...){
//       std::cout << "Exception occured parsing line: "<<  i <<"\n";
//       std::cout << line << "\n";
//     }
    
//   }

//   return data; 
// }

void 
print_device_arr(float* arr, int size, int mini_batch_size=1){
  float* host_ptr = new float[size];
  gpuErrchk( cudaMemcpy(host_ptr, arr, sizeof(float) * size, cudaMemcpyDeviceToHost) );

  for (size_t i = 0; i < mini_batch_size; i++)
  {
    for (size_t j = 0; j < size / mini_batch_size; j++)
    {
      /* code */
      auto ind = i * (size / mini_batch_size) + j;
      printf("%.1f, ", host_ptr[ind]);
    }
    printf("\n");
  }
  delete[] host_ptr;
}


void
remap_indices(const int* global_indices,
              float* local_indices,
              float* global_remap,
              const int size){
  //Some bookkeeping
  #pragma omp for
  for (auto i = 0; i < size; i++){
    local_indices[i] = -1.0;
    global_remap[i] = -1.0;
  }
  
  std::unordered_map<int, int> tracker;
  int cur_local_index = 0;  
  for(auto i = 0; i < size; ++i){
    int glob_ind = global_indices[i];

    if (glob_ind >= 0){
      auto search = tracker.find(glob_ind);
      if (search == tracker.end()){
        // global index not in tracker
        tracker[glob_ind] = cur_local_index;
        local_indices[i] = static_cast<float>(cur_local_index);
        global_remap[cur_local_index] = static_cast<float>(glob_ind);
        cur_local_index++;
      }else{
        local_indices[i] = static_cast<float>(tracker[glob_ind]);
      }
    }
  }
}
