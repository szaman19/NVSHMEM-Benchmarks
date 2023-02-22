NVSHMEM_HOME :=/p/vast1/zaman2/nvshmem_src_2.8.0-3/build
MPI_HOME :=/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-8.3.1
CUDA_HOME :=/usr/tce/packages/cuda/cuda-11.7.0
CC :=/usr/tce/packages/gcc/gcc-8.3.1/bin/g++
NVCC :=/usr/tce/packages/cuda/cuda-11.7.0/bin/nvcc
GENCODE_SM70  := -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS	:= $(GENCODE_SM70)


NVCC_FLAGS = -forward-unknown-to-host-compiler --expt-relaxed-constexpr -ccbin=$(CC)
NVCC_FLAGS += -std=c++17 -x cu -rdc=true $(GENCODE_FLAGS) -isystem=$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -Iinclude/
NVCC_LDFLAGS = -forward-unknown-to-host-compiler -ccbin=$(CC) -lgomp -L$(NVSHMEM_HOME)/lib \
								-lnvshmem -L$(MPI_HOME)/lib -lmpi_ibm -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvToolsExt -lnvidia-ml

executable = Single_GPU_Scatter Single_GPU_Gather Single_GPU_GCN \
						 Multi_GPU_Scatter Multi_GPU_Gather Multi_GPU_GCN \
						 Shift_Multi_Scatter Shift_Multi_Gather

all: Makefile  SINGLE_SCATTER SINGLE_GATHER MULTI_SCATTER MULTI_GATHER
	
SINGLE_SCATTER: Single_GPU_Scatter_Kernel.cu
	$(NVCC) $(NVCC_FLAGS) -c -fopenmp -O3 Single_GPU_Scatter_Kernel.cu -o Single_GPU_Scatter_Kernel.o
	$(NVCC) $(GENCODE_FLAGS) Single_GPU_Scatter_Kernel.o -o Single_GPU_Scatter $(NVCC_LDFLAGS)

SINGLE_GATHER: Single_GPU_Gather_Kernel.cu
	$(NVCC) $(NVCC_FLAGS) -c -fopenmp -O3 Single_GPU_Gather_Kernel.cu -o Single_GPU_Gather.o
	$(NVCC) $(GENCODE_FLAGS) Single_GPU_Gather.o -o Single_GPU_Gather $(NVCC_LDFLAGS)

SINGLE_GCN: Single_GPU_GCN_Kernel.cu
	$(NVCC) $(NVCC_FLAGS) -c -fopenmp -O3 Single_GPU_GCN_Kernel.cu -o Single_GPU_GCN_Kernel.o
	$(NVCC) $(GENCODE_FLAGS) Single_GPU_GCN_Kernel.o -o Single_GPU_GCN $(NVCC_LDFLAGS) -lcublas

MULTI_SCATTER: Multi_GPU_Scatter_Kernel.cu
	$(NVCC) $(NVCC_FLAGS) -O3 -fopenmp Multi_GPU_Scatter_Kernel.cu -c -o Multi_GPU_Scatter_Kernel.o
	$(NVCC) $(GENCODE_FLAGS) Multi_GPU_Scatter_Kernel.o -o Multi_GPU_Scatter $(NVCC_LDFLAGS)

MULTI_GATHER: Multi_GPU_Gather_Kernel.cu
	$(NVCC) $(NVCC_FLAGS) -O3 -fopenmp Multi_GPU_Gather_Kernel.cu -c -o Multi_GPU_Gather_Kernel.o
	$(NVCC) $(GENCODE_FLAGS) Multi_GPU_Gather_Kernel.o -o Multi_GPU_Gather $(NVCC_LDFLAGS) 

MULTI_GATHER_WARP: Multi_GPU_Gather_Kernel_Warp.cu
	$(NVCC) $(NVCC_FLAGS) -O3 -fopenmp Multi_GPU_Gather_Kernel_Warp.cu -c -o Multi_GPU_Gather_Kernel_Warp.o
	$(NVCC) $(GENCODE_FLAGS) Multi_GPU_Gather_Kernel_Warp.o -o Multi_GPU_Gather_Warp $(NVCC_LDFLAGS) 

NVSHMEM_TRIAL: nvshmem_ex.cu
	$(NVCC) $(NVCC_FLAGS) -O3 -fopenmp nvshmem_ex.cu -c -o test.o 
	$(NVCC) $(GENCODE_FLAGS) test.o -o test $(NVCC_LDFLAGS)

MULTI_GCN: Multi_GPU_GCN_Kernel.cu
	$(NVCC) $(NVCC_FLAGS) -O3 -fopenmp Multi_GPU_GCN_Kernel.cu -c -o Multi_GPU_GCN_Kernel.o
	$(NVCC) $(GENCODE_FLAGS) Multi_GPU_GCN_Kernel.o -o Multi_GPU_GCN $(NVCC_LDFLAGS) -lcublas

SHIFT_MULTI_SCATTER: Shift_Ind_Multi_Scatter.cu
	$(NVCC) $(NVCC_FLAGS) -O3 -fopenmp Shift_Ind_Multi_Scatter.cu -c -o Shift_Ind_Multi_Scatter.o
	$(NVCC) $(GENCODE_FLAGS) Shift_Ind_Multi_Scatter.o -o Shift_Multi_Scatter $(NVCC_LDFLAGS)

SHIFT_MULTI_GATHER: Shift_Ind_Multi_Gather.cu
	$(NVCC) $(NVCC_FLAGS) -O3 -fopenmp Shift_Ind_Multi_Gather.cu -c -o Shift_Ind_Multi_Gather.o
	$(NVCC) $(GENCODE_FLAGS) Shift_Ind_Multi_Gather.o -o Shift_Multi_Gather $(NVCC_LDFLAGS)

MULTI_LOCAL_SCATTER: Multi_GPU_Scatter_Kernel_Local_Reduction.cu
	$(NVCC) $(NVCC_FLAGS) -c -O3 -fopenmp Multi_GPU_Scatter_Kernel_Local_Reduction.cu -o Multi_GPU_Scatter_Kernel_Local_Reduction.o
	$(NVCC) $(GENCODE_FLAGS) Multi_GPU_Scatter_Kernel_Local_Reduction.o -o Multi_GPU_Scatter_Local $(NVCC_LDFLAGS)


.PHONY: clean

clean: 
	rm *.o 
