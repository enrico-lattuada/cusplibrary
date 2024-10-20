/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>

#include <cusp/system/cuda/detail/cuda_launch_config.h>

#if THRUST_VERSION >= 200400
#define __CCCL_DEVICE _CCCL_DEVICE
#define __CCCL_FORCEINLINE _CCCL_FORCEINLINE
#else
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#define __CCCL_DEVICE __CCCL_DEVICE
__CCCL_FORCEINLINE __CCCL_FORCEINLINE __thrust_forceinline__
#endif

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{

template <unsigned int _ThreadsPerBlock = 0,
          unsigned int _BlocksPerMultiprocessor = 0>
struct launch_bounds
{
  typedef thrust::detail::integral_constant<unsigned int, _ThreadsPerBlock>         ThreadsPerBlock;
  typedef thrust::detail::integral_constant<unsigned int, _BlocksPerMultiprocessor> BlocksPerMultiprocessor;
};

struct thread_array : public launch_bounds<>
{
// CUDA built-in variables require nvcc
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int thread_index(void) const { return threadIdx.x; }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int thread_count(void) const { return blockDim.x * gridDim.x; } 
#else
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int thread_index(void) const { return 0; }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int thread_count(void) const { return 0; } 
#endif // THRUST_DEVICE_COMPILER_NVCC
};

struct blocked_thread_array : public launch_bounds<>
{
// CUDA built-in variables require nvcc
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int thread_index(void)    const { return threadIdx.x; }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int block_dimension(void) const { return blockDim.x;  } 
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int block_index(void)     const { return blockIdx.x;  }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int grid_dimension(void)  const { return gridDim.x;   }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int linear_index(void)    const { return block_dimension() * block_index() + thread_index(); }
  __CCCL_DEVICE __CCCL_FORCEINLINE void         barrier(void)               { __syncthreads();    }
#else
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int thread_index(void)    const { return 0; }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int block_dimension(void) const { return 0; }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int block_index(void)     const { return 0; }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int grid_dimension(void)  const { return 0; }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int linear_index(void)    const { return 0; }
  __CCCL_DEVICE __CCCL_FORCEINLINE void         barrier(void)               {           }
#endif // THRUST_DEVICE_COMPILER_NVCC
};

template <unsigned int _ThreadsPerBlock>
struct statically_blocked_thread_array : public launch_bounds<_ThreadsPerBlock,1>
{
// CUDA built-in variables require nvcc
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int thread_index(void)    const { return threadIdx.x;      }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int block_dimension(void) const { return _ThreadsPerBlock; } // minor optimization
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int block_index(void)     const { return blockIdx.x;       }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int grid_dimension(void)  const { return gridDim.x;        }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int linear_index(void)    const { return block_dimension() * block_index() + thread_index(); }
  __CCCL_DEVICE __CCCL_FORCEINLINE void         barrier(void)               { __syncthreads();    }
#else
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int thread_index(void)    const { return 0; }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int block_dimension(void) const { return 0; }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int block_index(void)     const { return 0; }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int grid_dimension(void)  const { return 0; }
  __CCCL_DEVICE __CCCL_FORCEINLINE unsigned int linear_index(void)    const { return 0; }
  __CCCL_DEVICE __CCCL_FORCEINLINE void         barrier(void)               {           }
#endif // THRUST_DEVICE_COMPILER_NVCC
};

template<typename Closure, typename Size1, typename Size2>
  void launch_closure(Closure f, Size1 num_blocks, Size2 block_size);

template<typename Closure, typename Size1, typename Size2, typename Size3>
  void launch_closure(Closure f, Size1 num_blocks, Size2 block_size, Size3 smem_size);

/*! Returns a copy of the cudaFuncAttributes structure
 *  that is associated with a given Closure
 */
template <typename Closure>
function_attributes_t closure_attributes(void);

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

#include <cusp/system/cuda/detail/launch_closure.inl>
