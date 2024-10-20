/*
 *  Copyright 2008-2014 NVIDIA Corporation
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

#include <cusp/iterator/iterator_traits.h>

#include <thrust/device_allocator.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/iterator_traits.h>
#if THRUST_VERSION >= 200500
#include <cuda/std/type_traits>
#endif

#include <memory>

namespace cusp
{

#if THRUST_VERSION >= 200500
template<typename T, typename MemorySpace>
struct default_memory_allocator
        : thrust::detail::eval_if<
        ::cuda::std::is_same<MemorySpace, host_memory>::value,
        thrust::detail::identity_< std::allocator<T> >,
        thrust::detail::identity_< thrust::device_malloc_allocator<T> >
        >
{};
#else
template<typename T, typename MemorySpace>
struct default_memory_allocator
        : thrust::detail::eval_if<
        thrust::detail::is_same<MemorySpace, host_memory>::value,
        thrust::detail::identity_< std::allocator<T> >,
        thrust::detail::identity_< thrust::device_malloc_allocator<T> >
        >
{};
#endif

template <typename MemorySpace1, typename MemorySpace2, typename MemorySpace3, typename MemorySpace4>
struct minimum_space
{
    typedef typename thrust::detail::minimum_system<MemorySpace1, MemorySpace2, MemorySpace3, MemorySpace4>::type type;
};

} // end namespace cusp

