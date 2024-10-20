/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

/*! \file defs.h
 *  \brief CBLAS utility definitions for interface routines
 */

#pragma once

#include <cublas_v2.h>

#if THRUST_VERSION >= 200500
#include <cuda/std/type_traits>
#endif

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace cublas
{

struct cublas_format {};

struct upper   : public cublas_format {};
struct lower   : public cublas_format {};
struct unit    : public cublas_format {};
struct nonunit : public cublas_format {};

struct cublas_transpose_op { const static cublasOperation_t order = CUBLAS_OP_T; };
struct cublas_normal_op    { const static cublasOperation_t order = CUBLAS_OP_N; };

#if THRUST_VERSION >= 200500
template< typename LayoutFormat >
struct Orientation : thrust::detail::eval_if<
                          ::cuda::std::disjunction<
                              ::cuda::std::is_same<LayoutFormat, cusp::row_major_base<thrust::detail::true_type> >,
                              ::cuda::std::is_same<LayoutFormat, cusp::column_major_base<thrust::detail::false_type> >
                          >::value, // end or_
                          thrust::detail::identity_<cublas_normal_op>,
                          thrust::detail::identity_<cublas_transpose_op>
                       >
{};
#else
template< typename LayoutFormat >
struct Orientation : thrust::detail::eval_if<
                          thrust::detail::or_<
                              thrust::detail::is_same<LayoutFormat, cusp::row_major_base<thrust::detail::true_type> >,
                              thrust::detail::is_same<LayoutFormat, cusp::column_major_base<thrust::detail::false_type> >
                          >::value, // end or_
                          thrust::detail::identity_<cublas_normal_op>,
                          thrust::detail::identity_<cublas_transpose_op>
                       >
{};
#endif

} // end namespace cublas
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

