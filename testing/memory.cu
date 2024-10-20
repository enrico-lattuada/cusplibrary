#include <unittest/unittest.h>

#include <cusp/memory.h>

#if THRUST_VERSION >= 200500
#include <cuda/std/type_traits>
#endif

void TestMinimumSpace(void)
{
    typedef cusp::host_memory   H;
    typedef cusp::device_memory D;
    typedef cusp::any_memory    A;

    #if THRUST_VERSION >= 200500
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<H,H>::type, H>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<H,A>::type, H>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<A,H>::type, H>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<D,D>::type, D>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<D,A>::type, D>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<A,D>::type, D>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<A,A>::type, A>::value), true);

    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<H,H,A>::type, H>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<H,A,A>::type, H>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<A,H,A>::type, H>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<D,D,A>::type, D>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<D,A,A>::type, D>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<A,D,A>::type, D>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<A,A,A>::type, A>::value), true);

    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<H,H,H>::type, H>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<H,A,H>::type, H>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<A,H,H>::type, H>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<D,D,D>::type, D>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<D,A,D>::type, D>::value), true);
    ASSERT_EQUAL(((bool) ::cuda::std::is_same<cusp::minimum_space<A,D,D>::type, D>::value), true);
    #else
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<H,H>::type, H>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<H,A>::type, H>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<A,H>::type, H>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<D,D>::type, D>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<D,A>::type, D>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<A,D>::type, D>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<A,A>::type, A>::value), true);

    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<H,H,A>::type, H>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<H,A,A>::type, H>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<A,H,A>::type, H>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<D,D,A>::type, D>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<D,A,A>::type, D>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<A,D,A>::type, D>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<A,A,A>::type, A>::value), true);

    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<H,H,H>::type, H>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<H,A,H>::type, H>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<A,H,H>::type, H>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<D,D,D>::type, D>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<D,A,D>::type, D>::value), true);
    ASSERT_EQUAL(((bool) thrust::detail::is_same<cusp::minimum_space<A,D,D>::type, D>::value), true);
    #endif
}
DECLARE_UNITTEST(TestMinimumSpace);

