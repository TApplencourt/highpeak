#undef MAD_4
#undef MAD_16
#undef MAD_64

#define MAD_4(x, y)     x = y*x+y;   y = x*y+x;   x = y*x+y;   y = x*y+x;
#define MAD_16(x, y)    MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);
#define MAD_64(x, y)    MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);


#include <CL/sycl.hpp>
#include <vector>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  const auto global_range = 1720320;

  //  _       _   _
  // |_)    _|_ _|_ _  ._
  // |_) |_| |   | (/_ |
  //

  // Crrate array
  std::vector<float> A(global_range);

    sycl::buffer<sycl::cl_float, 1> bufferA(A.data(), A.size());

    sycl::queue myQueue;
    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";


    for (int i = 0 ; i < 100; i++) {
    const auto s = std::chrono::high_resolution_clock::now();
    // Create a command_group to issue command to the group
    myQueue.submit([&](sycl::handler &cgh) {
      // Create an accesor for the sycl buffer. Trust me, use auto.
      auto accessorA = bufferA.get_access<sycl::access::mode::discard_write>(cgh);
      // Submit the kernel
      cgh.parallel_for<class hello_world>(
          //sycl::range<1>(global_range),
          sycl::nd_range<1>(global_range,128), 
          [=](sycl::nd_item<1> idx) {

        float y = idx.get_local_id(0);
        float x = 1.3f;
        for(int j=0; j<128; j++) {
            MAD_64(x,y);
        }
       accessorA[idx.get_global_id(0)] = y;
          }); // End of the kernel function
    });       // End of the queue commands
    myQueue.wait();
    const auto f =  std::chrono::high_resolution_clock::now();
    const double timed = std::chrono::duration_cast<std::chrono::nanoseconds>(f-s).count();
    
    const int workPerWI {128*64*2}; // Indicates flops executed per work-item
    const double gflops = (static_cast<float>(global_range) * static_cast<float>(workPerWI)) / timed ;
    std::cout << "glflops: " << gflops << std::endl;
    }
  return 0;
}
