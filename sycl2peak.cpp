#undef MAD_4
#undef MAD_16
#undef MAD_64

//https://stackoverflow.com/questions/20631922/expand-macro-inside-string-literal
// Not smart enought to understand it
#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)

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
  std::vector<double> A(global_range);

    sycl::buffer<sycl::cl_double, 1> bufferA(A.data(), A.size());

    sycl::queue myQueue;
    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    sycl::program p(myQueue.get_context());
    p.build_with_source(R"EOL(__kernel void hello_world(__global double *ptr) { 
        double y = (double) get_local_id(0);
        double x = 1.3f;
        for(int j=0; j<128; j++) {
            )EOL" STRINGIFY(MAD_16(x,y)) R"EOL(;
        }
        ptr[get_local_id(0)] = y; }
     )EOL");

    for (int i = 0 ; i < 100; i++) {
    const auto s = std::chrono::high_resolution_clock::now();
    // Create a command_group to issue command to the group
    myQueue.submit([&](sycl::handler &cgh) {
      // Create an accesor for the sycl buffer. Trust me, use auto.
      auto accessorA = bufferA.get_access<sycl::access::mode::discard_write>(cgh);
      // Submit the kernel
      cgh.set_arg(0,accessorA);
      cgh.parallel_for(sycl::nd_range<1>(global_range,64), p.get_kernel("hello_world"));
    });       // End of the queue commands
    myQueue.wait();
    const auto f =  std::chrono::high_resolution_clock::now();
    const double timed = std::chrono::duration_cast<std::chrono::nanoseconds>(f-s).count();
    
    const int workPerWI {128*16*2}; // Indicates flops executed per work-item
    const double gflops = (static_cast<float>(global_range) * static_cast<float>(workPerWI)) / timed ;
    std::cout << "glflops: " << gflops << std::endl;
    }
  return 0;
}
