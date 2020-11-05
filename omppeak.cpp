#undef MAD_4
#undef MAD_16
#undef MAD_64

#define MAD_4(x, y)     x = y*x+y;   y = x*y+x;   x = y*x+y;   y = x*y+x;
#define MAD_16(x, y)    MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);
#define MAD_64(x, y)    MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);

//Naive port of some portion of clpeak (https://github.com/krrishnarraj/clpeak/)
#include <omp.h>
#include <vector>
#include <iostream>

int main(){
    //Trick to not measure the JIT time 
    #pragma omp target
    {}
    // Random large number, lack HE introspection capabilites in OpenMP
    // should be equal to (devInfo.numCUs) * (devInfo.computeDPWgsPerCU) * (devInfo.maxWGSize);
    const int globalWIs{ 1720320 }; //{ 960*8*64 }; //32768*32}; //*32 };
    std::vector<double> V(globalWIs);
    double *ptr { V.data() };

    //Trick to not measure data-transfer time
    #pragma omp target enter data map(alloc: ptr[0:globalWIs])

    for (int i=0 ; i<100; i++) { 
        //Trick to not measure data-transfer time 
        double s = omp_get_wtime();
        #pragma omp target map(from: ptr[0:globalWIs])
        #pragma omp teams distribute parallel for simd
        for (int i=0 ; i<globalWIs; i++) {
            double x = 1.3f;
            double y = (double) i;
	        for(int j=0; j<128; j++) {
		        MAD_16(x,y);
	        }
	        ptr[i] = y;
        }
        const double timed = omp_get_wtime() - s;

        const int workPerWI {128*16*2}; // Indicates flops executed per work-item
        const double gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e9f;
        std::cout << "glflops: " << gflops << std::endl;
    }
   #pragma omp target exit data map(from: ptr[0:globalWIs])

}
