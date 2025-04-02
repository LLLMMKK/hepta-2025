CFLAG = -O3 -g -Wall -fopenmp -mavx -mavx2 -mfma -march=native -lcublas -lcudart -I/opt/cuda/targets/x86_64-linux/include
all:
	g++ driver.cc winograd.cc -std=c++11 ${CFLAG} -o winograd

clean:
	rm -f winograd