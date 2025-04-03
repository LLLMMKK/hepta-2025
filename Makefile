CFLAG = -O3 -g -lcublas
all:
	nvcc driver.cc winograd.cu -std=c++11 ${CFLAG} -o winograd
clean:
	rm -f winograd