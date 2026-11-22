CUDA ?= /usr/local/cuda
SANITIZER ?= $(CUDA)/compute-sanitizer
ARCH ?= 121

all: libledger_access.so patches.fatbin

libledger_access.so: tracer.cpp access.h
	$(CXX) -O2 -std=c++17 -shared -fPIC -o $@ $< -I$(CUDA)/include -I$(SANITIZER)/include \
		-L$(SANITIZER) -Wl,-rpath,$(SANITIZER) -lsanitizer-public -lcuda

patches.fatbin: patches.cu access.h
	$(CUDA)/bin/nvcc --fatbin --compile-as-tools-patch -O3 -I$(SANITIZER)/include \
		-gencode arch=compute_$(ARCH),code=sm_$(ARCH) -o $@ -c $<

clean:
	rm -f libledger_access.so patches.fatbin
