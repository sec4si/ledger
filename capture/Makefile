CUDA ?= /usr/local/cuda
CUPTI_INC ?= $(firstword $(wildcard $(CUDA)/extras/CUPTI/include $(CUDA)/targets/*/include))
CUPTI_LIB ?= $(firstword $(wildcard $(CUDA)/extras/CUPTI/lib64 $(CUDA)/targets/*/lib))

libledger.so: ledger_cupti.cc
	$(CXX) -O2 -std=c++17 -shared -fPIC -o $@ $< -I$(CUDA)/include -I$(CUPTI_INC) \
		-L$(CUPTI_LIB) -L$(CUDA)/lib64 -Wl,-rpath,$(CUPTI_LIB) -lcupti -lcuda -ldl

clean:
	rm -f libledger.so
