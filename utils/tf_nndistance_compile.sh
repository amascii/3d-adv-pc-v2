CUDA="10.0"
PYTHON="3.6"
/usr/local/cuda-$CUDA/bin/nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I /usr/local/lib/python$PYTHON/dist-packages/tensorflow/include -I /usr/local/cuda-$CUDA/include -I /usr/local/lib/python$PYTHON/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-$CUDA/lib64/ -L/usr/local/lib/python$PYTHON/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
