#CUDA="10.0"
#PYTHON="3.6"
#/usr/local/cuda-$CUDA/bin/
#nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I /usr/local/lib/python$PYTHON/dist-packages/tensorflow/include -I /usr/local/cuda-$CUDA/include -I /usr/local/lib/python$PYTHON/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-$CUDA/lib64/ -L/usr/local/lib/python$PYTHON/dist-packages/tensorflow -llibtensorflow_framework.so.2 -O2 -D_GLIBCXX_USE_CXX11_ABI=0


TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu \
${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o tf_nndistance_so.so tf_nndistance.cpp \
  tf_nndistance_g.cu.o ${TF_CFLAGS[@]} -fPIC ${TF_LFLAGS[@]}
#g++ -std=c++11 -shared tf_nndistance.cpp -o tf_nndistance_so.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
