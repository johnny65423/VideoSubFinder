# Parity tests: the GPU implementation of the "transform" step must give exactly the same bytes as the CPU code.
# Enabled with -DBUILD_GPU_PARITY_TESTS=ON.  Usage: gpu_parity [--video <path to a test video>]
add_executable(gpu_parity gpu_parity_main.cpp t1_kernels.cpp t4_pipeline.cpp)

target_compile_features(gpu_parity PUBLIC cxx_std_17)

target_include_directories(gpu_parity PRIVATE
	"../../Components/Include"
	"../../Components/IPAlgorithms"
	"../../Components/CUDAKernels"
	 ${wxWidgets_INCLUDE_DIRS}
	 ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(gpu_parity PRIVATE ${VideoSubFinderWXW_LINK_DIRS})

target_link_libraries(gpu_parity PRIVATE
	IPAlgorithms
	${CUDAKernels_LIB}
	${OpenCV_LIBS}
	${wxWidgets_LIBRARIES}
)

add_test(NAME gpu_parity COMMAND gpu_parity)
