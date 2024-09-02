cmake .. -DCUTLASS_NVCC_ARCHS=90a \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCUTLASS_UNITY_BUILD_ENABLED=ON \
    -DCUTLASS_LIBRARY_OPERATIONS=gemm \
    -DCUTLASS_LIBRARY_KERNELS="cutlass3x*bf16*" \
    -DCUTLASS_ENABLE_CUBLAS=ON

EA: Recall the profiler command: 

./tools/profiler/cutlass_profiler 
     --operation=gemm 
     --n=384 --m=106496 --k=16384 
     --A=bf16:row --B=bf16:column --C=bf16:column 
     --output=cutlass_profile_ffn1_384_transposed.csv

The files in tools / library / src / reference are:

conv2d.cu
conv3d.cu
conv_reference_operation.h
gemm_e4m3a_e4m3out.cu
gemm_e4m3a_e5m2out.cu
gemm_e5m2a_e4m3out.cu
gemm_e5m2a_e5m2out.cu
gemm_fp32out.cu
gemm_fp8in_bf16out.cu
gemm_fp8in_fp16out.cu
gemm_fp8in_fp32out.cu
gemm_fp_mixed_input.cu
gemm_fp_other.cu
gemm_int4.cu
gemm_int8_canonical.cu
gemm_int8_interleaved_32.cu
gemm_int8_interleaved_64.cu
gemm_reference_operation.h
initialize_reference_operations.cu

The only one with bf16 in the title is `gemm_fp8in_bf16out.cu`. `grep -rn bf16
tools` produces:

tools/library/CMakeLists.txt:232:  src/reference/gemm_fp8in_bf16out.cu
tools/library/src/util.cu:451:  {"bf16", "BF16", NumericTypeID::kBF16},
tools/library/src/util.cu:456:  {"cbf16", "CBF16", NumericTypeID::kCBF16},
tools/library/src/reduction/init_reduction_operations.cu:48:void initialize_reduce_add_linear_combination_f32_f32_bf16(Manifest &manifest);
tools/library/src/reduction/init_reduction_operations.cu:60:  initialize_reduce_add_linear_combination_f32_f32_bf16(manifest);
tools/library/src/reduction/reduction_device.cu:115:void initialize_reduce_add_linear_combination_f32_f32_bf16(Manifest &manifest) {
tools/library/src/reduction/reduction_device.cu:135:  using Operation_reduce_add_linear_combination_f32_f32_bf16 = cutlass::reduction::device::ReduceSplitK<
tools/library/src/reduction/reduction_device.cu:144:    Operation_reduce_add_linear_combination_f32_f32_bf16>(
tools/library/src/reduction/reduction_device.cu:145:      "reduce_add_linear_combination_f32_f32_bf16"
tools/library/src/reference/gemm_fp8in_bf16out.cu:49:void initialize_gemm_reference_operations_fp8in_bf16out(Manifest &manifest) {
tools/library/src/reference/initialize_reference_operations.cu:55:void initialize_gemm_reference_operations_fp8in_bf16out(Manifest &manifest);
tools/library/src/reference/initialize_reference_operations.cu:81:  initialize_gemm_reference_operations_fp8in_bf16out(manifest);

grep -nw def python/cutlass_library/generator.py

49:def logging_prefix(indent_level: int = 0) -> str:
55:def log_debug_line(line: str, indent_level: int = 0) -> None:
70:def _add_package_disablement_flag(argparser):
99:def CudaToolkitVersionSatisfies(semantic_ver_string, major, minor, patch = 0):
118:def EpilogueAlignment(max_alignment, tile, epilogue_steps = 8):
121:  def product(X, identity = 1):
130:def DefaultSwizzlingFunctor():
135:def CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, \
174:def CreateGemmUniversal3xOperator(
223:def CreateSparseGemmOperator(manifest, layouts, tile_descriptions, data_type, \
261:def CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, data_type, \
294:def CreateGemmGroupedOperator(manifest, layouts, tile_descriptions, data_type, \
330:def CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, data_type, \
377:def CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, data_type, \
417:def CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, data_type, \
476:def CreateConv2dOperator(manifest, layout, tile_descriptions, data_type, alignment_constraints, \
585:def CreateConv2dFixedChannelsOperator(manifest, layout, tile_descriptions, data_type, channel_counts, \
632:def CreateConv2dFewChannelsOperator(manifest, layout, tile_descriptions, data_type, channel_counts, \
677:def CreateConv3dOperator(manifest, layout, tile_descriptions, data_type, alignment, \
752:def CreateDepthwiseConv2dOperator(manifest, layout, tile_descriptions, data_type, alignment_constraints, \
818:  def __init__(self,
863:  def __str__(self):
866:  def is_complex(self):
874:  def is_mixed_input(self):
877:  def accumulator_type(self):
883:  def short_math_name(self):
889:  def is_tensor_op(self):
896:  def instruction_shape_string(self):
909:  def intermediate_type_string(self):
926:  def core_name(self):
932:  def extended_name(self):
941:  def is_complex(self):
949:  def layout_names(self):
958:  def extended_name(self):
968:  def configuration_name(self):
985:  def procedural_name(self):
988:def convolution_tensor_layout_type_to_operation_kind(layout: LayoutType) -> OperationKind:
996:def CreateConvOperator3x(manifest: Manifest,
1075:  def input_and_output_layouts(spatial_dim: int, kind: ConvKind) -> Tuple[LayoutType, LayoutType]:
1098:  def dims_to_layouts(A_B_C: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> \
1121:  def make_combinations():
1169:def GenerateSM50_Simt(manifest, cuda_version):
1221:def GenerateSM50_Simt_complex(manifest, cuda_version):
1268:def GenerateSM50(manifest, cuda_version):
1276:def GenerateSM60_Simt(manifest, cuda_version):
1319:def GenerateSM60_Simt_DepthwiseConv2d(manifest, cuda_version):
1393:def GenerateSM60(manifest, cuda_version):
1401:def GenerateSM61_Simt(manifest, cuda_version):
1453:def GenerateSM61(manifest, cuda_version):
1460:def GenerateSM70_TensorOp_884(manifest, cuda_version):
1531:def GenerateSM70_PlanarComplexTensorOp_884(manifest, cuda_version):
1598:def GenerateSM70_WmmaTensorOp_161616(manifest, cuda_version):
1660:def GenerateSM70(manifest, cuda_version):
1672:def GenerateSM75_TensorOp_1688_FewChannels(manifest, cuda_version, math_inst):
1713:def GenerateSM75_TensorOp_1688(manifest, cuda_version):
1791:def GenerateSM75_PlanarComplexTensorOp_1688(manifest, cuda_version):
1859:def GenerateSM75_TensorOp_8816_TN(manifest, cuda_version):
1961:def GenerateSM75_TensorOp_8816_Interleaved(manifest, cuda_version):
2020:def GenerateSM75_TensorOp_8832_TN(manifest, cuda_version):
2102:def GenerateSM75_TensorOp_8832_Interleaved(manifest, cuda_version):
2162:def GenerateSM75_TensorOp_88128(manifest, cuda_version):
2207:def GenerateSM75_WmmaTensorOp_161616(manifest, cuda_version):
2265:def GenerateSM75_Simt_complex(manifest, cuda_version):
2301:def GenerateSM75(manifest, cuda_version):
2317:def GenerateSM80_TensorOp_16816(manifest, cuda_version):
2414:def GenerateSM80_SparseTensorOp_16832(manifest, cuda_version):
2492:def GenerateSM80_PlanarComplexTensorOp_16816(manifest, cuda_version):
2565:def GenerateSM80_TensorOp_16816_mixed_input_upcast_a(manifest, cuda_version):
2663:def GenerateSM80_TensorOp_16816_mixed_input_upcast_b(manifest, cuda_version):
2767:def GenerateSM80_TensorOp_16832_TN(manifest, cuda_version):
2860:def GenerateSM80_SparseTensorOp_16864_TN(manifest, cuda_version):
2915:def GenerateSM80_TensorOp_16832_Interleaved(manifest, cuda_version):
2969:def GenerateSM80_TensorOp_16864_TN(manifest, cuda_version):
3044:def GenerateSM80_SparseTensorOp_168128_TN(manifest, cuda_version):
3098:def GenerateSM80_TensorOp_16864_Interleaved(manifest, cuda_version):
3151:def GenerateSM80_TensorOp_168256(manifest, cuda_version):
3209:def GenerateSM80_TensorOp_1688(manifest, cuda_version):
3284:def GenerateSM80_TensorOp_1688_fast_math(manifest, cuda_version):
3352:def GenerateSM80_TensorOp_1688_fast_fp32_math(manifest, cuda_version):
3403:def GenerateSM80_TensorOp_1688_fast_fp32_math_complex(manifest, cuda_version):
3451:def GenerateSM80_SparseTensorOp_16816_fast_math(manifest, cuda_version):
3500:def GenerateSM80_TensorOp_1688_complex(manifest, cuda_version):
3549:def GenerateSM80_TensorOp_1688_rank_k(manifest, cuda_version):
3608:def GenerateSM80_TensorOp_1688_rank_k_complex(manifest, cuda_version):
3663:def GenerateSM80_TensorOp_1688_trmm(manifest, cuda_version):
3730:def GenerateSM80_TensorOp_1688_trmm_complex(manifest, cuda_version):
3792:def GenerateSM80_TensorOp_1688_symm(manifest, cuda_version):
3857:def GenerateSM80_TensorOp_1688_symm_complex(manifest, cuda_version):
3915:def GenerateSM80_TensorOp_884(manifest, cuda_version):
3962:def GenerateSM80_TensorOp_884_complex(manifest, cuda_version):
4018:def GenerateSM80_TensorOp_884_complex_gaussian(manifest, cuda_version):
4065:def GenerateSM80_TensorOp_884_rank_k(manifest, cuda_version):
4110:def GenerateSM80_TensorOp_884_rank_k_complex(manifest, cuda_version):
4160:def GenerateSM80_TensorOp_884_rank_k_complex_gaussian(manifest, cuda_version):
4209:def GenerateSM80_TensorOp_884_trmm(manifest, cuda_version):
4257:def GenerateSM80_TensorOp_884_trmm_complex(manifest, cuda_version):
4311:def GenerateSM80_TensorOp_884_trmm_complex_gaussian(manifest, cuda_version):
4362:def GenerateSM80_TensorOp_884_symm(manifest, cuda_version):
4410:def GenerateSM80_TensorOp_884_symm_complex(manifest, cuda_version):
4462:def GenerateSM80_TensorOp_884_symm_complex_gaussian(manifest, cuda_version):
4516:def GenerateSM80_Simt_f32(manifest, cuda_version):
4568:def GenerateSM80_Simt_f64(manifest, cuda_version):
4613:def GenerateSM80_Simt_complex(manifest, cuda_version):
4670:def GenerateSM80(manifest, cuda_version):
4714:def GenerateSM89_TensorOp_16832_fp8(manifest, cuda_version):
4845:def GenerateSM89_SparseTensorOp_16864_fp8(manifest, cuda_version):
4943:def GenerateSM89(manifest, cuda_version):
4950:def GenerateSM90_TensorOp_16b_WGMMA_gemm(manifest, cuda_version):
5152:def GenerateSM90_TensorOp_16b_WGMMA_alignx_gemm(manifest, cuda_version):
5300:def GenerateSM90_TensorOp_tf32_WGMMA_gemm(manifest, cuda_version):
5424:def GenerateSM90_TensorOp_tf32_WGMMA_alignx_gemm(manifest, cuda_version):
5509:def GenerateSM90_TensorOp_int8_WGMMA_gemm(manifest, cuda_version):
5614:def GenerateSM90_TensorOp_int8_WGMMA_alignx_gemm(manifest, cuda_version):
5688:def GenerateSM90_TensorOp_fp8_WGMMA_gemm(manifest, cuda_version):
5943:def GenerateSM90_TensorOp_fp8_WGMMA_alignx_gemm(manifest, cuda_version):
6126:def GenerateSM90_TensorOp_1684(manifest, cuda_version):
6173:def GenerateSM90_TensorOp_1684_complex(manifest, cuda_version):
6230:def GenerateSM90_TensorOp_1684_complex_gaussian(manifest, cuda_version):
6277:def GenerateSM90_TensorOp_1684_rank_k(manifest, cuda_version):
6322:def GenerateSM90_TensorOp_1684_rank_k_complex(manifest, cuda_version):
6372:def GenerateSM90_TensorOp_1684_rank_k_complex_gaussian(manifest, cuda_version):
6421:def GenerateSM90_TensorOp_1684_trmm(manifest, cuda_version):
6469:def GenerateSM90_TensorOp_1684_trmm_complex(manifest, cuda_version):
6523:def GenerateSM90_TensorOp_1684_trmm_complex_gaussian(manifest, cuda_version):
6574:def GenerateSM90_TensorOp_1684_symm(manifest, cuda_version):
6622:def GenerateSM90_TensorOp_1684_symm_complex(manifest, cuda_version):
6674:def GenerateSM90_TensorOp_1684_symm_complex_gaussian(manifest, cuda_version):
6727:def GenerateSM90_Conv3x(manifest, cuda_version,
7056:  def make_math_instruction(data_types: Dict[str, DataType],
7090:def GenerateSM90(manifest, cuda_version):
7115:def numeric_log_level(log_level: str) -> int:
7134:def define_parser():

# Basics

CMake Version: 3.30.2
-- CUTLASS 3.5.1
-- CUDART: /usr/local/cuda/lib64/libcudart.so
-- CUDA Driver: /usr/local/cuda/lib64/stubs/libcuda.so
-- NVRTC: /usr/local/cuda/lib64/libnvrtc.so

-- Make cute::tuple be the new standard-layout tuple type

-- Enable caching of reference results in conv unit tests
-- Enable rigorous conv problem sizes in conv unit tests

-- Using NVCC flags: --expt-relaxed-constexpr;
                     -DCUTE_USE_PACKED_TUPLE=1;
                     -DCUTLASS_TEST_LEVEL=0;
                     -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1;
                     -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1;
                     -DCUTLASS_DEBUG_TRACE_LEVEL=0;
                     -Xcompiler=-Wconversion;
                     -Xcompiler=-fno-strict-aliasing

-- Configuring cublas ...
-- cuBLAS: /usr/local/cuda/lib64/libcublas.so
-- cuBLAS: /usr/local/cuda/include
-- Configuring cuBLAS ... done.

sm90
bf16 
s 64 x 128 x 16

# Generating

-- Generating /scratch/ericauld/build/tools/library/cutlass_library_objs.unity.a2ea0bb712b9.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_objs.unity.0b0987e2b09f.cu
-- Completed generation of library instances. See /scratch/ericauld/build/tools/library/library_instance_generation.log for more information.
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_objs.unity.27c27562c34d.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.03ee82b94fe3.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.d7352d15db3c.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.3728ec116d12.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.2bea91a60e90.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.d256f90bdb83.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.22ad4af05b20.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.6897d0e98182.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.1bb6b8a96b79.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.36024735e9c3.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.7c9f44d59c10.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.44b9772d17cb.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.531cdc96d4f4.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.7d371d8679a6.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.831452285776.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.6f2fa492092b.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.3ad9e49438a2.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.63c450706c93.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.d73e42e1ee11.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.96fd6a0dfa14.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.5b0452ec69a1.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.f2921a657dc7.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.2f31f29b3d94.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.6040e9813631.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.fde475b535bb.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.71379530af42.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.eb11bd170e44.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.70b039a5eb0d.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.e5b6ad3e2a29.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.fb6aa9c9d4e6.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.a5e65e4dbf73.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.efa9b5ee56eb.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.c763be6477f3.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.e007cb8ca3f9.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.314cb66d7847.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.252794b1e600.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.0d2908f9c4ac.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.d9dc65651a6f.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.57735bd44f04.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.5cb6e6d3234b.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.2629cb2ff7ea.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.00c75d010f40.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.e6b76171a5cb.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.117e0d4b7cd6.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.51ec5fd72b69.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.d336023d0ad7.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.3cdd6555ef9d.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.06f7e67a5271.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.955dcec15ed9.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.a3bcaf4d2d44.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.0849e4926bb7.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.ee818a08b1c3.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.947dd8d5ad07.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.7b3b783ce367.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.4a76251ebe08.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.e46f883cefe6.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.b1773487f6cd.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.df8d1ca52016.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.060eb2bec7e7.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.f3b8593a8536.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.2ad53c0bc4f9.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.27305da7398c.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.639382efcfb3.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.64fa966451ed.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.01bda7bd6636.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.6e72da242e73.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.8f2c5efb63e0.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.995c6e3cdd57.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.bf98820590eb.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.448edc543eeb.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.d0e55ac95c39.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.1afe5d921db8.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.00f61105d236.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.dda846077b49.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.2e01d81764f3.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.b73f0461b80a.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.2b6e5849ea1d.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.78f277725e40.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.9b0c9dbe33df.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.3d65c13dfc1a.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.5edc13ef8bc9.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.2daac7d1ed30.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.2be607683fee.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.fefc6b9c927b.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.bb30e67c5b50.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.1ba7334dbe7a.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.ccf59fec5f63.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.fdbde623c470.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.02dd0fe59ff8.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.c6ae6a84e4b7.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.236723a9e2e0.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.8edd96d8907c.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.018c4f39fe19.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.edd08a7670aa.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.222abce35b7d.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.fc1ec84ec84c.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.c3006c2d923a.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.c9ef13665e00.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.667496160d08.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.45a2f28d1b3e.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.05b7fef86415.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.fbcd730a7605.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.3a1678f5b1b2.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.7bce3b96ce46.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.864ec3638ef2.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.859034de5589.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.7efd946048c0.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.17aad01a8fe6.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.d482b118e4ea.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.4e469bdda497.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.a21ee6a4f9fb.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.f384159556dc.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.86568175c2f0.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.efdf70856d28.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.39a4b47e19e6.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.b6d2052604dc.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.98dbad860b8a.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.unity.7048c788484b.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.unity.6933637a60f6.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.unity.9780b840e266.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.unity.00d616b139a8.cu
-- Generating /scratch/ericauld/build/tools/library/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.unity.fe44c13e7222.cu
-- Generating /scratch/ericauld/build/tools/profiler/cutlass_profiler.unity.74876761f2a6.cu
-- Configuring done (6.1s)
-- Generating done (0.5s)
-- Build files have been written to: /scratch/ericauld/build

[  0%] Building CXX object tools/library/CMakeFiles/cutlass_library_objs.dir/src/manifest.cpp.o
[  2%] Building CUDA object tools/library/CMakeFiles/cutlass_library_objs.dir/cutlass_library_objs.unity.a2ea0bb712b9.cu.o
[  2%] Building CUDA object tools/library/CMakeFiles/cutlass_library_objs.dir/cutlass_library_objs.unity.0b0987e2b09f.cu.o
[  2%] Building CXX object tools/library/CMakeFiles/cutlass_library_objs.dir/generated/initialize_all.cpp.o
[  2%] Building CUDA object tools/library/CMakeFiles/cutlass_library_objs.dir/cutlass_library_objs.unity.27c27562c34d.cu.o
[  2%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.03ee82b94fe3.cu.o
[  2%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.d7352d15db3c.cu.o
[  5%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.3728ec116d12.cu.o
[  7%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.unity.6933637a60f6.cu.o
[  7%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.unity.7048c788484b.cu.o
[  7%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.2bea91a60e90.cu.o
[  7%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.unity.9780b840e266.cu.o
[  7%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.d256f90bdb83.cu.o
[  7%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.7d371d8679a6.cu.o
[ 10%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.22ad4af05b20.cu.o
[ 10%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.6f2fa492092b.cu.o
[ 10%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.unity.fe44c13e7222.cu.o
[ 12%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.831452285776.cu.o
[ 12%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.3ad9e49438a2.cu.o
[ 12%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.6897d0e98182.cu.o
[ 15%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs.unity.00d616b139a8.cu.o
[ 15%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.7b3b783ce367.cu.o
[ 15%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.4a76251ebe08.cu.o
[ 15%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.d73e42e1ee11.cu.o
[ 17%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.36024735e9c3.cu.o
[ 17%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.e46f883cefe6.cu.o
[ 17%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.63c450706c93.cu.o
[ 17%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.bb30e67c5b50.cu.o
[ 17%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.1bb6b8a96b79.cu.o
[ 20%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.6040e9813631.cu.o
[ 20%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.7c9f44d59c10.cu.o
[ 25%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.b1773487f6cd.cu.o
[ 25%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.96fd6a0dfa14.cu.o
[ 25%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.5b0452ec69a1.cu.o
[ 25%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.bf98820590eb.cu.o
[ 25%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.e007cb8ca3f9.cu.o
[ 25%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.060eb2bec7e7.cu.o
[ 25%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.d0e55ac95c39.cu.o
[ 25%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.71379530af42.cu.o
[ 28%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.df8d1ca52016.cu.o
[ 28%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.fde475b535bb.cu.o
[ 28%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.448edc543eeb.cu.o
[ 28%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs.unity.f3b8593a8536.cu.o
[ 28%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.531cdc96d4f4.cu.o
[ 30%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.1ba7334dbe7a.cu.o
[ 30%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.1afe5d921db8.cu.o
[ 30%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.314cb66d7847.cu.o
[ 30%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs.unity.44b9772d17cb.cu.o
[ 30%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.fdbde623c470.cu.o
[ 30%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.f2921a657dc7.cu.o
[ 33%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.00f61105d236.cu.o
[ 33%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.ccf59fec5f63.cu.o
[ 33%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.2b6e5849ea1d.cu.o
[ 35%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.2ad53c0bc4f9.cu.o
[ 35%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.eb11bd170e44.cu.o
[ 35%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.78f277725e40.cu.o
[ 38%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs.unity.2f31f29b3d94.cu.o
[ 41%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.252794b1e600.cu.o
[ 41%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.70b039a5eb0d.cu.o
[ 43%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.02dd0fe59ff8.cu.o
[ 46%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.3a1678f5b1b2.cu.o
[ 48%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.117e0d4b7cd6.cu.o
[ 48%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.4e469bdda497.cu.o
[ 48%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.018c4f39fe19.cu.o
[ 48%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.fb6aa9c9d4e6.cu.o
[ 48%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.e5b6ad3e2a29.cu.o
[ 51%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.a5e65e4dbf73.cu.o
[ 51%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.efa9b5ee56eb.cu.o
[ 51%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs.unity.c763be6477f3.cu.o
[ 51%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.0d2908f9c4ac.cu.o
[ 51%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.7bce3b96ce46.cu.o
[ 51%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.c6ae6a84e4b7.cu.o
[ 51%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.51ec5fd72b69.cu.o
[ 51%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.236723a9e2e0.cu.o
[ 53%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs.unity.8edd96d8907c.cu.o
[ 53%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.edd08a7670aa.cu.o
[ 56%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.222abce35b7d.cu.o
[ 56%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.fc1ec84ec84c.cu.o
[ 56%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.d336023d0ad7.cu.o
[ 56%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.c3006c2d923a.cu.o
[ 58%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.3cdd6555ef9d.cu.o
[ 58%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.06f7e67a5271.cu.o
[ 58%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.955dcec15ed9.cu.o
[ 61%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.a3bcaf4d2d44.cu.o
[ 64%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.c9ef13665e00.cu.o
[ 64%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.0849e4926bb7.cu.o
[ 64%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.667496160d08.cu.o
[ 64%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.45a2f28d1b3e.cu.o
[ 66%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.a21ee6a4f9fb.cu.o
[ 66%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.f384159556dc.cu.o
[ 66%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.864ec3638ef2.cu.o
[ 69%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.859034de5589.cu.o
[ 69%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.7efd946048c0.cu.o
[ 69%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.17aad01a8fe6.cu.o
[ 69%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs.unity.d482b118e4ea.cu.o
[ 69%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.d9dc65651a6f.cu.o
[ 71%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.9b0c9dbe33df.cu.o
[ 71%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.27305da7398c.cu.o
[ 74%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.639382efcfb3.cu.o
[ 74%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.dda846077b49.cu.o
[ 74%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.2e01d81764f3.cu.o
[ 74%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.86568175c2f0.cu.o
[ 76%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.efdf70856d28.cu.o
[ 79%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs.unity.b73f0461b80a.cu.o
[ 79%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.05b7fef86415.cu.o
[ 79%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.ee818a08b1c3.cu.o
[ 82%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs.unity.fbcd730a7605.cu.o
[ 82%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.64fa966451ed.cu.o
[ 82%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.01bda7bd6636.cu.o
[ 84%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.6e72da242e73.cu.o
[ 84%] Built target cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_objs
[ 84%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.39a4b47e19e6.cu.o
[ 84%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.3d65c13dfc1a.cu.o
[ 84%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.5edc13ef8bc9.cu.o
[ 84%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.b6d2052604dc.cu.o
[ 84%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.dir/cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs.unity.98dbad860b8a.cu.o
[ 84%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.8f2c5efb63e0.cu.o
[ 84%] Built target cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2_objs
[ 84%] Linking CUDA shared library libcutlass_gemm_sm90_bf16_s64x128x32gemm_e4m3.so
[ 87%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.2daac7d1ed30.cu.o
[ 87%] Built target cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3
[ 87%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.2be607683fee.cu.o
[ 87%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs.unity.995c6e3cdd57.cu.o
[ 87%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs.unity.fefc6b9c927b.cu.o
[ 87%] Linking CUDA shared library libcutlass_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2.so
[ 87%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.57735bd44f04.cu.o
[ 87%] Built target cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16_objs
[ 87%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs.unity.947dd8d5ad07.cu.o
[ 87%] Built target cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e4m3_e5m2
[ 87%] Linking CUDA shared library libcutlass_gemm_sm90_bf16_s64x128x16gemm_bf16.so
[ 87%] Built target cutlass_library_gemm_sm90_bf16_s64x128x16gemm_bf16
[ 89%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.5cb6e6d3234b.cu.o
[ 89%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.2629cb2ff7ea.cu.o
[ 89%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.00c75d010f40.cu.o
[ 92%] Building CUDA object tools/library/CMakeFiles/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.dir/cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs.unity.e6b76171a5cb.cu.o
[ 92%] Built target cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16_objs
[ 92%] Linking CUDA shared library libcutlass_gemm_sm90_void_s64x256x16gemm_bf16.so
[ 92%] Built target cutlass_library_gemm_sm90_void_s64x256x16gemm_bf16
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16_objs
[ 92%] Linking CUDA shared library libcutlass_gemm_sm90_bf16_s64x256x16gemm_bf16.so
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x256x16gemm_bf16
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3_objs
[ 92%] Linking CUDA shared library libcutlass_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3.so
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_e4m3
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3_objs
[ 92%] Linking CUDA shared library libcutlass_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3.so
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_e4m3
[ 92%] Built target cutlass_library_gemm_sm90_s64x128x16gemm_bf16_objs
[ 92%] Linking CUDA shared library libcutlass_gemm_sm90_s64x128x16gemm_bf16.so
[ 92%] Built target cutlass_library_gemm_sm90_s64x128x16gemm_bf16
[ 92%] Built target cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16_objs
[ 92%] Linking CUDA shared library libcutlass_gemm_sm90_void_s64x128x16gemm_bf16.so
[ 92%] Built target cutlass_library_gemm_sm90_void_s64x128x16gemm_bf16
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2_objs
[ 92%] Linking CUDA shared library libcutlass_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2.so
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_e5m2
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2_objs
[ 92%] Linking CUDA shared library libcutlass_gemm_sm90_bf16_s64x128x32gemm_e5m2.so
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x128x32gemm_e5m2
[ 92%] Built target cutlass_library_gemm_sm90_s64x256x16gemm_bf16_objs
[ 92%] Linking CUDA shared library libcutlass_gemm_sm90_s64x256x16gemm_bf16.so
[ 92%] Built target cutlass_library_gemm_sm90_s64x256x16gemm_bf16
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3_objs
[ 92%] Linking CUDA shared library libcutlass_gemm_sm90_bf16_s64x256x32gemm_e4m3.so
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e4m3
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2_objs
[ 92%] Linking CUDA shared library libcutlass_gemm_sm90_bf16_s64x256x32gemm_e5m2.so
[ 92%] Built target cutlass_library_gemm_sm90_bf16_s64x256x32gemm_e5m2
[ 92%] Built target cutlass_library_objs
[ 94%] Linking CXX shared library libcutlass.so
[ 94%] Built target cutlass_library
[ 94%] Building CXX object tools/profiler/CMakeFiles/cutlass_profiler.dir/src/main.cpp.o
[ 94%] Building CXX object tools/profiler/CMakeFiles/cutlass_profiler.dir/src/performance_report.cpp.o
[ 94%] Building CXX object tools/profiler/CMakeFiles/cutlass_profiler.dir/src/enumerated_types.cpp.o
[ 97%] Building CXX object tools/profiler/CMakeFiles/cutlass_profiler.dir/src/cudnn_helpers.cpp.o
[ 97%] Building CXX object tools/profiler/CMakeFiles/cutlass_profiler.dir/src/gpu_timer.cpp.o
[100%] Building CUDA object tools/profiler/CMakeFiles/cutlass_profiler.dir/cutlass_profiler.unity.74876761f2a6.cu.o
[100%] Building CXX object tools/profiler/CMakeFiles/cutlass_profiler.dir/src/problem_space.cpp.o
[100%] Linking CXX executable cutlass_profiler
[100%] Built target cutlass_profiler

stderr

ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x2x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x2x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_2x1x1_0_nnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_2x1x1_0_nnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_2x1x1_0_ntn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_2x1x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_2x1x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x1x1_0_tnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x1x1_0_tnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x1x1_0_ntn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x2x1_0_ttn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x1x1_0_ntn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_2x1x1_0_ntn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x2x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x2x1_0_tnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_2x1x1_0_ntn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x1x1_0_nnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x1x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x2x1_0_ntn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_nnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x1x1_0_nnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x2x1_0_ntn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_2x1x1_0_tnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x2x1_0_nnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x1x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_2x1x1_0_tnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_2x1x1_0_ntn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x2x1_0_ttn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x2x1_0_nnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x2x1_0_nnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x2x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x2x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x2x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_2x1x1_0_tnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_2x1x1_0_nnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_2x1x1_0_nnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_2x1x1_0_tnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_2x1x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_2x1x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_ntn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_256x128x64_2x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_tnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_256x128x64_1x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_ntn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_256x128x64_1x2x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_tnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_256x128x64_2x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x1x1_0_ttn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x2x1_0_ntn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x2x1_0_tnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_2x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI128cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x2x1_0_ntn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x2x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x2x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_2x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_2x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_2x1x1_0_ntn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_f32_128x256x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_f32_128x256x64_2x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_f32_128x256x64_1x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x2x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_f32_128x256x64_1x2x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_bf16_128x256x64_1x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_f32_128x256x64_2x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_bf16_128x256x64_1x2x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_2x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_bf16_128x256x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_f32_128x256x64_1x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_256x128x64_1x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_bf16_128x256x64_1x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_f32_128x256x64_1x2x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x2x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_bf16_128x256x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_f32_128x256x64_2x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_bf16_128x256x64_2x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_2x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x2x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_f32_128x256x64_1x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_bf16_128x256x64_1x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_f32_128x256x64_1x2x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_2x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_bf16_128x256x64_1x2x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_f32_128x256x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_bf16_128x256x64_2x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_f32_128x256x64_1x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_bf16_128x256x64_1x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_bf16_128x256x64_1x2x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_void_bf16_128x256x64_2x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x2x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI119cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_ttn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_2x1x1_0_ntn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_2x1x1_0_tnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_2x1x1_0_tnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x1x1_0_ntn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x1x1_0_ntn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x1x1_0_nnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x1x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x1x1_0_ntn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x1x1_0_tnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x2x1_0_ntn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x1x1_0_tnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x1x1_0_ntn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x2x1_0_ntn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x2x1_0_nnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x2x1_0_tnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_2x1x1_0_ntn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x2x1_0_ntn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x2x1_0_ttn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x2x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x2x1_0_ntn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_2x1x1_0_tnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_2x1x1_0_tnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_2x1x1_0_ntn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x2x1_0_ttn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x2x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_2x1x1_0_ttn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e4m3_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_f32_f32_256x128x64_1x1x1_0_ttn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x1x1_0_tnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI126cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x1x1_0_tnn_align8_warpspecialized_pingpong_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x2x1_0_tnn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_256x128x64_1x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_256x128x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_256x128x64_2x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_256x128x64_1x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_256x128x64_1x2x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_bf16_256x128x64_1x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_bf16_256x128x64_1x2x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_bf16_256x128x64_2x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_bf16_256x128x64_1x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_bf16_256x128x64_1x2x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_256x128x64_1x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_bf16_256x128x64_2x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_256x128x64_1x2x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI124cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_256x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_bf16_256x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_bf16_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_bf16_256x128x64_1x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_bf16_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_bf16_256x128x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_bf16_256x128x64_2x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_bf16_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI123cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI123cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI123cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x2x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI123cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_2x1x1_0_tnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI123cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI123cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x2x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI123cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_2x1x1_0_ntn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI123cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x2x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI123cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI123cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_2x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e4m3_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e5m2_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_bf16_256x128x64_1x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI125cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_bf16_256x128x64_1x2x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI123cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x1x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI123cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x2x1_0_ttn_align8_warpspecialized_pingpong_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e5m2_128x256x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI117cutlass3x_sm90_tensorop_s64x256x16gemm_bf16_bf16_f32_f32_f32_128x256x64_1x1x1_0_ttn_align8_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e4m3_128x256x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e5m2_128x256x128_1x2x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI135cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI144cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI121cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e5m2_128x256x128_1x1x1_0_tnn_align16_warpspecialized_epi_nosmemEEvNT_6ParamsE'
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the wgmma pipeline in the function '_ZN7cutlass13device_kernelI141cutlass3x_sm90_tensorop_s64x256x32gemm_e5m2_e5m2_f32_bf16_e4m3_128x256x128_1x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tmaEEvNT_6ParamsE'
/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(178): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=2, Signed=true, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
        result = Element(rnd);
                 ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "Element cutlass::reference::device::detail::RandomGaussianFunc<Element>::operator()() [with Element=cutlass::int2b_t]" at line 149 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::BlockForEach<Element,Func>(Element *, size_t, Func::Params) [with Element=cutlass::int2b_t, Func=cutlass::reference::device::detail::RandomGaussianFunc<cutlass::int2b_t>]" at line 122 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::BlockForEach<Element, Func>::BlockForEach(Element *, size_t, Func::Params, int, int, cudaStream_t) [with Element=cutlass::int2b_t, Func=cutlass::reference::device::detail::RandomGaussianFunc<cutlass::int2b_t>]" at line 394
            instantiation of "void cutlass::reference::device::BlockFillRandomGaussian(Element *, size_t, uint64_t, cutlass::RealType<Element>::Type, cutlass::RealType<Element>::Type, int, cudaStream_t) [with Element=cutlass::int2b_t]" at line 1746
            instantiation of "void cutlass::reference::device::BlockFillRandom(Element *, size_t, uint64_t, cutlass::Distribution, cudaStream_t) [with Element=cutlass::int2b_t]" at line 613 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(494): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=2, Signed=true, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
        result = Element(rnd);
                 ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "Element cutlass::reference::device::detail::RandomUniformFunc<Element>::operator()() [with Element=cutlass::int2b_t]" at line 149 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::BlockForEach<Element,Func>(Element *, size_t, Func::Params) [with Element=cutlass::int2b_t, Func=cutlass::reference::device::detail::RandomUniformFunc<cutlass::int2b_t>]" at line 122 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::BlockForEach<Element, Func>::BlockForEach(Element *, size_t, Func::Params, int, int, cudaStream_t) [with Element=cutlass::int2b_t, Func=cutlass::reference::device::detail::RandomUniformFunc<cutlass::int2b_t>]" at line 731
            instantiation of "void cutlass::reference::device::BlockFillRandomUniform(Element *, size_t, uint64_t, cutlass::RealType<Element>::Type, cutlass::RealType<Element>::Type, int, cudaStream_t) [with Element=cutlass::int2b_t]" at line 1756
            instantiation of "void cutlass::reference::device::BlockFillRandom(Element *, size_t, uint64_t, cutlass::Distribution, cudaStream_t) [with Element=cutlass::int2b_t]" at line 613 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(178): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=4, Signed=true, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
        result = Element(rnd);
                 ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "Element cutlass::reference::device::detail::RandomGaussianFunc<Element>::operator()() [with Element=cutlass::int4b_t]" at line 149 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::BlockForEach<Element,Func>(Element *, size_t, Func::Params) [with Element=cutlass::int4b_t, Func=cutlass::reference::device::detail::RandomGaussianFunc<cutlass::int4b_t>]" at line 122 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::BlockForEach<Element, Func>::BlockForEach(Element *, size_t, Func::Params, int, int, cudaStream_t) [with Element=cutlass::int4b_t, Func=cutlass::reference::device::detail::RandomGaussianFunc<cutlass::int4b_t>]" at line 394
            instantiation of "void cutlass::reference::device::BlockFillRandomGaussian(Element *, size_t, uint64_t, cutlass::RealType<Element>::Type, cutlass::RealType<Element>::Type, int, cudaStream_t) [with Element=cutlass::int4b_t]" at line 1746
            instantiation of "void cutlass::reference::device::BlockFillRandom(Element *, size_t, uint64_t, cutlass::Distribution, cudaStream_t) [with Element=cutlass::int4b_t]" at line 621 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(494): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=4, Signed=true, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
        result = Element(rnd);
                 ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "Element cutlass::reference::device::detail::RandomUniformFunc<Element>::operator()() [with Element=cutlass::int4b_t]" at line 149 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::BlockForEach<Element,Func>(Element *, size_t, Func::Params) [with Element=cutlass::int4b_t, Func=cutlass::reference::device::detail::RandomUniformFunc<cutlass::int4b_t>]" at line 122 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::BlockForEach<Element, Func>::BlockForEach(Element *, size_t, Func::Params, int, int, cudaStream_t) [with Element=cutlass::int4b_t, Func=cutlass::reference::device::detail::RandomUniformFunc<cutlass::int4b_t>]" at line 731
            instantiation of "void cutlass::reference::device::BlockFillRandomUniform(Element *, size_t, uint64_t, cutlass::RealType<Element>::Type, cutlass::RealType<Element>::Type, int, cudaStream_t) [with Element=cutlass::int4b_t]" at line 1756
            instantiation of "void cutlass::reference::device::BlockFillRandom(Element *, size_t, uint64_t, cutlass::Distribution, cudaStream_t) [with Element=cutlass::int4b_t]" at line 621 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(178): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=1, Signed=false, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
        result = Element(rnd);
                 ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "Element cutlass::reference::device::detail::RandomGaussianFunc<Element>::operator()() [with Element=cutlass::uint1b_t]" at line 149 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::BlockForEach<Element,Func>(Element *, size_t, Func::Params) [with Element=cutlass::uint1b_t, Func=cutlass::reference::device::detail::RandomGaussianFunc<cutlass::uint1b_t>]" at line 122 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::BlockForEach<Element, Func>::BlockForEach(Element *, size_t, Func::Params, int, int, cudaStream_t) [with Element=cutlass::uint1b_t, Func=cutlass::reference::device::detail::RandomGaussianFunc<cutlass::uint1b_t>]" at line 394
            instantiation of "void cutlass::reference::device::BlockFillRandomGaussian(Element *, size_t, uint64_t, cutlass::RealType<Element>::Type, cutlass::RealType<Element>::Type, int, cudaStream_t) [with Element=cutlass::uint1b_t]" at line 1746
            instantiation of "void cutlass::reference::device::BlockFillRandom(Element *, size_t, uint64_t, cutlass::Distribution, cudaStream_t) [with Element=cutlass::uint1b_t]" at line 661 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(494): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=1, Signed=false, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
        result = Element(rnd);
                 ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "Element cutlass::reference::device::detail::RandomUniformFunc<Element>::operator()() [with Element=cutlass::uint1b_t]" at line 149 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::BlockForEach<Element,Func>(Element *, size_t, Func::Params) [with Element=cutlass::uint1b_t, Func=cutlass::reference::device::detail::RandomUniformFunc<cutlass::uint1b_t>]" at line 122 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::BlockForEach<Element, Func>::BlockForEach(Element *, size_t, Func::Params, int, int, cudaStream_t) [with Element=cutlass::uint1b_t, Func=cutlass::reference::device::detail::RandomUniformFunc<cutlass::uint1b_t>]" at line 731
            instantiation of "void cutlass::reference::device::BlockFillRandomUniform(Element *, size_t, uint64_t, cutlass::RealType<Element>::Type, cutlass::RealType<Element>::Type, int, cudaStream_t) [with Element=cutlass::uint1b_t]" at line 1756
            instantiation of "void cutlass::reference::device::BlockFillRandom(Element *, size_t, uint64_t, cutlass::Distribution, cudaStream_t) [with Element=cutlass::uint1b_t]" at line 661 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(178): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=2, Signed=false, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
        result = Element(rnd);
                 ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "Element cutlass::reference::device::detail::RandomGaussianFunc<Element>::operator()() [with Element=cutlass::uint2b_t]" at line 149 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::BlockForEach<Element,Func>(Element *, size_t, Func::Params) [with Element=cutlass::uint2b_t, Func=cutlass::reference::device::detail::RandomGaussianFunc<cutlass::uint2b_t>]" at line 122 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::BlockForEach<Element, Func>::BlockForEach(Element *, size_t, Func::Params, int, int, cudaStream_t) [with Element=cutlass::uint2b_t, Func=cutlass::reference::device::detail::RandomGaussianFunc<cutlass::uint2b_t>]" at line 394
            instantiation of "void cutlass::reference::device::BlockFillRandomGaussian(Element *, size_t, uint64_t, cutlass::RealType<Element>::Type, cutlass::RealType<Element>::Type, int, cudaStream_t) [with Element=cutlass::uint2b_t]" at line 1746
            instantiation of "void cutlass::reference::device::BlockFillRandom(Element *, size_t, uint64_t, cutlass::Distribution, cudaStream_t) [with Element=cutlass::uint2b_t]" at line 669 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(494): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=2, Signed=false, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
        result = Element(rnd);
                 ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "Element cutlass::reference::device::detail::RandomUniformFunc<Element>::operator()() [with Element=cutlass::uint2b_t]" at line 149 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::BlockForEach<Element,Func>(Element *, size_t, Func::Params) [with Element=cutlass::uint2b_t, Func=cutlass::reference::device::detail::RandomUniformFunc<cutlass::uint2b_t>]" at line 122 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::BlockForEach<Element, Func>::BlockForEach(Element *, size_t, Func::Params, int, int, cudaStream_t) [with Element=cutlass::uint2b_t, Func=cutlass::reference::device::detail::RandomUniformFunc<cutlass::uint2b_t>]" at line 731
            instantiation of "void cutlass::reference::device::BlockFillRandomUniform(Element *, size_t, uint64_t, cutlass::RealType<Element>::Type, cutlass::RealType<Element>::Type, int, cudaStream_t) [with Element=cutlass::uint2b_t]" at line 1756
            instantiation of "void cutlass::reference::device::BlockFillRandom(Element *, size_t, uint64_t, cutlass::Distribution, cudaStream_t) [with Element=cutlass::uint2b_t]" at line 669 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(178): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=4, Signed=false, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
        result = Element(rnd);
                 ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "Element cutlass::reference::device::detail::RandomGaussianFunc<Element>::operator()() [with Element=cutlass::uint4b_t]" at line 149 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::BlockForEach<Element,Func>(Element *, size_t, Func::Params) [with Element=cutlass::uint4b_t, Func=cutlass::reference::device::detail::RandomGaussianFunc<cutlass::uint4b_t>]" at line 122 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::BlockForEach<Element, Func>::BlockForEach(Element *, size_t, Func::Params, int, int, cudaStream_t) [with Element=cutlass::uint4b_t, Func=cutlass::reference::device::detail::RandomGaussianFunc<cutlass::uint4b_t>]" at line 394
            instantiation of "void cutlass::reference::device::BlockFillRandomGaussian(Element *, size_t, uint64_t, cutlass::RealType<Element>::Type, cutlass::RealType<Element>::Type, int, cudaStream_t) [with Element=cutlass::uint4b_t]" at line 1746
            instantiation of "void cutlass::reference::device::BlockFillRandom(Element *, size_t, uint64_t, cutlass::Distribution, cudaStream_t) [with Element=cutlass::uint4b_t]" at line 677 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(494): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=4, Signed=false, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
        result = Element(rnd);
                 ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "Element cutlass::reference::device::detail::RandomUniformFunc<Element>::operator()() [with Element=cutlass::uint4b_t]" at line 149 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::BlockForEach<Element,Func>(Element *, size_t, Func::Params) [with Element=cutlass::uint4b_t, Func=cutlass::reference::device::detail::RandomUniformFunc<cutlass::uint4b_t>]" at line 122 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::BlockForEach<Element, Func>::BlockForEach(Element *, size_t, Func::Params, int, int, cudaStream_t) [with Element=cutlass::uint4b_t, Func=cutlass::reference::device::detail::RandomUniformFunc<cutlass::uint4b_t>]" at line 731
            instantiation of "void cutlass::reference::device::BlockFillRandomUniform(Element *, size_t, uint64_t, cutlass::RealType<Element>::Type, cutlass::RealType<Element>::Type, int, cudaStream_t) [with Element=cutlass::uint4b_t]" at line 1756
            instantiation of "void cutlass::reference::device::BlockFillRandom(Element *, size_t, uint64_t, cutlass::Distribution, cudaStream_t) [with Element=cutlass::uint4b_t]" at line 677 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(1626): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=2, Signed=true, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
            sum = Element(static_cast<float>(sum) +
                  ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "void cutlass::reference::device::detail::TensorFillLinearFunc<Element, Layout>::operator()(const cutlass::reference::device::detail::TensorFillLinearFunc<Element, Layout>::TensorCoord &) [with Element=cutlass::int2b_t, Layout=cutlass::layout::PackedVectorLayout]" at line 82 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "cutlass::reference::device::kernel::detail::TensorForEachHelper<Func, Rank, 0>::TensorForEachHelper(Func &, const cutlass::Coord<Rank, int, int64_t> &, cutlass::Coord<Rank, int, int64_t> &, int64_t) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::int2b_t, cutlass::layout::PackedVectorLayout>, Rank=1]" at line 109 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::TensorForEach<Func,Rank,Params>(cutlass::Coord<Rank, int, int64_t>, Params) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::int2b_t, cutlass::layout::PackedVectorLayout>, Rank=1, Params=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::int2b_t, cutlass::layout::PackedVectorLayout>::Params]" at line 59 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::TensorForEach<Func, Rank, Params>::TensorForEach(cutlass::Coord<Rank, int, int64_t>, Params, int, int, cudaStream_t) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::int2b_t, cutlass::layout::PackedVectorLayout>, Rank=1, Params=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::int2b_t, cutlass::layout::PackedVectorLayout>::Params]" at line 1661
            instantiation of "void cutlass::reference::device::TensorFillLinear(cutlass::TensorView<Element, Layout>, const cutlass::Array<Element, Layout::kRank, <expression>> &, Element, cudaStream_t) [with Element=cutlass::int2b_t, Layout=cutlass::layout::PackedVectorLayout]" at line 1719
            instantiation of "void cutlass::reference::device::BlockFillSequential(Element *, int64_t, Element, Element) [with Element=cutlass::int2b_t]" at line 1061 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(1626): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=4, Signed=true, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
            sum = Element(static_cast<float>(sum) +
                  ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "void cutlass::reference::device::detail::TensorFillLinearFunc<Element, Layout>::operator()(const cutlass::reference::device::detail::TensorFillLinearFunc<Element, Layout>::TensorCoord &) [with Element=cutlass::int4b_t, Layout=cutlass::layout::PackedVectorLayout]" at line 82 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "cutlass::reference::device::kernel::detail::TensorForEachHelper<Func, Rank, 0>::TensorForEachHelper(Func &, const cutlass::Coord<Rank, int, int64_t> &, cutlass::Coord<Rank, int, int64_t> &, int64_t) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::int4b_t, cutlass::layout::PackedVectorLayout>, Rank=1]" at line 109 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::TensorForEach<Func,Rank,Params>(cutlass::Coord<Rank, int, int64_t>, Params) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::int4b_t, cutlass::layout::PackedVectorLayout>, Rank=1, Params=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::int4b_t, cutlass::layout::PackedVectorLayout>::Params]" at line 59 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::TensorForEach<Func, Rank, Params>::TensorForEach(cutlass::Coord<Rank, int, int64_t>, Params, int, int, cudaStream_t) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::int4b_t, cutlass::layout::PackedVectorLayout>, Rank=1, Params=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::int4b_t, cutlass::layout::PackedVectorLayout>::Params]" at line 1661
            instantiation of "void cutlass::reference::device::TensorFillLinear(cutlass::TensorView<Element, Layout>, const cutlass::Array<Element, Layout::kRank, <expression>> &, Element, cudaStream_t) [with Element=cutlass::int4b_t, Layout=cutlass::layout::PackedVectorLayout]" at line 1719
            instantiation of "void cutlass::reference::device::BlockFillSequential(Element *, int64_t, Element, Element) [with Element=cutlass::int4b_t]" at line 1069 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(1626): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=1, Signed=false, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
            sum = Element(static_cast<float>(sum) +
                  ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "void cutlass::reference::device::detail::TensorFillLinearFunc<Element, Layout>::operator()(const cutlass::reference::device::detail::TensorFillLinearFunc<Element, Layout>::TensorCoord &) [with Element=cutlass::uint1b_t, Layout=cutlass::layout::PackedVectorLayout]" at line 82 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "cutlass::reference::device::kernel::detail::TensorForEachHelper<Func, Rank, 0>::TensorForEachHelper(Func &, const cutlass::Coord<Rank, int, int64_t> &, cutlass::Coord<Rank, int, int64_t> &, int64_t) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint1b_t, cutlass::layout::PackedVectorLayout>, Rank=1]" at line 109 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::TensorForEach<Func,Rank,Params>(cutlass::Coord<Rank, int, int64_t>, Params) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint1b_t, cutlass::layout::PackedVectorLayout>, Rank=1, Params=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint1b_t, cutlass::layout::PackedVectorLayout>::Params]" at line 59 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::TensorForEach<Func, Rank, Params>::TensorForEach(cutlass::Coord<Rank, int, int64_t>, Params, int, int, cudaStream_t) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint1b_t, cutlass::layout::PackedVectorLayout>, Rank=1, Params=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint1b_t, cutlass::layout::PackedVectorLayout>::Params]" at line 1661
            instantiation of "void cutlass::reference::device::TensorFillLinear(cutlass::TensorView<Element, Layout>, const cutlass::Array<Element, Layout::kRank, <expression>> &, Element, cudaStream_t) [with Element=cutlass::uint1b_t, Layout=cutlass::layout::PackedVectorLayout]" at line 1719
            instantiation of "void cutlass::reference::device::BlockFillSequential(Element *, int64_t, Element, Element) [with Element=cutlass::uint1b_t]" at line 1109 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(1626): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=2, Signed=false, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
            sum = Element(static_cast<float>(sum) +
                  ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "void cutlass::reference::device::detail::TensorFillLinearFunc<Element, Layout>::operator()(const cutlass::reference::device::detail::TensorFillLinearFunc<Element, Layout>::TensorCoord &) [with Element=cutlass::uint2b_t, Layout=cutlass::layout::PackedVectorLayout]" at line 82 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "cutlass::reference::device::kernel::detail::TensorForEachHelper<Func, Rank, 0>::TensorForEachHelper(Func &, const cutlass::Coord<Rank, int, int64_t> &, cutlass::Coord<Rank, int, int64_t> &, int64_t) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint2b_t, cutlass::layout::PackedVectorLayout>, Rank=1]" at line 109 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::TensorForEach<Func,Rank,Params>(cutlass::Coord<Rank, int, int64_t>, Params) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint2b_t, cutlass::layout::PackedVectorLayout>, Rank=1, Params=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint2b_t, cutlass::layout::PackedVectorLayout>::Params]" at line 59 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::TensorForEach<Func, Rank, Params>::TensorForEach(cutlass::Coord<Rank, int, int64_t>, Params, int, int, cudaStream_t) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint2b_t, cutlass::layout::PackedVectorLayout>, Rank=1, Params=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint2b_t, cutlass::layout::PackedVectorLayout>::Params]" at line 1661
            instantiation of "void cutlass::reference::device::TensorFillLinear(cutlass::TensorView<Element, Layout>, const cutlass::Array<Element, Layout::kRank, <expression>> &, Element, cudaStream_t) [with Element=cutlass::uint2b_t, Layout=cutlass::layout::PackedVectorLayout]" at line 1719
            instantiation of "void cutlass::reference::device::BlockFillSequential(Element *, int64_t, Element, Element) [with Element=cutlass::uint2b_t]" at line 1117 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_fill.h(1626): warning #1444-D: function "cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with Bits=4, Signed=false, T=float, Enable=void]" was declared deprecated ("Implicit conversion is deprecated; please use explicit construction instead")
            sum = Element(static_cast<float>(sum) +
                  ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h(77): note #3287-D: because of a "deprecated" attribute
    [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
      ^
          detected during:
            instantiation of "void cutlass::reference::device::detail::TensorFillLinearFunc<Element, Layout>::operator()(const cutlass::reference::device::detail::TensorFillLinearFunc<Element, Layout>::TensorCoord &) [with Element=cutlass::uint4b_t, Layout=cutlass::layout::PackedVectorLayout]" at line 82 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "cutlass::reference::device::kernel::detail::TensorForEachHelper<Func, Rank, 0>::TensorForEachHelper(Func &, const cutlass::Coord<Rank, int, int64_t> &, cutlass::Coord<Rank, int, int64_t> &, int64_t) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint4b_t, cutlass::layout::PackedVectorLayout>, Rank=1]" at line 109 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/kernel/tensor_foreach.h
            instantiation of "void cutlass::reference::device::kernel::TensorForEach<Func,Rank,Params>(cutlass::Coord<Rank, int, int64_t>, Params) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint4b_t, cutlass::layout::PackedVectorLayout>, Rank=1, Params=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint4b_t, cutlass::layout::PackedVectorLayout>::Params]" at line 59 of /home/ericauld/r/cutlass/tools/util/include/cutlass/util/reference/device/tensor_foreach.h
            instantiation of "cutlass::reference::device::TensorForEach<Func, Rank, Params>::TensorForEach(cutlass::Coord<Rank, int, int64_t>, Params, int, int, cudaStream_t) [with Func=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint4b_t, cutlass::layout::PackedVectorLayout>, Rank=1, Params=cutlass::reference::device::detail::TensorFillLinearFunc<cutlass::uint4b_t, cutlass::layout::PackedVectorLayout>::Params]" at line 1661
            instantiation of "void cutlass::reference::device::TensorFillLinear(cutlass::TensorView<Element, Layout>, const cutlass::Array<Element, Layout::kRank, <expression>> &, Element, cudaStream_t) [with Element=cutlass::uint4b_t, Layout=cutlass::layout::PackedVectorLayout]" at line 1719
            instantiation of "void cutlass::reference::device::BlockFillSequential(Element *, int64_t, Element, Element) [with Element=cutlass::uint4b_t]" at line 1125 of /home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu

/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu: In member function ‘void cutlass::profiler::DeviceAllocation::initialize_sequential_device(cutlass::Distribution)’:
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1061:175: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 2; bool Signed = true]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1061 |     cutlass::reference::device::BlockFillSequential<int2b_t>(
      |                                                                                                                                                                               ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1061:223: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 2; bool Signed = true]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1061 |     cutlass::reference::device::BlockFillSequential<int2b_t>(
      |                                                                                                                                                                                                                               ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1069:175: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 4; bool Signed = true]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1069 |     cutlass::reference::device::BlockFillSequential<int4b_t>(
      |                                                                                                                                                                               ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1069:223: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 4; bool Signed = true]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1069 |     cutlass::reference::device::BlockFillSequential<int4b_t>(
      |                                                                                                                                                                                                                               ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1109:178: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 1; bool Signed = false]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1109 |     cutlass::reference::device::BlockFillSequential<uint1b_t>(
      |                                                                                                                                                                                  ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1109:227: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 1; bool Signed = false]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1109 |     cutlass::reference::device::BlockFillSequential<uint1b_t>(
      |                                                                                                                                                                                                                                   ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1117:178: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 2; bool Signed = false]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1117 |     cutlass::reference::device::BlockFillSequential<uint2b_t>(
      |                                                                                                                                                                                  ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1117:227: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 2; bool Signed = false]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1117 |     cutlass::reference::device::BlockFillSequential<uint2b_t>(
      |                                                                                                                                                                                                                                   ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1125:178: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 4; bool Signed = false]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1125 |     cutlass::reference::device::BlockFillSequential<uint4b_t>(
      |                                                                                                                                                                                  ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1125:227: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 4; bool Signed = false]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1125 |     cutlass::reference::device::BlockFillSequential<uint4b_t>(
      |                                                                                                                                                                                                                                   ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu: In member function ‘void cutlass::profiler::DeviceAllocation::initialize_sequential_host(cutlass::Distribution)’:
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1291:181: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 2; bool Signed = true]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1291 |     cutlass::reference::host::BlockFillSequential<int2b_t>(
      |                                                                                                                                                                                     ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1291:229: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 2; bool Signed = true]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1291 |     cutlass::reference::host::BlockFillSequential<int2b_t>(
      |                                                                                                                                                                                                                                     ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1299:181: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 4; bool Signed = true]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1299 |     cutlass::reference::host::BlockFillSequential<int4b_t>(
      |                                                                                                                                                                                     ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1299:229: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 4; bool Signed = true]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1299 |     cutlass::reference::host::BlockFillSequential<int4b_t>(
      |                                                                                                                                                                                                                                     ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1339:184: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 1; bool Signed = false]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1339 |     cutlass::reference::host::BlockFillSequential<uint1b_t>(
      |                                                                                                                                                                                        ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1339:233: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 1; bool Signed = false]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1339 |     cutlass::reference::host::BlockFillSequential<uint1b_t>(
      |                                                                                                                                                                                                                                         ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1347:184: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 2; bool Signed = false]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1347 |     cutlass::reference::host::BlockFillSequential<uint2b_t>(
      |                                                                                                                                                                                        ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1347:233: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 2; bool Signed = false]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1347 |     cutlass::reference::host::BlockFillSequential<uint2b_t>(
      |                                                                                                                                                                                                                                         ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1355:184: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 4; bool Signed = false]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1355 |     cutlass::reference::host::BlockFillSequential<uint4b_t>(
      |                                                                                                                                                                                        ^
/home/ericauld/r/cutlass/include/cutlass/integer_subbyte.h:79:1: note: declared here
   79 |   integer_subbyte(T value)
      | ^ ~~~~~~~~~~~~~
/home/ericauld/r/cutlass/tools/profiler/src/device_allocation.cu:1355:233: warning: ‘cutlass::integer_subbyte<Bits, Signed>::integer_subbyte(T) [with T = double; Enable = void; int Bits = 4; bool Signed = false]’ is deprecated: Implicit conversion is deprecated; please use explicit construction instead [-Wdeprecated-declarations]
 1355 |     cutlass::reference::host::BlockFillSequential<uint4b_t>(
      |                                                                                                                                                                                                                                         ^
/home/ericauld/r/cutlass/include/cutlass/integer_s