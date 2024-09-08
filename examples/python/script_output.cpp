using namespace cute;

/*
EA: This is expressing the computation

O, alpha, C, beta, E (aux), b
      |->
F, F + ReLU(F + 1) + b

where

F := alpha * O + beta * C + E
 */

using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
  cute::Shape<_128, _128, _64>, cutlass::epilogue::collective::EpilogueTileAuto,
  cutlass::half_t, cutlass::half_t,
  cutlass::epilogue::TmaWarpSpecialized
>;

using ElementC = cutlass::half_t;
using StrideC = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
using TensorC = cutlass::epilogue::fusion::Sm90SrcFetch<cutlass::half_t>;

using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

using Alpha = cutlass::epilogue::fusion::Sm90ScalarBroadcast<
    float, cute::Stride<cute::Int<0>, cute::Int<0>, cute::Int<0>>, 1, cutlass::multiplies
>;

using AuxDescriptor = cutlass::epilogue::collective::detail::AuxLoadDescriptor<EpilogueDescriptor, cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, cutlass::half_t>;

using Aux = cutlass::epilogue::fusion::Sm90AuxLoad<
    AuxDescriptor::Stages, typename AuxDescriptor::EpilogueTile, cutlass::half_t,
    cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, typename AuxDescriptor::SmemLayoutAtom, typename AuxDescriptor::CopyOpS2R
>;

/*
EA: template params for struct Sm90AuxLoad:
  int Stages,
  class EpilogueTile,
  class Element,
  class StrideMNL,
  class SmemLayoutAtom,
  class CopyOpS2R,
  int Alignment = 128 / sizeof_bits_v<Element>,
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
*/

using Beta = cutlass::epilogue::fusion::Sm90ScalarBroadcast<
    float, cute::Stride<cute::Int<0>, cute::Int<0>, cute::Int<0>>, 1, cutlass::multiplies
>;

using Bias = cutlass::epilogue::fusion::Sm90ColBroadcast<
    0 /*Stages*/, typename EpilogueDescriptor::TileShape, cutlass::half_t,
    cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>
>;

using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<
    Compute0,
    Alpha,
    Accum>;

using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<
    Compute1,
    Beta,
    TensorC>;

using Compute2 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute2 = cutlass::epilogue::fusion::Sm90EVT<
    Compute2,
    EVTCompute1,
    Aux>;

using Compute3 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute3 = cutlass::epilogue::fusion::Sm90EVT<
    Compute3,
    EVTCompute0,
    EVTCompute2>;

using FDescriptor = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
    EpilogueDescriptor, cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, cutlass::half_t
>;

using F = cutlass::epilogue::fusion::Sm90AuxStore<
    FDescriptor::Stages, typename FDescriptor::EpilogueTile, cutlass::half_t,
    cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, typename FDescriptor::SmemLayoutAtom,
    typename FDescriptor::CopyOpR2S
>;

using EVTF = cutlass::epilogue::fusion::Sm90EVT<
    F,
    EVTCompute3>;

using Imm10 = cutlass::epilogue::fusion::Sm90ScalarBroadcast<
    float, cute::Stride<cute::Int<0>, cute::Int<0>, cute::Int<0>>, 1, cutlass::multiplies
>;

using Compute4 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute5 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::epilogue::thread::ReLu, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute6 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute7 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, cutlass::half_t, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using DagCompute7 = cutlass::epilogue::fusion::Sm90TopologicalVisitor<
    float,
    cute::tuple<
        cute::seq<>,
        cute::seq<>,
        cute::seq<>,
        cute::seq<0, 2>,
        cute::seq<3>,
        cute::seq<4, 1>,
        cute::seq<5, 0>,
    >,
    EVTF,
    Bias,
    Imm10,
    Compute4,
    Compute5,
    Compute6,
    Compute7
>;

using ElementD = cutlass::half_t;
using StrideD = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;



using CollectiveEpilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2,cute::_1,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    ElementC, StrideC, 8,
    ElementD, StrideD, 8,
    cutlass::epilogue::TmaWarpSpecialized,
    DagCompute7
  >::CollectiveOp;

using CollectiveMainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2,cute::_1,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_2x1x1_0_ttt_align8_warpspecialized_pingpong_epi_tma
using cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_2x1x1_0_ttt_align8_warpspecialized_pingpong_epi_tma_base = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler
>;

// Define named type
struct cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_2x1x1_0_ttt_align8_warpspecialized_pingpong_epi_tma_type :
  public cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_2x1x1_0_ttt_align8_warpspecialized_pingpong_epi_tma_base { };

