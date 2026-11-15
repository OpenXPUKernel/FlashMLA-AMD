// Native AMD GPU short-row pipeline with compile-time row offsets.
#define FLASH_MLA_AMD_COMPACT_PIPELINE 1
#define FLASH_MLA_AMD_COMPACT_STATIC_LAYOUT 1
#define FLASH_MLA_AMD_COMPACT_DIRECT_LANE 1
#define FLASH_MLA_AMD_COMPACT_UNROLL_ROWS 1
#include "fused_sparse_attention_compact.hip"
