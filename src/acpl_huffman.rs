//! A-CPL (Advanced Coupling) Huffman codebooks (Annex A.3, Tables
//! A.34..A.57 of ETSI TS 103 190-1 V1.4.1).
//!
//! Twenty-four Huffman codebooks ship with the AC-4 spec for advanced
//! coupling. They are organised as four `(data_type)` × two
//! `(quant_mode)` × three `(hcb_type)` matrix:
//!
//! ```text
//!   data_type ∈ { ALPHA, BETA, BETA3, GAMMA }
//!   quant_mode ∈ { COARSE, FINE }
//!   hcb_type   ∈ { F0, DF, DT }
//! ```
//!
//! All 24 are walked by §4.3.11.6.1 Pseudocode 8 `get_acpl_hcb()` —
//! the spec-mandated lookup that the `acpl_huff_data()` parser
//! (§4.2.13.7 Table 65) hits at runtime to pick the right codebook.
//!
//! Per the spec, `acpl_huff_data()` calls `huff_decode_diff` for *every*
//! symbol — F0, DF and DT alike — meaning the recovered value is always
//! `symbol_index - cb_off`. For F0 codebooks of ALPHA / BETA / BETA3 the
//! `cb_off` is 0, so it's a no-op; for the GAMMA F0 codebooks the
//! `cb_off` is non-zero (10 for COARSE, 20 for FINE), so it actually
//! shifts the recovered value.
//!
//! The 24 tables are transcribed verbatim from the normative ETSI
//! accompaniment file `ts_10319001v010401p0-tables.c` (lines 521..793
//! of that file). They are uncopyrightable numeric constants that
//! exactly track the published §A.3 / Tables A.34..A.57 codebook
//! headers.
//!
//! Naming mirrors the C accompaniment file 1:1 so the mapping is
//! auditable:
//!
//! ```text
//!   ACPL_HCB_ALPHA_COARSE_F0   ALPHA_COARSE_DF   ALPHA_COARSE_DT
//!   ACPL_HCB_ALPHA_FINE_F0     ALPHA_FINE_DF     ALPHA_FINE_DT
//!   ACPL_HCB_BETA_COARSE_F0    BETA_COARSE_DF    BETA_COARSE_DT
//!   ACPL_HCB_BETA_FINE_F0      BETA_FINE_DF      BETA_FINE_DT
//!   ACPL_HCB_BETA3_COARSE_F0   BETA3_COARSE_DF   BETA3_COARSE_DT
//!   ACPL_HCB_BETA3_FINE_F0     BETA3_FINE_DF     BETA3_FINE_DT
//!   ACPL_HCB_GAMMA_COARSE_F0   GAMMA_COARSE_DF   GAMMA_COARSE_DT
//!   ACPL_HCB_GAMMA_FINE_F0     GAMMA_FINE_DF     GAMMA_FINE_DT
//! ```
//!
//! Widest codeword across all 24 codebooks is 18 bits
//! (`ACPL_HCB_GAMMA_FINE_DT`), so `u32` is more than enough.

// ============================================================================
// ALPHA (Table A.34..A.39)
// ============================================================================

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.34 — ACPL_HCB_ALPHA_COARSE_F0_LEN[17]
pub static ACPL_HCB_ALPHA_COARSE_F0_LEN: &[u8] =
    &[10, 10, 9, 8, 6, 6, 5, 2, 1, 3, 5, 7, 7, 8, 9, 10, 10];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.34 — ACPL_HCB_ALPHA_COARSE_F0_CW[17]
pub static ACPL_HCB_ALPHA_COARSE_F0_CW: &[u32] = &[
    0x0003be, 0x0003fe, 0x0001fe, 0x0000fe, 0x00003e, 0x00003a, 0x00001e, 0x000002, 0x000000,
    0x000006, 0x00001c, 0x00007e, 0x000076, 0x0000ee, 0x0001de, 0x0003ff, 0x0003bf,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.35 — ACPL_HCB_ALPHA_FINE_F0_LEN[33]
pub static ACPL_HCB_ALPHA_FINE_F0_LEN: &[u8] = &[
    10, 12, 11, 11, 10, 10, 9, 8, 7, 7, 8, 7, 6, 6, 4, 3, 1, 3, 4, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10,
    10, 11, 12, 10,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.35 — ACPL_HCB_ALPHA_FINE_F0_CW[33]
pub static ACPL_HCB_ALPHA_FINE_F0_CW: &[u32] = &[
    0x0002ce, 0x000b5e, 0x0004fe, 0x0005ae, 0x00027e, 0x0002de, 0x00016a, 0x0000b2, 0x00004a,
    0x00004b, 0x0000b6, 0x00004e, 0x000024, 0x00002e, 0x00000a, 0x000006, 0x000000, 0x000007,
    0x000008, 0x00002f, 0x000026, 0x000058, 0x0000b4, 0x00009e, 0x00016e, 0x000166, 0x0002df,
    0x0002cf, 0x00027c, 0x00027d, 0x0004ff, 0x000b5f, 0x0002d6,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.36 — ACPL_HCB_ALPHA_COARSE_DF_LEN[33]
pub static ACPL_HCB_ALPHA_COARSE_DF_LEN: &[u8] = &[
    15, 18, 17, 17, 16, 15, 15, 13, 12, 11, 10, 9, 8, 7, 4, 3, 1, 2, 5, 7, 8, 9, 10, 11, 12, 13,
    15, 16, 16, 17, 16, 18, 15,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.36 — ACPL_HCB_ALPHA_COARSE_DF_CW[33]
pub static ACPL_HCB_ALPHA_COARSE_DF_CW: &[u32] = &[
    0x007c76, 0x03e3fe, 0x01f1f6, 0x01f1f7, 0x00f8ea, 0x007c74, 0x007c7c, 0x001f1c, 0x000f9e,
    0x0007ce, 0x0003e2, 0x0001f0, 0x0000fa, 0x00007e, 0x00000e, 0x000006, 0x000000, 0x000002,
    0x00001e, 0x00007f, 0x0000fb, 0x0001f2, 0x0003e6, 0x0007c6, 0x000f9f, 0x001f1e, 0x007c7e,
    0x00f8fe, 0x00f8fa, 0x01f1fe, 0x00f8eb, 0x03e3ff, 0x007c77,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.37 — ACPL_HCB_ALPHA_FINE_DF_LEN[65]
pub static ACPL_HCB_ALPHA_FINE_DF_LEN: &[u8] = &[
    13, 17, 17, 17, 16, 17, 17, 17, 17, 16, 16, 16, 15, 15, 14, 13, 13, 12, 12, 11, 11, 11, 10, 10,
    10, 9, 8, 7, 7, 5, 4, 3, 1, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 15,
    15, 16, 16, 17, 16, 16, 17, 16, 16, 17, 17, 17, 13,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.37 — ACPL_HCB_ALPHA_FINE_DF_CW[65]
pub static ACPL_HCB_ALPHA_FINE_DF_CW: &[u32] = &[
    0x0011de, 0x011ffe, 0x013dea, 0x013df6, 0x008eea, 0x013df7, 0x013dee, 0x013deb, 0x013dec,
    0x008eee, 0x008ffe, 0x009efe, 0x0047fe, 0x004f7c, 0x0023fe, 0x0011fe, 0x0013fe, 0x0008f6,
    0x0009ee, 0x000476, 0x00047a, 0x0004f6, 0x00023a, 0x00027a, 0x00027e, 0x00013e, 0x00009a,
    0x00004c, 0x00004e, 0x000012, 0x00000a, 0x000006, 0x000000, 0x000007, 0x00000b, 0x000010,
    0x000022, 0x000046, 0x00009b, 0x00013c, 0x00011c, 0x00023e, 0x00023c, 0x0004fe, 0x00047e,
    0x0009fe, 0x0008fe, 0x0008f7, 0x0013ff, 0x0011df, 0x0027bc, 0x004f7e, 0x004776, 0x009efa,
    0x009ef4, 0x013dfe, 0x008eeb, 0x008ee8, 0x013dff, 0x008ee9, 0x008eef, 0x011fff, 0x013ded,
    0x013def, 0x0011dc,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.38 — ACPL_HCB_ALPHA_COARSE_DT_LEN[33]
pub static ACPL_HCB_ALPHA_COARSE_DT_LEN: &[u8] = &[
    14, 16, 15, 16, 15, 15, 14, 13, 12, 12, 10, 9, 8, 7, 5, 3, 1, 2, 4, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 15, 16, 15, 16, 14,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.38 — ACPL_HCB_ALPHA_COARSE_DT_CW[33]
pub static ACPL_HCB_ALPHA_COARSE_DT_CW: &[u32] = &[
    0x003efc, 0x00fbfa, 0x007ddc, 0x00fbfe, 0x007dde, 0x007dfc, 0x003ef6, 0x001f76, 0x000fba,
    0x000fbe, 0x0003ec, 0x0001f2, 0x0000f8, 0x00007e, 0x00001e, 0x000006, 0x000000, 0x000002,
    0x00000e, 0x00007f, 0x0000fa, 0x0001f3, 0x0003ed, 0x0007dc, 0x000fbc, 0x001f7a, 0x003ef7,
    0x007dfe, 0x007ddf, 0x00fbff, 0x007ddd, 0x00fbfb, 0x003efd,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.39 — ACPL_HCB_ALPHA_FINE_DT_LEN[65]
pub static ACPL_HCB_ALPHA_FINE_DT_LEN: &[u8] = &[
    16, 18, 18, 18, 17, 17, 17, 18, 17, 17, 17, 16, 16, 16, 15, 15, 14, 14, 13, 13, 13, 12, 11, 11,
    10, 10, 9, 9, 7, 6, 5, 3, 1, 2, 5, 6, 7, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15,
    16, 16, 16, 17, 17, 17, 17, 17, 18, 17, 18, 18, 18, 16,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.39 — ACPL_HCB_ALPHA_FINE_DT_CW[65]
pub static ACPL_HCB_ALPHA_FINE_DT_CW: &[u32] = &[
    0x00eeee, 0x03b3ee, 0x03b3f6, 0x03b3fc, 0x01d9bc, 0x01d9bd, 0x01d9b2, 0x03b3fe, 0x01d9be,
    0x01d9f6, 0x01d9fc, 0x00ecda, 0x00ecfa, 0x00eeef, 0x00766e, 0x007776, 0x003b3a, 0x003bba,
    0x001d9a, 0x001ddc, 0x001dde, 0x000eec, 0x000764, 0x000772, 0x0003b0, 0x0003b8, 0x0001da,
    0x0001de, 0x000072, 0x000038, 0x00001e, 0x000006, 0x000000, 0x000002, 0x00001f, 0x00003a,
    0x000073, 0x0001df, 0x0001db, 0x0003ba, 0x0003b1, 0x000773, 0x000765, 0x000eed, 0x000ecc,
    0x001d9e, 0x001d9c, 0x003bbe, 0x003b3b, 0x00777e, 0x00767c, 0x00eefe, 0x00ecfc, 0x00ecd8,
    0x01d9fd, 0x01d9fa, 0x01d9bf, 0x01d9b6, 0x01d9b3, 0x03b3fd, 0x01d9b7, 0x03b3ff, 0x03b3ef,
    0x03b3f7, 0x00eeff,
];

// ============================================================================
// BETA (Table A.40..A.45)
// ============================================================================

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.40 — ACPL_HCB_BETA_COARSE_F0_LEN[5]
pub static ACPL_HCB_BETA_COARSE_F0_LEN: &[u8] = &[1, 2, 3, 4, 4];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.40 — ACPL_HCB_BETA_COARSE_F0_CW[5]
pub static ACPL_HCB_BETA_COARSE_F0_CW: &[u32] = &[0x000000, 0x000002, 0x000006, 0x00000e, 0x00000f];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.41 — ACPL_HCB_BETA_FINE_F0_LEN[9]
pub static ACPL_HCB_BETA_FINE_F0_LEN: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8, 8];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.41 — ACPL_HCB_BETA_FINE_F0_CW[9]
pub static ACPL_HCB_BETA_FINE_F0_CW: &[u32] = &[
    0x000000, 0x000002, 0x000006, 0x00000e, 0x00001e, 0x00003e, 0x00007e, 0x0000fe, 0x0000ff,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.42 — ACPL_HCB_BETA_COARSE_DF_LEN[9]
pub static ACPL_HCB_BETA_COARSE_DF_LEN: &[u8] = &[8, 6, 4, 3, 1, 2, 5, 7, 8];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.42 — ACPL_HCB_BETA_COARSE_DF_CW[9]
pub static ACPL_HCB_BETA_COARSE_DF_CW: &[u32] = &[
    0x0000fe, 0x00003e, 0x00000e, 0x000006, 0x000000, 0x000002, 0x00001e, 0x00007e, 0x0000ff,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.43 — ACPL_HCB_BETA_FINE_DF_LEN[17]
pub static ACPL_HCB_BETA_FINE_DF_LEN: &[u8] =
    &[13, 12, 10, 9, 8, 7, 5, 3, 1, 2, 4, 7, 8, 9, 9, 11, 13];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.43 — ACPL_HCB_BETA_FINE_DF_CW[17]
pub static ACPL_HCB_BETA_FINE_DF_CW: &[u32] = &[
    0x001f1e, 0x000f8e, 0x0003e2, 0x0001f2, 0x0000fa, 0x00007e, 0x00001e, 0x000006, 0x000000,
    0x000002, 0x00000e, 0x00007f, 0x0000fb, 0x0001f3, 0x0001f0, 0x0007c6, 0x001f1f,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.44 — ACPL_HCB_BETA_COARSE_DT_LEN[9]
pub static ACPL_HCB_BETA_COARSE_DT_LEN: &[u8] = &[8, 7, 5, 3, 1, 2, 4, 6, 8];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.44 — ACPL_HCB_BETA_COARSE_DT_CW[9]
pub static ACPL_HCB_BETA_COARSE_DT_CW: &[u32] = &[
    0x0000fe, 0x00007e, 0x00001e, 0x000006, 0x000000, 0x000002, 0x00000e, 0x00003e, 0x0000ff,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.45 — ACPL_HCB_BETA_FINE_DT_LEN[17]
pub static ACPL_HCB_BETA_FINE_DT_LEN: &[u8] =
    &[15, 14, 12, 10, 8, 7, 5, 3, 1, 2, 4, 7, 7, 9, 11, 13, 15];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.45 — ACPL_HCB_BETA_FINE_DT_CW[17]
pub static ACPL_HCB_BETA_FINE_DT_CW: &[u32] = &[
    0x007dfe, 0x003efe, 0x000fbe, 0x0003ee, 0x0000fa, 0x00007e, 0x00001e, 0x000006, 0x000000,
    0x000002, 0x00000e, 0x00007f, 0x00007c, 0x0001f6, 0x0007de, 0x001f7e, 0x007dff,
];

// ============================================================================
// BETA3 (Table A.46..A.51)
// ============================================================================

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.46 — ACPL_HCB_BETA3_COARSE_F0_LEN[9]
pub static ACPL_HCB_BETA3_COARSE_F0_LEN: &[u8] = &[5, 3, 3, 2, 2, 3, 4, 6, 6];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.46 — ACPL_HCB_BETA3_COARSE_F0_CW[9]
pub static ACPL_HCB_BETA3_COARSE_F0_CW: &[u32] = &[
    0x000001, 0x000006, 0x000007, 0x000001, 0x000002, 0x000001, 0x000001, 0x000001, 0x000000,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.47 — ACPL_HCB_BETA3_FINE_F0_LEN[17]
pub static ACPL_HCB_BETA3_FINE_F0_LEN: &[u8] = &[7, 5, 4, 4, 4, 3, 3, 3, 3, 3, 4, 5, 6, 6, 7, 7, 7];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.47 — ACPL_HCB_BETA3_FINE_F0_CW[17]
pub static ACPL_HCB_BETA3_FINE_F0_CW: &[u32] = &[
    0x00000d, 0x000002, 0x000000, 0x00000c, 0x00000e, 0x000001, 0x000003, 0x000005, 0x000004,
    0x000002, 0x00000d, 0x00001f, 0x00003d, 0x000007, 0x000078, 0x00000c, 0x000079,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.48 — ACPL_HCB_BETA3_COARSE_DF_LEN[17]
pub static ACPL_HCB_BETA3_COARSE_DF_LEN: &[u8] =
    &[13, 12, 12, 11, 9, 6, 4, 2, 1, 3, 5, 7, 9, 11, 12, 13, 9];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.48 — ACPL_HCB_BETA3_COARSE_DF_CW[17]
pub static ACPL_HCB_BETA3_COARSE_DF_CW: &[u32] = &[
    0x000a93, 0x000548, 0x00054b, 0x0002a7, 0x0000ab, 0x000014, 0x000004, 0x000000, 0x000001,
    0x000003, 0x00000b, 0x00002b, 0x0000aa, 0x0002a6, 0x00054a, 0x000a92, 0x0000a8,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.49 — ACPL_HCB_BETA3_FINE_DF_LEN[33]
pub static ACPL_HCB_BETA3_FINE_DF_LEN: &[u8] = &[
    14, 15, 14, 13, 13, 12, 11, 11, 9, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12,
    13, 14, 14, 14, 15,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.49 — ACPL_HCB_BETA3_FINE_DF_CW[33]
pub static ACPL_HCB_BETA3_FINE_DF_CW: &[u32] = &[
    0x0019e9, 0x0033f7, 0x0019f3, 0x000cf5, 0x000cfc, 0x00067d, 0x00033c, 0x0007ff, 0x0000ce,
    0x000066, 0x000032, 0x000018, 0x00000d, 0x000007, 0x000002, 0x000000, 0x000002, 0x000006,
    0x00000e, 0x00001e, 0x00003e, 0x00007e, 0x0000fe, 0x0001fe, 0x0003fe, 0x0007fe, 0x00067f,
    0x00067b, 0x000cf8, 0x0019fa, 0x0019f2, 0x0019e8, 0x0033f6,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.50 — ACPL_HCB_BETA3_COARSE_DT_LEN[17]
pub static ACPL_HCB_BETA3_COARSE_DT_LEN: &[u8] =
    &[15, 15, 14, 12, 10, 7, 5, 3, 1, 2, 4, 6, 8, 11, 14, 14, 9];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.50 — ACPL_HCB_BETA3_COARSE_DT_CW[17]
pub static ACPL_HCB_BETA3_COARSE_DT_CW: &[u32] = &[
    0x000adc, 0x000add, 0x00056c, 0x00015a, 0x000057, 0x00000b, 0x000003, 0x000001, 0x000001,
    0x000001, 0x000000, 0x000004, 0x000014, 0x0000ac, 0x00056f, 0x00056d, 0x00002a,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.51 — ACPL_HCB_BETA3_FINE_DT_LEN[33]
pub static ACPL_HCB_BETA3_FINE_DT_LEN: &[u8] = &[
    16, 16, 16, 16, 16, 16, 15, 14, 12, 11, 10, 9, 8, 7, 5, 3, 1, 2, 4, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 15, 16, 16, 16, 16,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.51 — ACPL_HCB_BETA3_FINE_DT_CW[33]
pub static ACPL_HCB_BETA3_FINE_DT_CW: &[u32] = &[
    0x00501e, 0x00501d, 0x00501c, 0x00501b, 0x00510e, 0x00510d, 0x002809, 0x001442, 0x000500,
    0x000281, 0x000141, 0x0000a1, 0x000052, 0x00002a, 0x00000b, 0x000003, 0x000001, 0x000000,
    0x000004, 0x00002b, 0x000053, 0x0000a3, 0x000145, 0x000289, 0x000511, 0x000a20, 0x001405,
    0x00280c, 0x002808, 0x00510f, 0x00510c, 0x00501f, 0x00501a,
];

// ============================================================================
// GAMMA (Table A.52..A.57)
// ============================================================================

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.52 — ACPL_HCB_GAMMA_COARSE_F0_LEN[21]
pub static ACPL_HCB_GAMMA_COARSE_F0_LEN: &[u8] = &[
    13, 13, 13, 13, 11, 9, 7, 6, 5, 3, 2, 3, 3, 4, 3, 3, 8, 11, 12, 13, 13,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.52 — ACPL_HCB_GAMMA_COARSE_F0_CW[21]
pub static ACPL_HCB_GAMMA_COARSE_F0_CW: &[u32] = &[
    0x000af4, 0x000af8, 0x000af9, 0x000afb, 0x0002bc, 0x0000ae, 0x00002a, 0x000014, 0x00000b,
    0x000001, 0x000003, 0x000005, 0x000000, 0x000004, 0x000004, 0x000003, 0x000056, 0x0002bf,
    0x00057b, 0x000af5, 0x000afa,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.53 — ACPL_HCB_GAMMA_FINE_F0_LEN[41]
pub static ACPL_HCB_GAMMA_FINE_F0_LEN: &[u8] = &[
    12, 13, 13, 12, 12, 12, 12, 11, 9, 10, 9, 8, 8, 7, 7, 6, 5, 5, 4, 4, 3, 3, 4, 4, 5, 5, 5, 5, 4,
    3, 4, 7, 8, 9, 10, 11, 11, 12, 12, 12, 12,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.53 — ACPL_HCB_GAMMA_FINE_F0_CW[41]
pub static ACPL_HCB_GAMMA_FINE_F0_CW: &[u32] = &[
    0x0004b6, 0x001c6d, 0x001c6c, 0x00049b, 0x0004b5, 0x0004b7, 0x000e35, 0x00024e, 0x0001c7,
    0x00038c, 0x000097, 0x000048, 0x0000e2, 0x000070, 0x000073, 0x000013, 0x000008, 0x000017,
    0x000005, 0x00000c, 0x000004, 0x000001, 0x00000d, 0x00000a, 0x00001f, 0x00001e, 0x000016,
    0x00001d, 0x000006, 0x000000, 0x000007, 0x000072, 0x00004a, 0x000092, 0x00012c, 0x00024f,
    0x00024c, 0x000e34, 0x0004b4, 0x00049a, 0x000e37,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.54 — ACPL_HCB_GAMMA_COARSE_DF_LEN[41]
pub static ACPL_HCB_GAMMA_COARSE_DF_LEN: &[u8] = &[
    16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 14, 13, 13, 11, 10, 8, 7, 4, 2, 1, 3, 5, 7, 8, 10,
    11, 13, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 8,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.54 — ACPL_HCB_GAMMA_COARSE_DF_CW[41]
pub static ACPL_HCB_GAMMA_COARSE_DF_CW: &[u32] = &[
    0x0053e1, 0x0053e0, 0x0053db, 0x0053da, 0x0053d9, 0x0053e2, 0x0053e4, 0x0053ea, 0x0053eb,
    0x0029ea, 0x0029f4, 0x0014f4, 0x000a78, 0x000a7f, 0x000299, 0x00014d, 0x000051, 0x00002a,
    0x000004, 0x000000, 0x000001, 0x000003, 0x00000b, 0x00002b, 0x000052, 0x00014e, 0x000298,
    0x000a7e, 0x000a79, 0x0014f7, 0x0029f6, 0x0053ef, 0x0053ee, 0x0053e7, 0x0053e6, 0x0053e3,
    0x0053e5, 0x0053d8, 0x0053d7, 0x0053d6, 0x000050,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.55 — ACPL_HCB_GAMMA_FINE_DF_LEN[81]
pub static ACPL_HCB_GAMMA_FINE_DF_LEN: &[u8] = &[
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 16, 16, 16, 15, 15, 15,
    14, 14, 13, 13, 13, 12, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 1, 3, 4, 5, 6, 7, 9, 9, 10, 11, 11,
    12, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.55 — ACPL_HCB_GAMMA_FINE_DF_CW[81]
pub static ACPL_HCB_GAMMA_FINE_DF_CW: &[u32] = &[
    0x013e1f, 0x013e35, 0x013e1e, 0x013e1d, 0x013e1c, 0x013e1b, 0x013e1a, 0x013e19, 0x013e34,
    0x013e33, 0x013e18, 0x013ec2, 0x013ec1, 0x013ece, 0x013edf, 0x013e17, 0x013ede, 0x013edd,
    0x009d52, 0x009f18, 0x009f1b, 0x004eaa, 0x004ea8, 0x004fb1, 0x002753, 0x002757, 0x0013a8,
    0x0013e0, 0x0013ee, 0x0009d6, 0x0004e9, 0x0004fa, 0x00027b, 0x00013c, 0x00009c, 0x00004d,
    0x000021, 0x000012, 0x00000b, 0x000007, 0x000000, 0x000006, 0x00000a, 0x000011, 0x000020,
    0x00004c, 0x00013f, 0x00013b, 0x00027a, 0x0004f9, 0x0004e8, 0x0009d7, 0x0013ef, 0x0013e2,
    0x0027da, 0x0027c7, 0x002752, 0x004fb6, 0x004eac, 0x004eab, 0x009f65, 0x009d5a, 0x009d53,
    0x013ecd, 0x013edc, 0x013ecc, 0x013ecf, 0x013ec9, 0x013e32, 0x013ec3, 0x013e16, 0x013ec0,
    0x013ec8, 0x013e15, 0x013e14, 0x013e13, 0x013e12, 0x013e11, 0x013e10, 0x013ab7, 0x013ab6,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.56 — ACPL_HCB_GAMMA_COARSE_DT_LEN[41]
pub static ACPL_HCB_GAMMA_COARSE_DT_LEN: &[u8] = &[
    17, 17, 17, 17, 16, 17, 16, 16, 16, 15, 14, 13, 12, 12, 10, 9, 8, 7, 5, 3, 1, 2, 4, 7, 8, 10,
    11, 12, 13, 13, 14, 15, 16, 16, 16, 17, 17, 17, 17, 17, 9,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.56 — ACPL_HCB_GAMMA_COARSE_DT_CW[41]
pub static ACPL_HCB_GAMMA_COARSE_DT_CW: &[u32] = &[
    0x00a7f3, 0x00a7f1, 0x00a7f9, 0x00a7f8, 0x0050e1, 0x00a7fe, 0x0050e8, 0x0050eb, 0x0053fe,
    0x0029fd, 0x00143b, 0x000a1b, 0x00050c, 0x00053e, 0x000142, 0x0000a0, 0x000052, 0x00002b,
    0x00000b, 0x000003, 0x000001, 0x000000, 0x000004, 0x00002a, 0x000051, 0x00014e, 0x00029e,
    0x00050f, 0x000a7e, 0x000a1a, 0x001439, 0x002871, 0x0050ea, 0x0050e9, 0x0050e0, 0x00a7ff,
    0x00a7fb, 0x00a7fa, 0x00a7f2, 0x00a7f0, 0x0000a6,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.57 — ACPL_HCB_GAMMA_FINE_DT_LEN[81]
pub static ACPL_HCB_GAMMA_FINE_DT_LEN: &[u8] = &[
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 16, 16, 16, 15,
    15, 15, 14, 14, 13, 13, 12, 12, 11, 10, 9, 8, 7, 6, 5, 2, 1, 3, 5, 6, 7, 8, 10, 10, 11, 12, 13,
    13, 14, 14, 15, 15, 15, 15, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 18, 18, 18,
];

// FROM: ts_10319001v010401p0-tables.c §A.3 Table A.57 — ACPL_HCB_GAMMA_FINE_DT_CW[81]
pub static ACPL_HCB_GAMMA_FINE_DT_CW: &[u32] = &[
    0x031e44, 0x031d1d, 0x031e42, 0x031e16, 0x031e41, 0x031e47, 0x031d1c, 0x031e43, 0x031e73,
    0x031e72, 0x031e15, 0x031e70, 0x031e75, 0x031e7f, 0x031e7e, 0x018e88, 0x018d8b, 0x018e8f,
    0x018f0e, 0x018f3e, 0x00c746, 0x00c796, 0x00c79e, 0x006361, 0x0063c9, 0x0063d8, 0x0031d0,
    0x0031e6, 0x0018d9, 0x0018f1, 0x000c6d, 0x000c7a, 0x00063b, 0x00031c, 0x00018c, 0x0000c1,
    0x000062, 0x000033, 0x00001b, 0x000002, 0x000000, 0x000007, 0x00001a, 0x000032, 0x000061,
    0x0000c0, 0x00031f, 0x00031a, 0x000637, 0x000c75, 0x0018f7, 0x0018e9, 0x0031ed, 0x0031e0,
    0x0063d9, 0x0063ca, 0x006363, 0x006360, 0x00c786, 0x00c745, 0x018f3b, 0x018f2e, 0x018e89,
    0x018d88, 0x018d8a, 0x018d89, 0x031e5f, 0x031e74, 0x031e40, 0x031e71, 0x031e46, 0x031e5e,
    0x031e1f, 0x031e45, 0x031e1e, 0x031e14, 0x031e17, 0x031e13, 0x031e12, 0x031e11, 0x031e10,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// One row of the §A.3 sweep — `(name, len_table, cw_table,
    /// codebook_length, cb_off)`. The last two fields are taken
    /// straight from the published spec table headers.
    type AcplTableRow = (&'static str, &'static [u8], &'static [u32], usize, i32);

    /// All 24 tables paired up so we can sweep them with one helper.
    fn all_acpl_tables() -> Vec<AcplTableRow> {
        vec![
            (
                "ACPL_HCB_ALPHA_COARSE_F0",
                ACPL_HCB_ALPHA_COARSE_F0_LEN,
                ACPL_HCB_ALPHA_COARSE_F0_CW,
                17,
                0,
            ),
            (
                "ACPL_HCB_ALPHA_FINE_F0",
                ACPL_HCB_ALPHA_FINE_F0_LEN,
                ACPL_HCB_ALPHA_FINE_F0_CW,
                33,
                0,
            ),
            (
                "ACPL_HCB_ALPHA_COARSE_DF",
                ACPL_HCB_ALPHA_COARSE_DF_LEN,
                ACPL_HCB_ALPHA_COARSE_DF_CW,
                33,
                16,
            ),
            (
                "ACPL_HCB_ALPHA_FINE_DF",
                ACPL_HCB_ALPHA_FINE_DF_LEN,
                ACPL_HCB_ALPHA_FINE_DF_CW,
                65,
                32,
            ),
            (
                "ACPL_HCB_ALPHA_COARSE_DT",
                ACPL_HCB_ALPHA_COARSE_DT_LEN,
                ACPL_HCB_ALPHA_COARSE_DT_CW,
                33,
                16,
            ),
            (
                "ACPL_HCB_ALPHA_FINE_DT",
                ACPL_HCB_ALPHA_FINE_DT_LEN,
                ACPL_HCB_ALPHA_FINE_DT_CW,
                65,
                32,
            ),
            (
                "ACPL_HCB_BETA_COARSE_F0",
                ACPL_HCB_BETA_COARSE_F0_LEN,
                ACPL_HCB_BETA_COARSE_F0_CW,
                5,
                0,
            ),
            (
                "ACPL_HCB_BETA_FINE_F0",
                ACPL_HCB_BETA_FINE_F0_LEN,
                ACPL_HCB_BETA_FINE_F0_CW,
                9,
                0,
            ),
            (
                "ACPL_HCB_BETA_COARSE_DF",
                ACPL_HCB_BETA_COARSE_DF_LEN,
                ACPL_HCB_BETA_COARSE_DF_CW,
                9,
                4,
            ),
            (
                "ACPL_HCB_BETA_FINE_DF",
                ACPL_HCB_BETA_FINE_DF_LEN,
                ACPL_HCB_BETA_FINE_DF_CW,
                17,
                8,
            ),
            (
                "ACPL_HCB_BETA_COARSE_DT",
                ACPL_HCB_BETA_COARSE_DT_LEN,
                ACPL_HCB_BETA_COARSE_DT_CW,
                9,
                4,
            ),
            (
                "ACPL_HCB_BETA_FINE_DT",
                ACPL_HCB_BETA_FINE_DT_LEN,
                ACPL_HCB_BETA_FINE_DT_CW,
                17,
                8,
            ),
            (
                "ACPL_HCB_BETA3_COARSE_F0",
                ACPL_HCB_BETA3_COARSE_F0_LEN,
                ACPL_HCB_BETA3_COARSE_F0_CW,
                9,
                0,
            ),
            (
                "ACPL_HCB_BETA3_FINE_F0",
                ACPL_HCB_BETA3_FINE_F0_LEN,
                ACPL_HCB_BETA3_FINE_F0_CW,
                17,
                0,
            ),
            (
                "ACPL_HCB_BETA3_COARSE_DF",
                ACPL_HCB_BETA3_COARSE_DF_LEN,
                ACPL_HCB_BETA3_COARSE_DF_CW,
                17,
                8,
            ),
            (
                "ACPL_HCB_BETA3_FINE_DF",
                ACPL_HCB_BETA3_FINE_DF_LEN,
                ACPL_HCB_BETA3_FINE_DF_CW,
                33,
                16,
            ),
            (
                "ACPL_HCB_BETA3_COARSE_DT",
                ACPL_HCB_BETA3_COARSE_DT_LEN,
                ACPL_HCB_BETA3_COARSE_DT_CW,
                17,
                8,
            ),
            (
                "ACPL_HCB_BETA3_FINE_DT",
                ACPL_HCB_BETA3_FINE_DT_LEN,
                ACPL_HCB_BETA3_FINE_DT_CW,
                33,
                16,
            ),
            (
                "ACPL_HCB_GAMMA_COARSE_F0",
                ACPL_HCB_GAMMA_COARSE_F0_LEN,
                ACPL_HCB_GAMMA_COARSE_F0_CW,
                21,
                10,
            ),
            (
                "ACPL_HCB_GAMMA_FINE_F0",
                ACPL_HCB_GAMMA_FINE_F0_LEN,
                ACPL_HCB_GAMMA_FINE_F0_CW,
                41,
                20,
            ),
            (
                "ACPL_HCB_GAMMA_COARSE_DF",
                ACPL_HCB_GAMMA_COARSE_DF_LEN,
                ACPL_HCB_GAMMA_COARSE_DF_CW,
                41,
                20,
            ),
            (
                "ACPL_HCB_GAMMA_FINE_DF",
                ACPL_HCB_GAMMA_FINE_DF_LEN,
                ACPL_HCB_GAMMA_FINE_DF_CW,
                81,
                40,
            ),
            (
                "ACPL_HCB_GAMMA_COARSE_DT",
                ACPL_HCB_GAMMA_COARSE_DT_LEN,
                ACPL_HCB_GAMMA_COARSE_DT_CW,
                41,
                20,
            ),
            (
                "ACPL_HCB_GAMMA_FINE_DT",
                ACPL_HCB_GAMMA_FINE_DT_LEN,
                ACPL_HCB_GAMMA_FINE_DT_CW,
                81,
                40,
            ),
        ]
    }

    /// Each table's len[] / cw[] arrays must agree with the spec table
    /// header's `codebook_length`.
    #[test]
    fn all_acpl_tables_have_declared_length() {
        for (name, lens, cws, n, _) in all_acpl_tables() {
            assert_eq!(
                lens.len(),
                n,
                "{name}: expected len-array of length {n}, got {}",
                lens.len()
            );
            assert_eq!(
                cws.len(),
                n,
                "{name}: expected cw-array of length {n}, got {}",
                cws.len()
            );
        }
    }

    /// Every codeword has to fit inside its declared bit-length, and
    /// the bit length itself has to be in the supported `1..=32` range.
    #[test]
    fn all_acpl_codewords_fit_in_declared_length() {
        for (name, lens, cws, _, _) in all_acpl_tables() {
            for (i, (&l, &c)) in lens.iter().zip(cws.iter()).enumerate() {
                assert!(
                    l > 0 && l <= 32,
                    "{name}[{i}]: declared length {l} is out of supported range"
                );
                let max = if l == 32 { u32::MAX } else { (1u32 << l) - 1 };
                assert!(
                    c <= max,
                    "{name}[{i}]: codeword 0x{c:x} exceeds {l}-bit limit"
                );
            }
        }
    }

    /// Kraft inequality saturation: for a complete binary prefix code
    /// `Σ 2^(-len_i) == 1`. Any failure here is a transcription bug.
    /// We use a 64-bit numerator with a 2^32 denominator so even the
    /// 18-bit GAMMA_FINE_DT codebook has plenty of headroom.
    #[test]
    fn all_acpl_tables_kraft_sum_equals_one() {
        let denom: u128 = 1u128 << 32;
        for (name, lens, _, _, _) in all_acpl_tables() {
            let mut sum_num: u128 = 0;
            for &l in lens {
                sum_num += denom >> l as u128;
            }
            assert_eq!(sum_num, denom, "{name}: Kraft sum != 1");
        }
    }

    /// Pair-by-pair prefix-code property: equal-length codewords must
    /// be distinct, and shorter codewords must not be a prefix of
    /// longer ones. This catches transcription bugs that Kraft alone
    /// can't (Kraft is a necessary but not sufficient condition).
    #[test]
    fn all_acpl_tables_are_prefix_codes() {
        for (name, lens, cws, _, _) in all_acpl_tables() {
            for i in 0..lens.len() {
                let li = lens[i];
                let ci = cws[i];
                for j in (i + 1)..lens.len() {
                    let lj = lens[j];
                    let cj = cws[j];
                    if li == lj {
                        assert_ne!(
                            ci, cj,
                            "{name} collision at {i},{j}: same length {li} cw 0x{ci:x}"
                        );
                    } else {
                        let (short_len, short_cw, long_len, long_cw) = if li < lj {
                            (li, ci, lj, cj)
                        } else {
                            (lj, cj, li, ci)
                        };
                        let prefix = long_cw >> (long_len - short_len);
                        assert_ne!(
                            prefix, short_cw,
                            "{name} prefix conflict at {i},{j}: \
                             {short_len}-bit cw 0x{short_cw:x} prefixes \
                             {long_len}-bit cw 0x{long_cw:x}"
                        );
                    }
                }
            }
        }
    }
}
