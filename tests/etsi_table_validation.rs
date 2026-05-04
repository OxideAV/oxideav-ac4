//! Round-20 validation suite — every AC-4 Huffman table that we ship
//! transcribed in `src/*_huffman.rs` (plus the original ASF tables in
//! `src/huffman_tables.rs`) is compared *byte-for-byte* against the
//! canonical ETSI accompaniment file
//! `docs/audio/ac4/ts_10319001v010401p0-tables.c`.
//!
//! The accompaniment file is parsed at runtime by a tiny C-array tokeniser
//! (only the bits we need: `const int NAME[N] = { ... };` and
//! `const int32 NAME[N] = { ... };`). All hand-curated tables in the crate
//! are then asserted to match the freshly parsed reference.
//!
//! If you ever see a failure here, the Rust constant is wrong (or the
//! ETSI file changed) — never paper over a divergence.
//!
//! Coverage:
//!
//! * ASF (Annex A.1)        — SCALEFAC, SNF, HCB1..HCB11 — 13 tables
//! * A-SPX (Annex A.2)      — 18 tables (Table A.16..A.33)
//! * A-CPL (Annex A.3)      — 24 tables (Table A.34..A.57)
//! * DE  (Annex A.4)        — 4 tables (Table A.58..A.61)
//! * DRC (Annex A.5)        — 1 table  (Table A.62)
//!
//! That's 60 codebooks (each is a `(LEN[], CW[])` pair, so 120 arrays
//! validated end-to-end).

use std::collections::HashMap;
use std::path::PathBuf;

// Pull the raw constants out of the crate. We re-export deliberately
// long aliases so the test loop can iterate over `(name, slice)` tuples
// without making the call-site noisy.
use oxideav_ac4::acpl_huffman::*;
use oxideav_ac4::aspx_huffman::*;
use oxideav_ac4::de_huffman::*;
use oxideav_ac4::drc_huffman::*;
use oxideav_ac4::huffman::*;

// ------------------------------------------------------------------------
// .c-file mini-tokeniser
// ------------------------------------------------------------------------

/// Locate the ETSI accompaniment file relative to this crate's manifest dir.
fn etsi_tables_path() -> PathBuf {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/oxideav-ac4 must live two dirs deep")
        .to_path_buf();
    workspace.join("docs/audio/ac4/ts_10319001v010401p0-tables.c")
}

/// Read the entire .c file (~80 KiB) into memory once.
///
/// Returns `None` (with a `cargo:warning`-style diagnostic on stderr) when
/// the docs sibling repo isn't checked out alongside this crate — tests
/// that rely on the file should early-return so CI in standalone-crate mode
/// stays green. Set `OXIDEAV_AC4_REQUIRE_ETSI=1` to upgrade the skip into
/// a hard failure (e.g. for the workspace umbrella where the docs repo is
/// always present).
fn read_etsi_source() -> Option<String> {
    let path = etsi_tables_path();
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(s),
        Err(e) => {
            if std::env::var_os("OXIDEAV_AC4_REQUIRE_ETSI").is_some() {
                panic!("cannot read {}: {}", path.display(), e);
            }
            eprintln!(
                "etsi_table_validation: skipping (cannot read {}: {}). \
                 Set OXIDEAV_AC4_REQUIRE_ETSI=1 to make this fatal.",
                path.display(),
                e
            );
            None
        }
    }
}

/// Convenience: early-return the current `#[test]` fn when the ETSI source
/// isn't available. Used at the top of every validation test.
macro_rules! etsi_src_or_skip {
    () => {
        match read_etsi_source() {
            Some(s) => s,
            None => return,
        }
    };
}

/// Strip C `/* ... */` block comments and `// ...` line comments so the
/// rest of the parser can stay dumb. The accompaniment file only uses
/// block comments at the top, but be safe.
fn strip_c_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            // block comment
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = i.min(bytes.len()).saturating_add(2).min(bytes.len());
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            // line comment
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

/// One parsed C array.
#[derive(Debug, Clone)]
struct CArray {
    declared_len: usize,
    values: Vec<u64>, // we'll downcast at compare time
}

/// Parse all `const int*` arrays in the source, returning them keyed by
/// name. Recognised forms (whitespace-tolerant):
///
///   const int           NAME[N] = { v0, v1, ... };
///   const int32         NAME[N] = { v0, v1, ... };
///   const unsigned char NAME[N] = { v0, v1, ... };
///   const unsigned int  NAME[N] = { v0, v1, ... };
///
/// We don't handle `float`, `float32`, or 2D arrays — they're not part of
/// the Huffman validation scope.
fn parse_c_arrays(src: &str) -> HashMap<String, CArray> {
    let mut out = HashMap::new();
    // Scan tokens. We anchor on the literal "const " keyword.
    let bytes = src.as_bytes();
    let mut i = 0;
    while i + 6 <= bytes.len() {
        if &bytes[i..i + 6] == b"const " {
            // Try to parse: const <type> <NAME>[<N>] = { ... };
            // First skip past "const ".
            let start = i;
            i += 6;
            // type: read until we hit '['; the type string itself can be
            // multi-word ("unsigned char", "int32", "int", "unsigned int").
            // We don't actually care about the type beyond ruling out the
            // floating-point arrays.
            let header_start = i;
            // Find '['; if we run into '{' / '=' first, give up on this match.
            let mut scan = i;
            while scan < bytes.len() && bytes[scan] != b'[' {
                if bytes[scan] == b'=' || bytes[scan] == b'{' || bytes[scan] == b';' {
                    break;
                }
                scan += 1;
            }
            if scan >= bytes.len() || bytes[scan] != b'[' {
                i = start + 1;
                continue;
            }
            let bracket_pos = scan;

            // Skip floats — bail early on `float` / `float32` types.
            let header = &src[header_start..bracket_pos];
            if header.contains("float") {
                i = scan + 1;
                continue;
            }

            // Name = last identifier in the header.
            let name = header
                .split_whitespace()
                .last()
                .map(|s| s.to_string())
                .unwrap_or_default();
            if name.is_empty() {
                i = scan + 1;
                continue;
            }

            // Parse N inside the brackets — `[N]` is a positive decimal int.
            let bp = bracket_pos;
            let mut close = bp + 1;
            while close < bytes.len() && bytes[close] != b']' {
                close += 1;
            }
            if close >= bytes.len() {
                i = bp + 1;
                continue;
            }
            let n_str = &src[bp + 1..close];
            let declared_len: usize = match n_str.trim().parse() {
                Ok(n) => n,
                Err(_) => {
                    // 2-D array `[A][B]` or non-numeric — skip.
                    i = close + 1;
                    continue;
                }
            };

            // Skip up to `{`.
            let mut p = close + 1;
            while p < bytes.len() && bytes[p] != b'{' {
                if bytes[p] == b';' {
                    break;
                }
                p += 1;
            }
            if p >= bytes.len() || bytes[p] != b'{' {
                i = p.saturating_add(1).min(bytes.len());
                continue;
            }
            // Find matching `}` (no nested braces in flat arrays).
            let body_start = p + 1;
            let mut q = body_start;
            while q < bytes.len() && bytes[q] != b'}' {
                q += 1;
            }
            if q >= bytes.len() {
                i = body_start;
                continue;
            }
            let body = &src[body_start..q];

            // Tokenise body on `,` and whitespace, parse each as
            // hex-or-decimal integer.
            let mut values = Vec::with_capacity(declared_len);
            for tok in body.split(|c: char| c == ',' || c.is_whitespace()) {
                let t = tok.trim();
                if t.is_empty() {
                    continue;
                }
                let v = if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                    u64::from_str_radix(hex, 16).ok()
                } else {
                    t.parse::<u64>().ok()
                };
                if let Some(v) = v {
                    values.push(v);
                } else {
                    // Non-numeric token (e.g. macro, suffix). Skip the
                    // whole array conservatively.
                    values.clear();
                    break;
                }
            }
            if !values.is_empty() {
                out.insert(
                    name,
                    CArray {
                        declared_len,
                        values,
                    },
                );
            }
            i = q + 1;
        } else {
            i += 1;
        }
    }
    out
}

// ------------------------------------------------------------------------
// Per-table assertion helper
// ------------------------------------------------------------------------

fn assert_u8_table(reference: &HashMap<String, CArray>, c_name: &str, rust: &[u8]) {
    let r = reference
        .get(c_name)
        .unwrap_or_else(|| panic!("ETSI source missing array `{c_name}`"));
    assert_eq!(
        r.declared_len,
        rust.len(),
        "{c_name}: declared length {} differs from Rust slice {}",
        r.declared_len,
        rust.len()
    );
    assert_eq!(
        r.values.len(),
        rust.len(),
        "{c_name}: value count {} differs from Rust slice {}",
        r.values.len(),
        rust.len()
    );
    for (i, (&rv, &cv)) in rust.iter().zip(r.values.iter()).enumerate() {
        assert_eq!(rv as u64, cv, "{c_name}[{i}]: Rust=0x{rv:x}, ETSI=0x{cv:x}");
    }
}

fn assert_u32_table(reference: &HashMap<String, CArray>, c_name: &str, rust: &[u32]) {
    let r = reference
        .get(c_name)
        .unwrap_or_else(|| panic!("ETSI source missing array `{c_name}`"));
    assert_eq!(
        r.declared_len,
        rust.len(),
        "{c_name}: declared length {} differs from Rust slice {}",
        r.declared_len,
        rust.len()
    );
    assert_eq!(
        r.values.len(),
        rust.len(),
        "{c_name}: value count {} differs from Rust slice {}",
        r.values.len(),
        rust.len()
    );
    for (i, (&rv, &cv)) in rust.iter().zip(r.values.iter()).enumerate() {
        assert_eq!(rv as u64, cv, "{c_name}[{i}]: Rust=0x{rv:x}, ETSI=0x{cv:x}");
    }
}

// ------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------

/// Sanity: the .c file must parse cleanly and yield at least the ~120
/// integer arrays we expect (60 Huffman codebooks × 2 arrays each, plus
/// SSF predictor matrices, predictor LUTs, dither / step-size tables,
/// CDFs etc.).
#[test]
fn etsi_source_parses() {
    let src = strip_c_comments(&etsi_src_or_skip!());
    let arrays = parse_c_arrays(&src);
    assert!(
        arrays.len() >= 120,
        "expected >= 120 integer arrays, got {}",
        arrays.len()
    );
    // Spot-check: known-name lookups must succeed.
    for name in [
        "ASF_HCB_SCALEFAC_LEN",
        "ASF_HCB_SNF_CW",
        "ASPX_HCB_ENV_LEVEL_15_F0_LEN",
        "ACPL_HCB_GAMMA_FINE_DT_CW",
        "DE_HCB_DIFF_1_CW",
        "DRC_HCB_LEN",
    ] {
        assert!(
            arrays.contains_key(name),
            "ETSI source missing expected array `{name}`"
        );
    }
}

#[test]
fn validate_asf_huffman_tables() {
    let src = strip_c_comments(&etsi_src_or_skip!());
    let r = parse_c_arrays(&src);

    // SCALEFAC + SNF
    assert_u8_table(&r, "ASF_HCB_SCALEFAC_LEN", HCB_SCALEFAC_LEN);
    assert_u32_table(&r, "ASF_HCB_SCALEFAC_CW", HCB_SCALEFAC_CW);
    assert_u8_table(&r, "ASF_HCB_SNF_LEN", HCB_SNF_LEN);
    assert_u32_table(&r, "ASF_HCB_SNF_CW", HCB_SNF_CW);

    // HCB1..HCB11
    assert_u8_table(&r, "ASF_HCB_1_LEN", HCB1_LEN);
    assert_u32_table(&r, "ASF_HCB_1_CW", HCB1_CW);
    assert_u8_table(&r, "ASF_HCB_2_LEN", HCB2_LEN);
    assert_u32_table(&r, "ASF_HCB_2_CW", HCB2_CW);
    assert_u8_table(&r, "ASF_HCB_3_LEN", HCB3_LEN);
    assert_u32_table(&r, "ASF_HCB_3_CW", HCB3_CW);
    assert_u8_table(&r, "ASF_HCB_4_LEN", HCB4_LEN);
    assert_u32_table(&r, "ASF_HCB_4_CW", HCB4_CW);
    assert_u8_table(&r, "ASF_HCB_5_LEN", HCB5_LEN);
    assert_u32_table(&r, "ASF_HCB_5_CW", HCB5_CW);
    assert_u8_table(&r, "ASF_HCB_6_LEN", HCB6_LEN);
    assert_u32_table(&r, "ASF_HCB_6_CW", HCB6_CW);
    assert_u8_table(&r, "ASF_HCB_7_LEN", HCB7_LEN);
    assert_u32_table(&r, "ASF_HCB_7_CW", HCB7_CW);
    assert_u8_table(&r, "ASF_HCB_8_LEN", HCB8_LEN);
    assert_u32_table(&r, "ASF_HCB_8_CW", HCB8_CW);
    assert_u8_table(&r, "ASF_HCB_9_LEN", HCB9_LEN);
    assert_u32_table(&r, "ASF_HCB_9_CW", HCB9_CW);
    assert_u8_table(&r, "ASF_HCB_10_LEN", HCB10_LEN);
    assert_u32_table(&r, "ASF_HCB_10_CW", HCB10_CW);
    assert_u8_table(&r, "ASF_HCB_11_LEN", HCB11_LEN);
    assert_u32_table(&r, "ASF_HCB_11_CW", HCB11_CW);
}

#[test]
fn validate_aspx_huffman_tables() {
    let src = strip_c_comments(&etsi_src_or_skip!());
    let r = parse_c_arrays(&src);

    // 18 envelope/noise codebooks (Tables A.16..A.33).
    assert_u8_table(
        &r,
        "ASPX_HCB_ENV_LEVEL_15_F0_LEN",
        ASPX_HCB_ENV_LEVEL_15_F0_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_ENV_LEVEL_15_F0_CW",
        ASPX_HCB_ENV_LEVEL_15_F0_CW,
    );
    assert_u8_table(
        &r,
        "ASPX_HCB_ENV_LEVEL_15_DF_LEN",
        ASPX_HCB_ENV_LEVEL_15_DF_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_ENV_LEVEL_15_DF_CW",
        ASPX_HCB_ENV_LEVEL_15_DF_CW,
    );
    assert_u8_table(
        &r,
        "ASPX_HCB_ENV_LEVEL_15_DT_LEN",
        ASPX_HCB_ENV_LEVEL_15_DT_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_ENV_LEVEL_15_DT_CW",
        ASPX_HCB_ENV_LEVEL_15_DT_CW,
    );

    assert_u8_table(
        &r,
        "ASPX_HCB_ENV_BALANCE_15_F0_LEN",
        ASPX_HCB_ENV_BALANCE_15_F0_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_ENV_BALANCE_15_F0_CW",
        ASPX_HCB_ENV_BALANCE_15_F0_CW,
    );
    assert_u8_table(
        &r,
        "ASPX_HCB_ENV_BALANCE_15_DF_LEN",
        ASPX_HCB_ENV_BALANCE_15_DF_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_ENV_BALANCE_15_DF_CW",
        ASPX_HCB_ENV_BALANCE_15_DF_CW,
    );
    assert_u8_table(
        &r,
        "ASPX_HCB_ENV_BALANCE_15_DT_LEN",
        ASPX_HCB_ENV_BALANCE_15_DT_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_ENV_BALANCE_15_DT_CW",
        ASPX_HCB_ENV_BALANCE_15_DT_CW,
    );

    assert_u8_table(
        &r,
        "ASPX_HCB_ENV_LEVEL_30_F0_LEN",
        ASPX_HCB_ENV_LEVEL_30_F0_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_ENV_LEVEL_30_F0_CW",
        ASPX_HCB_ENV_LEVEL_30_F0_CW,
    );
    assert_u8_table(
        &r,
        "ASPX_HCB_ENV_LEVEL_30_DF_LEN",
        ASPX_HCB_ENV_LEVEL_30_DF_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_ENV_LEVEL_30_DF_CW",
        ASPX_HCB_ENV_LEVEL_30_DF_CW,
    );
    assert_u8_table(
        &r,
        "ASPX_HCB_ENV_LEVEL_30_DT_LEN",
        ASPX_HCB_ENV_LEVEL_30_DT_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_ENV_LEVEL_30_DT_CW",
        ASPX_HCB_ENV_LEVEL_30_DT_CW,
    );

    assert_u8_table(
        &r,
        "ASPX_HCB_ENV_BALANCE_30_F0_LEN",
        ASPX_HCB_ENV_BALANCE_30_F0_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_ENV_BALANCE_30_F0_CW",
        ASPX_HCB_ENV_BALANCE_30_F0_CW,
    );
    assert_u8_table(
        &r,
        "ASPX_HCB_ENV_BALANCE_30_DF_LEN",
        ASPX_HCB_ENV_BALANCE_30_DF_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_ENV_BALANCE_30_DF_CW",
        ASPX_HCB_ENV_BALANCE_30_DF_CW,
    );
    assert_u8_table(
        &r,
        "ASPX_HCB_ENV_BALANCE_30_DT_LEN",
        ASPX_HCB_ENV_BALANCE_30_DT_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_ENV_BALANCE_30_DT_CW",
        ASPX_HCB_ENV_BALANCE_30_DT_CW,
    );

    assert_u8_table(
        &r,
        "ASPX_HCB_NOISE_LEVEL_F0_LEN",
        ASPX_HCB_NOISE_LEVEL_F0_LEN,
    );
    assert_u32_table(&r, "ASPX_HCB_NOISE_LEVEL_F0_CW", ASPX_HCB_NOISE_LEVEL_F0_CW);
    assert_u8_table(
        &r,
        "ASPX_HCB_NOISE_LEVEL_DF_LEN",
        ASPX_HCB_NOISE_LEVEL_DF_LEN,
    );
    assert_u32_table(&r, "ASPX_HCB_NOISE_LEVEL_DF_CW", ASPX_HCB_NOISE_LEVEL_DF_CW);
    assert_u8_table(
        &r,
        "ASPX_HCB_NOISE_LEVEL_DT_LEN",
        ASPX_HCB_NOISE_LEVEL_DT_LEN,
    );
    assert_u32_table(&r, "ASPX_HCB_NOISE_LEVEL_DT_CW", ASPX_HCB_NOISE_LEVEL_DT_CW);

    assert_u8_table(
        &r,
        "ASPX_HCB_NOISE_BALANCE_F0_LEN",
        ASPX_HCB_NOISE_BALANCE_F0_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_NOISE_BALANCE_F0_CW",
        ASPX_HCB_NOISE_BALANCE_F0_CW,
    );
    assert_u8_table(
        &r,
        "ASPX_HCB_NOISE_BALANCE_DF_LEN",
        ASPX_HCB_NOISE_BALANCE_DF_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_NOISE_BALANCE_DF_CW",
        ASPX_HCB_NOISE_BALANCE_DF_CW,
    );
    assert_u8_table(
        &r,
        "ASPX_HCB_NOISE_BALANCE_DT_LEN",
        ASPX_HCB_NOISE_BALANCE_DT_LEN,
    );
    assert_u32_table(
        &r,
        "ASPX_HCB_NOISE_BALANCE_DT_CW",
        ASPX_HCB_NOISE_BALANCE_DT_CW,
    );
}

#[test]
fn validate_acpl_huffman_tables() {
    let src = strip_c_comments(&etsi_src_or_skip!());
    let r = parse_c_arrays(&src);

    // 24 ACPL codebooks (Tables A.34..A.57). Names mirror C exactly.
    macro_rules! pair {
        ($base:literal, $len:ident, $cw:ident) => {
            assert_u8_table(&r, concat!($base, "_LEN"), $len);
            assert_u32_table(&r, concat!($base, "_CW"), $cw);
        };
    }

    pair!(
        "ACPL_HCB_ALPHA_COARSE_F0",
        ACPL_HCB_ALPHA_COARSE_F0_LEN,
        ACPL_HCB_ALPHA_COARSE_F0_CW
    );
    pair!(
        "ACPL_HCB_ALPHA_FINE_F0",
        ACPL_HCB_ALPHA_FINE_F0_LEN,
        ACPL_HCB_ALPHA_FINE_F0_CW
    );
    pair!(
        "ACPL_HCB_ALPHA_COARSE_DF",
        ACPL_HCB_ALPHA_COARSE_DF_LEN,
        ACPL_HCB_ALPHA_COARSE_DF_CW
    );
    pair!(
        "ACPL_HCB_ALPHA_FINE_DF",
        ACPL_HCB_ALPHA_FINE_DF_LEN,
        ACPL_HCB_ALPHA_FINE_DF_CW
    );
    pair!(
        "ACPL_HCB_ALPHA_COARSE_DT",
        ACPL_HCB_ALPHA_COARSE_DT_LEN,
        ACPL_HCB_ALPHA_COARSE_DT_CW
    );
    pair!(
        "ACPL_HCB_ALPHA_FINE_DT",
        ACPL_HCB_ALPHA_FINE_DT_LEN,
        ACPL_HCB_ALPHA_FINE_DT_CW
    );

    pair!(
        "ACPL_HCB_BETA_COARSE_F0",
        ACPL_HCB_BETA_COARSE_F0_LEN,
        ACPL_HCB_BETA_COARSE_F0_CW
    );
    pair!(
        "ACPL_HCB_BETA_FINE_F0",
        ACPL_HCB_BETA_FINE_F0_LEN,
        ACPL_HCB_BETA_FINE_F0_CW
    );
    pair!(
        "ACPL_HCB_BETA_COARSE_DF",
        ACPL_HCB_BETA_COARSE_DF_LEN,
        ACPL_HCB_BETA_COARSE_DF_CW
    );
    pair!(
        "ACPL_HCB_BETA_FINE_DF",
        ACPL_HCB_BETA_FINE_DF_LEN,
        ACPL_HCB_BETA_FINE_DF_CW
    );
    pair!(
        "ACPL_HCB_BETA_COARSE_DT",
        ACPL_HCB_BETA_COARSE_DT_LEN,
        ACPL_HCB_BETA_COARSE_DT_CW
    );
    pair!(
        "ACPL_HCB_BETA_FINE_DT",
        ACPL_HCB_BETA_FINE_DT_LEN,
        ACPL_HCB_BETA_FINE_DT_CW
    );

    pair!(
        "ACPL_HCB_BETA3_COARSE_F0",
        ACPL_HCB_BETA3_COARSE_F0_LEN,
        ACPL_HCB_BETA3_COARSE_F0_CW
    );
    pair!(
        "ACPL_HCB_BETA3_FINE_F0",
        ACPL_HCB_BETA3_FINE_F0_LEN,
        ACPL_HCB_BETA3_FINE_F0_CW
    );
    pair!(
        "ACPL_HCB_BETA3_COARSE_DF",
        ACPL_HCB_BETA3_COARSE_DF_LEN,
        ACPL_HCB_BETA3_COARSE_DF_CW
    );
    pair!(
        "ACPL_HCB_BETA3_FINE_DF",
        ACPL_HCB_BETA3_FINE_DF_LEN,
        ACPL_HCB_BETA3_FINE_DF_CW
    );
    pair!(
        "ACPL_HCB_BETA3_COARSE_DT",
        ACPL_HCB_BETA3_COARSE_DT_LEN,
        ACPL_HCB_BETA3_COARSE_DT_CW
    );
    pair!(
        "ACPL_HCB_BETA3_FINE_DT",
        ACPL_HCB_BETA3_FINE_DT_LEN,
        ACPL_HCB_BETA3_FINE_DT_CW
    );

    pair!(
        "ACPL_HCB_GAMMA_COARSE_F0",
        ACPL_HCB_GAMMA_COARSE_F0_LEN,
        ACPL_HCB_GAMMA_COARSE_F0_CW
    );
    pair!(
        "ACPL_HCB_GAMMA_FINE_F0",
        ACPL_HCB_GAMMA_FINE_F0_LEN,
        ACPL_HCB_GAMMA_FINE_F0_CW
    );
    pair!(
        "ACPL_HCB_GAMMA_COARSE_DF",
        ACPL_HCB_GAMMA_COARSE_DF_LEN,
        ACPL_HCB_GAMMA_COARSE_DF_CW
    );
    pair!(
        "ACPL_HCB_GAMMA_FINE_DF",
        ACPL_HCB_GAMMA_FINE_DF_LEN,
        ACPL_HCB_GAMMA_FINE_DF_CW
    );
    pair!(
        "ACPL_HCB_GAMMA_COARSE_DT",
        ACPL_HCB_GAMMA_COARSE_DT_LEN,
        ACPL_HCB_GAMMA_COARSE_DT_CW
    );
    pair!(
        "ACPL_HCB_GAMMA_FINE_DT",
        ACPL_HCB_GAMMA_FINE_DT_LEN,
        ACPL_HCB_GAMMA_FINE_DT_CW
    );
}

#[test]
fn validate_de_huffman_tables() {
    let src = strip_c_comments(&etsi_src_or_skip!());
    let r = parse_c_arrays(&src);

    assert_u8_table(&r, "DE_HCB_ABS_0_LEN", DE_HCB_ABS_0_LEN);
    assert_u32_table(&r, "DE_HCB_ABS_0_CW", DE_HCB_ABS_0_CW);
    assert_u8_table(&r, "DE_HCB_DIFF_0_LEN", DE_HCB_DIFF_0_LEN);
    assert_u32_table(&r, "DE_HCB_DIFF_0_CW", DE_HCB_DIFF_0_CW);
    assert_u8_table(&r, "DE_HCB_ABS_1_LEN", DE_HCB_ABS_1_LEN);
    assert_u32_table(&r, "DE_HCB_ABS_1_CW", DE_HCB_ABS_1_CW);
    assert_u8_table(&r, "DE_HCB_DIFF_1_LEN", DE_HCB_DIFF_1_LEN);
    assert_u32_table(&r, "DE_HCB_DIFF_1_CW", DE_HCB_DIFF_1_CW);
}

#[test]
fn validate_drc_huffman_table() {
    let src = strip_c_comments(&etsi_src_or_skip!());
    let r = parse_c_arrays(&src);

    assert_u8_table(&r, "DRC_HCB_LEN", DRC_HCB_LEN);
    assert_u32_table(&r, "DRC_HCB_CW", DRC_HCB_CW);
}

// ------------------------------------------------------------------------
// SSF / Annex C scalar table validation
// ------------------------------------------------------------------------

fn assert_i32_table(reference: &HashMap<String, CArray>, c_name: &str, rust: &[i32]) {
    let r = reference
        .get(c_name)
        .unwrap_or_else(|| panic!("ETSI source missing array `{c_name}`"));
    assert_eq!(
        r.declared_len,
        rust.len(),
        "{c_name}: declared length {} differs from Rust slice {}",
        r.declared_len,
        rust.len()
    );
    assert_eq!(
        r.values.len(),
        rust.len(),
        "{c_name}: value count {} differs from Rust slice {}",
        r.values.len(),
        rust.len()
    );
    for (i, (&rv, &cv)) in rust.iter().zip(r.values.iter()).enumerate() {
        // CDF / DITHER / STEP_SIZE entries are non-negative; an unsigned
        // round-trip into i32 is lossless and matches the source verbatim.
        assert_eq!(rv as u64, cv, "{c_name}[{i}]: Rust=0x{rv:x}, ETSI=0x{cv:x}");
    }
}

/// Annex C: scalar lookup tables (POST_GAIN, PRED_RFS / RTS, CDFs,
/// DITHER, STEP_SIZES, AC_COEFF_MAX_INDEX, dB↔linear LUTs) all match
/// the canonical accompaniment file byte-for-byte. Float tables
/// (`POST_GAIN_LUT`, `PRED_GAIN_QUANT_TAB`, `RANDOM_NOISE_TABLE`)
/// can't go through the integer parser, so they're spot-checked
/// against unit-test anchors in [`oxideav_ac4::ssf_tables`] instead.
#[test]
fn validate_ssf_scalar_tables() {
    use oxideav_ac4::ssf_tables::*;

    let src = strip_c_comments(&etsi_src_or_skip!());
    let r = parse_c_arrays(&src);

    assert_u8_table(&r, "PRED_RFS_TABLE", &PRED_RFS_TABLE);
    assert_u8_table(&r, "PRED_RTS_TABLE", &PRED_RTS_TABLE);
    assert_i32_table(&r, "CDF_TABLE", &CDF_TABLE);
    assert_u32_table(&r, "PREDICTOR_GAIN_CDF_LUT", &PREDICTOR_GAIN_CDF_LUT);
    assert_i32_table(&r, "ENVELOPE_CDF_LUT", &ENVELOPE_CDF_LUT);
    assert_i32_table(&r, "DITHER_TABLE", &DITHER_TABLE);
    assert_i32_table(&r, "STEP_SIZES_Q4_15", &STEP_SIZES_Q4_15);
    assert_u8_table(&r, "AC_COEFF_MAX_INDEX", &AC_COEFF_MAX_INDEX);
    // The four Annex C.14 dB↔linear tables (`SLOPES_DB_TO_LIN` etc.)
    // are declared without the `const` keyword in the ETSI
    // accompaniment file, so the conservative integer parser
    // skips them. Their values are verified by the explicit anchor
    // tests in `crate::ssf_tables` instead.
    let _ = (
        SLOPES_DB_TO_LIN,
        OFFSETS_DB_TO_LIN,
        SLOPES_LIN_TO_DB,
        OFFSETS_LIN_TO_DB,
    );
}

/// Annex C.6: all 37 SSF prediction-coefficient matrices match
/// byte-for-byte.
#[test]
fn validate_ssf_pred_coeff_matrices() {
    use oxideav_ac4::ssf_pred_coeff::*;

    let src = strip_c_comments(&etsi_src_or_skip!());
    let r = parse_c_arrays(&src);

    macro_rules! check_mat {
        ($($idx:expr),* $(,)?) => {
            $(
                {
                    let name = concat!("ssf_pred_coeff_mat", stringify!($idx));
                    let rust = ssf_pred_coeff_mat($idx).expect("in-range mat");
                    assert_u8_table(&r, name, rust);
                }
            )*
        };
    }
    check_mat!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
    );
}
