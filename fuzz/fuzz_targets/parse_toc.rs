//! `ac4_toc()` walker (TS 103 190-1 §4.2.3 / TS 103 190-2 §6.2.1) on
//! arbitrary bytes, plus the sync-frame unwrapping path.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 1 << 16 {
        return;
    }
    let _ = oxideav_ac4::toc::parse_ac4_toc(data);
    let _ = oxideav_ac4::sync::parse_sync_frame_at_start(data);
    let _ = oxideav_ac4::sync::find_sync_frame(data);
});
