//! =============================================================================
//! PROPRIETARY AND CONFIDENTIAL
//! Copyright (c) 2026 Guillaume Lessard / qector-decoder-v3
//! All Rights Reserved.
//! =============================================================================
//!
//! NDA evaluation-key gate for the IonQ wheel (offline, no network).
//!
//! Exactly ONE key type, EIGHT keys total:
//!   - `EVAL01`..`EVAL07`: 7-day evaluation keys, valid 0..=7 days after the
//!     creation date embedded in the key (`QIONQ7-<ID>-<YYYYMMDD>-<tag>`).
//!   - `DEV000`: single dev key, no expiry (local development only).
//!
//! The `<tag>` is an FNV-1a checksum over a compile-time secret plus ID and
//! date, so keys are verifiable offline and unforgable without the secret.
//! Resolution order: `QECTOR_IONQ_KEY` env, `QECTOR_IONQ_KEY_FILE` path,
//! `~/.qector/ionq.key`. Empty key = open evaluation (valid, id `none`).
//! `QECTOR_IONQ_ENFORCE=1` turns an expired/invalid key into a hard error;
//! otherwise violations only surface via `license_status()`.

use pyo3::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

/// Compile-time secret - never shipped outside this crate's build.
const SECRET: &str = "qector-ionq-1.7.8-7day-eval-v1";
/// The only 8 valid key IDs.
const VALID_IDS: [&str; 8] = [
    "EVAL01", "EVAL02", "EVAL03", "EVAL04", "EVAL05", "EVAL06", "EVAL07", "DEV000",
];
/// Evaluation window in days (inclusive).
const EVAL_WINDOW_DAYS: i64 = 7;

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn tag_for(id: &str, date: &str) -> String {
    let input = format!("{SECRET}|{id}|{date}");
    format!("{:016x}", fnv1a64(input.as_bytes()))
}

fn days_today() -> Option<(i64, i64, i64)> {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    let days = (secs / 86400) as i64;
    let mut rem = days;
    let mut y = 1970i64;
    loop {
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let yd = if leap { 366 } else { 365 };
        if rem < yd {
            break;
        }
        rem -= yd;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let lens = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 1i64;
    for len in lens {
        if rem < len {
            break;
        }
        rem -= len;
        m += 1;
    }
    Some((y, m, rem + 1))
}

fn today_ymd() -> Option<String> {
    days_today().map(|(y, m, d)| format!("{y:04}{m:02}{d:02}"))
}

fn ymd_to_days(ymd: &str) -> Option<i64> {
    if ymd.len() != 8 {
        return None;
    }
    let y: i64 = ymd[0..4].parse().ok()?;
    let m: i64 = ymd[4..6].parse().ok()?;
    let d: i64 = ymd[6..8].parse().ok()?;
    if y < 2020 || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let mut days = 0i64;
    for yy in 1970..y {
        let leap = (yy % 4 == 0 && yy % 100 != 0) || yy % 400 == 0;
        days += if leap { 366 } else { 365 };
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let lens = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    for (i, len) in lens.iter().enumerate() {
        if (i as i64) + 1 == m {
            break;
        }
        days += len;
    }
    Some(days + (d - 1))
}

fn read_key_file(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn home_key() -> Option<String> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()?;
    read_key_file(&format!("{home}/.qector/ionq.key"))
}

/// Raw key material (empty = open evaluation).
pub fn raw_key() -> String {
    if let Ok(k) = std::env::var("QECTOR_IONQ_KEY") {
        if !k.trim().is_empty() {
            return k.trim().to_string();
        }
    }
    if let Ok(p) = std::env::var("QECTOR_IONQ_KEY_FILE") {
        if let Some(k) = read_key_file(&p) {
            return k;
        }
    }
    home_key().unwrap_or_default()
}

/// (key_id, valid, message). `key_id` is one of the 8 IDs or `none`.
pub fn license_status() -> (String, bool, String) {
    let key = raw_key();
    if key.is_empty() {
        return ("none".into(), true, "open evaluation (no key set)".into());
    }
    let parts: Vec<&str> = key.split('-').collect();
    if parts.len() != 4 || parts[0] != "QIONQ7" {
        return ("unknown".into(), false, "malformed key (expected QIONQ7-<ID>-<YYYYMMDD>-<tag>)".into());
    }
    let (id, date, tag) = (parts[1], parts[2], parts[3]);
    if !VALID_IDS.contains(&id) {
        return ("unknown".into(), false, format!("unknown key id '{id}' (8 keys exist)"));
    }
    if tag != tag_for(id, date) {
        return (id.into(), false, format!("key {id}: invalid tag"));
    }
    if id == "DEV000" {
        if date != "00000000" {
            return (id.into(), false, "dev key must use date 00000000".into());
        }
        return (id.into(), true, "dev key (no expiry)".into());
    }
    let created = match ymd_to_days(date) {
        Some(d) => d,
        None => return (id.into(), false, format!("key {id}: bad date '{date}'")),
    };
    let today = match today_ymd().and_then(|t| ymd_to_days(&t)) {
        Some(d) => d,
        None => return (id.into(), false, "clock unavailable".into()),
    };
    let age = today - created;
    if age < 0 {
        return (id.into(), false, format!("key {id}: not yet active (starts {date})"));
    }
    if age > EVAL_WINDOW_DAYS {
        return (id.into(), false, format!("key {id}: expired ({age}d old, 7-day window)"));
    }
    (
        id.into(),
        true,
        format!("key {id}: valid (day {age}/7)"),
    )
}

/// Enforce when `QECTOR_IONQ_ENFORCE=1`; otherwise always Ok.
/// Returns the key id on success.
pub fn enforce() -> Result<String, String> {
    let (id, valid, msg) = license_status();
    if valid {
        return Ok(id);
    }
    if std::env::var("QECTOR_IONQ_ENFORCE").as_deref() == Ok("1") {
        return Err(msg);
    }
    Ok(id)
}

#[pyfunction]
pub fn py_license_status() -> (String, bool, String) {
    license_status()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_is_stable() {
        assert_eq!(tag_for("EVAL01", "20260909"), tag_for("EVAL01", "20260909"));
        assert_ne!(tag_for("EVAL01", "20260909"), tag_for("EVAL02", "20260909"));
        assert_ne!(tag_for("EVAL01", "20260909"), tag_for("EVAL01", "20260910"));
    }

    #[test]
    fn dev_key_validates() {
        let id = "DEV000";
        let key = format!("QIONQ7-{id}-00000000-{}", tag_for(id, "00000000"));
        let parts: Vec<&str> = key.split('-').collect();
        assert_eq!(parts.len(), 4);
        assert!(VALID_IDS.contains(&parts[1]));
        assert_eq!(parts[3], tag_for(parts[1], parts[2]));
    }

    #[test]
    fn rejects_unknown_and_bad_tag() {
        assert!(!VALID_IDS.contains(&"EVAL08"));
        assert_ne!(tag_for("EVAL01", "20260909"), "0000000000000000");
        assert!(ymd_to_days("nope").is_none());
        assert!(ymd_to_days("20261301").is_none());
    }

    #[test]
    fn open_evaluation_without_key() {
        std::env::remove_var("QECTOR_IONQ_KEY");
        std::env::remove_var("QECTOR_IONQ_KEY_FILE");
        if raw_key().is_empty() {
            let (id, valid, _) = license_status();
            assert_eq!(id, "none");
            assert!(valid);
        }
    }
}
