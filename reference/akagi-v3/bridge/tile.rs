//! Majsoul ↔ mjai tile string conversion + mjai canonical ordering.
//!
//! Majsoul strings: `0m`/`0p`/`0s` denote red 5s; `1z`..`7z` are honors.
//! Mjai strings: `5mr`/`5pr`/`5sr` for reds; `E S W N P F C` for honors;
//! `?` is the unknown-tile placeholder used when we don't know what another
//! seat drew/holds.

use anyhow::{bail, Result};
use std::cmp::Ordering;

/// Look up the mjai tile string for a Majsoul tile string. Returns an error
/// for unknown inputs so a malformed liqi payload can't silently corrupt the
/// mjai stream.
pub fn ms_to_mjai(ms: &str) -> Result<&'static str> {
    Ok(match ms {
        "0m" => "5mr",
        "1m" => "1m",
        "2m" => "2m",
        "3m" => "3m",
        "4m" => "4m",
        "5m" => "5m",
        "6m" => "6m",
        "7m" => "7m",
        "8m" => "8m",
        "9m" => "9m",
        "0p" => "5pr",
        "1p" => "1p",
        "2p" => "2p",
        "3p" => "3p",
        "4p" => "4p",
        "5p" => "5p",
        "6p" => "6p",
        "7p" => "7p",
        "8p" => "8p",
        "9p" => "9p",
        "0s" => "5sr",
        "1s" => "1s",
        "2s" => "2s",
        "3s" => "3s",
        "4s" => "4s",
        "5s" => "5s",
        "6s" => "6s",
        "7s" => "7s",
        "8s" => "8s",
        "9s" => "9s",
        "1z" => "E",
        "2z" => "S",
        "3z" => "W",
        "4z" => "N",
        "5z" => "P",
        "6z" => "F",
        "7z" => "C",
        other => bail!("unknown majsoul tile: {other:?}"),
    })
}

/// Index of `pai` in mjai canonical order (smallest first):
/// `1m..4m, 5mr, 5m..9m, 1p..4p, 5pr, 5p..9p, 1s..4s, 5sr, 5s..9s, E S W N P F C, ?`.
/// Unknown strings sort last (one past `?`).
fn pai_rank(pai: &str) -> usize {
    const ORDER: [&str; 38] = [
        "1m", "2m", "3m", "4m", "5mr", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5pr",
        "5p", "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5sr", "5s", "6s", "7s", "8s", "9s",
        "E", "S", "W", "N", "P", "F", "C", "?",
    ];
    ORDER.iter().position(|t| *t == pai).unwrap_or(ORDER.len())
}

/// Comparator over mjai tile strings using the canonical order above.
pub fn compare_pai(a: &str, b: &str) -> Ordering {
    pai_rank(a).cmp(&pai_rank(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn red_fives_round_trip() {
        assert_eq!(ms_to_mjai("0m").unwrap(), "5mr");
        assert_eq!(ms_to_mjai("0p").unwrap(), "5pr");
        assert_eq!(ms_to_mjai("0s").unwrap(), "5sr");
    }

    #[test]
    fn honors_map_to_letters() {
        assert_eq!(ms_to_mjai("1z").unwrap(), "E");
        assert_eq!(ms_to_mjai("7z").unwrap(), "C");
    }

    #[test]
    fn unknown_tile_errors() {
        assert!(ms_to_mjai("8z").is_err());
        assert!(ms_to_mjai("garbage").is_err());
    }

    #[test]
    fn red_five_sorts_before_normal_five() {
        assert_eq!(compare_pai("5mr", "5m"), Ordering::Less);
        assert_eq!(compare_pai("5pr", "5p"), Ordering::Less);
    }

    #[test]
    fn full_sort_matches_canonical_order() {
        let mut tiles: Vec<String> = vec!["C", "1m", "5m", "5mr", "9p", "E", "?", "3s"]
            .into_iter()
            .map(String::from)
            .collect();
        tiles.sort_by(|a, b| compare_pai(a, b));
        assert_eq!(
            tiles,
            ["1m", "5mr", "5m", "9p", "3s", "E", "C", "?"].map(String::from),
        );
    }
}
