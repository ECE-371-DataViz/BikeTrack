#!/usr/bin/env python3
"""
Map LED index labels CSV to CitiBike station ID using GBFS station_information JSON.

Usage:
    python raspi_code/map_leds_to_station_ids.py \
      --in FinalLEDTable.csv \
      --out FinalLEDTable_with_ids.csv \
      [--url <station_information_url>] [--fuzzy-threshold 0.8]

Output CSV will contain: index,station_id,matched_name,score

The script tries an exact / normalized match first, then a fuzzy match.
"""

import argparse
import csv
import json
import re
import sys
from typing import Dict, Tuple, List

try:
    import requests
except Exception:
    requests = None

from difflib import get_close_matches

GBFS_URL_DEFAULT = "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_information.json"


def norm_name(s: str) -> str:
    s = s.strip().lower()
    # unify ampersand and 'and'
    s = s.replace("&", " and ")
    # remove excessive punctuation, keep commas
    s = re.sub(r"[^a-z0-9,\s]", "", s)
    # normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def build_name_to_id_map(stations: List[Dict]) -> Dict[str, str]:
    """Return mapping from normalized station name to station_id.
       If duplicate normalized names appear, later ones override earlier.
    """
    return {norm_name(s["name"]): s["station_id"] for s in stations}


def fetch_station_info(url: str) -> List[Dict]:
    if requests is None:
        raise RuntimeError("requests package is required to fetch GBFS data; install it with pip install requests")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    j = r.json()
    stations = j.get("data", {}).get("stations", [])
    return stations


def map_labels_to_ids(labels_csv: str, url: str, fuzzy_threshold: float = 0.8) -> Tuple[List[Tuple[str,str,str,float]], List[Tuple[str,str]]]:
    """Read labels CSV of form index,label and return (mapped rows, not_found_rows).

    mapped rows: [(index, station_id, matched_name, score)]
    not_found: [(index, label)]
    """
    stations = fetch_station_info(url)
    name_map = build_name_to_id_map(stations)
    station_names = list(name_map.keys())

    mapped = []
    not_found = []

    with open(labels_csv, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            index = row[0].strip()
            # label might be in second column or combined
            if len(row) > 1:
                label = row[1].strip()
            else:
                # If label hasn't been split, try to parse as everything after index
                label = ''.join(row[1:]).strip() if len(row) > 1 else ''
            if not label:
                not_found.append((index, label))
                continue

            normalized = norm_name(label)
            # Exact normalized match
            if normalized in name_map:
                mapped.append((index, name_map[normalized], station_names[station_names.index(normalized)], 1.0))
                continue

            # Try fuzzy: we get the closest candidate by name (difflib)
            # Note: difflib returns by similarity; set cut-off to fuzzy_threshold
            close = get_close_matches(normalized, station_names, n=1, cutoff=fuzzy_threshold)
            if close:
                matched_name = close[0]
                mapped.append((index, name_map[matched_name], matched_name, 1.0))
            else:
                # try more relaxed matching: try looking for partial overlaps
                candidates = [n for n in station_names if normalized in n or any(token in n for token in normalized.split() if len(token) > 2)]
                if candidates:
                    # choose best candidate by ratio of common tokens
                    best = candidates[0]
                    mapped.append((index, name_map[best], best, 0.5))
                else:
                    not_found.append((index, label))
    return mapped, not_found


def write_output(out_csv: str, mapped: List[Tuple[str,str,str,float]], not_found: List[Tuple[str,str]]):
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'station_id', 'matched_name', 'score'])
        for index, station_id, matched_name, score in mapped:
            writer.writerow([index, station_id, matched_name, score])
        if not_found:
            writer.writerow([])
            writer.writerow(['unmatched_index', 'original_label'])
            for index, label in not_found:
                writer.writerow([index, label])


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_csv', required=True, help='Input LED CSV: index,label')
    p.add_argument('--out', dest='out_csv', required=True, help='Output CSV: index,station_id,matched_name,score')
    p.add_argument('--url', default=GBFS_URL_DEFAULT, help='GBFS station_information URL')
    p.add_argument('--fuzzy-threshold', type=float, default=0.86, help='Difflib cutoff threshold 0..1')
    args = p.parse_args()

    print(f'Fetching station info from {args.url} ...')
    mapped, not_found = map_labels_to_ids(args.in_csv, args.url, args.fuzzy_threshold)
    print(f'Found {len(mapped)} matches, {len(not_found)} unmatched')

    print(f'Writing output CSV to {args.out_csv} ...')
    write_output(args.out_csv, mapped, not_found)
    print('Done')


if __name__ == '__main__':
    cli()
