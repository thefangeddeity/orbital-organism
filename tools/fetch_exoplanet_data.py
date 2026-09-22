#!/usr/bin/env python3
"""
Fetches real, current orbital parameters from the NASA Exoplanet
Archive and caches them to state/exoplanet_cache.json. This is the
"curiosity queue" real_systems.py's own docstring anticipated: a
bounded, logged, explicit fetch -- never called from inside the
render/boot path (same reasoning as dispatch_tanzania.py living outside
organism.py's real-time loop: a network round-trip has no place in a
frame budget). real_systems.py only ever READS the cache this writes;
it never reaches the network itself.

Run manually, or schedule it (cron / Task Scheduler) for periodic
refresh:

    python3 fetch_exoplanet_data.py

Queries the `pscomppars` table (composite/default parameters -- one
clean row per planet), not the raw `ps` table (one row per publication,
would need its own dedupe/preference logic). Confirmed working query
shape against the real API:

    select pl_name,hostname,pl_orbsmax,pl_orbper,pl_orbeccen,pl_bmasse,
           pl_rade,pl_orbincl
    from pscomppars where hostname like 'Proxima%'

Alpha Centauri's A/B stellar binary orbit is NOT in this database --
confirmed empirically (zero rows for any 'alf Cen%' host pattern) --
because this is a planet archive, not a stellar-multiplicity catalog.
Rather than silently having nothing to say about it, the cache records
that this source was checked and doesn't cover it, so real_systems.py
can tell "never fetched" apart from "fetched, and this system isn't
here."
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ORGANISM_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = ORGANISM_DIR / "state"
CACHE_PATH = STATE_DIR / "exoplanet_cache.json"

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# hostname patterns to fetch, keyed by the real_systems.py world-key
# each feeds. Alpha Centauri is included deliberately DESPITE being
# known-empty (see module docstring) so a run always records a fresh
# "checked, not covered" timestamp rather than reusing a stale one.
HOST_QUERIES = {
    "proxima": "Proxima%",
    "alpha_centauri": "alf Cen%",
}

COLUMNS = (
    "pl_name,hostname,pl_orbsmax,pl_orbsmaxerr1,pl_orbper,pl_orbeccen,"
    "pl_bmasse,pl_rade,pl_orbincl"
)

REQUEST_TIMEOUT = 20.0


def _run_query(hostname_pattern: str) -> list[dict]:
    query = (
        f"select {COLUMNS} from pscomppars "
        f"where hostname like '{hostname_pattern}'"
    )
    url = f"{TAP_URL}?query={urllib.parse.quote(query)}&format=json"

    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as response:
        body = response.read()

    return json.loads(body)


def fetch_all() -> dict:
    """
    Queries every entry in HOST_QUERIES and returns a cache-shaped dict.
    A per-host failure (network error, malformed response) is recorded
    as an explicit error string for that host rather than aborting the
    whole run or silently dropping the entry -- a partial real fetch is
    still more current than the fully-stale cache it would otherwise
    leave untouched.
    """
    fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    systems = {}

    for key, pattern in HOST_QUERIES.items():
        try:
            rows = _run_query(pattern)
            systems[key] = {"rows": rows, "error": None}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            systems[key] = {"rows": [], "error": str(exc)}

    return {
        "fetched_at": fetched_at,
        "source": TAP_URL,
        "systems": systems,
    }


def main() -> int:
    print(f"Fetching from {TAP_URL} ...")
    cache = fetch_all()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temp = CACHE_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    temp.replace(CACHE_PATH)

    ok = 0
    for key, entry in cache["systems"].items():
        if entry["error"]:
            print(f"  {key}: FAILED -- {entry['error']}")
        elif entry["rows"]:
            names = ", ".join(row.get("pl_name", "?") for row in entry["rows"])
            print(f"  {key}: {len(entry['rows'])} row(s) -- {names}")
            ok += 1
        else:
            print(f"  {key}: 0 rows (not covered by this archive)")

    print(f"\nCached to {CACHE_PATH}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
