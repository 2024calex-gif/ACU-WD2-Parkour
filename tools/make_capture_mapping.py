"""Build the mapping.json for the WD2Vaults *capture session* package.

The capture session patches NOTHING. Its mapping only lists original clips, so the plugin records
the vaults they play and dumps the original clip data (dump_only = true, no source file).

Input : acu_data/anim_clip_names.txt   (tab separated, no header: hex id, name, size, container)
        v2/data/acu_dumps/             (existing dumps; clips already dumped are skipped)
Output: v2/out/capture_session/mapping.json

Entry format:
  {"handle": "0x<HEX>", "name": "<clip name>", "source": "", "enabled": true, "dump_only": true}
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "acu_data", "anim_clip_names.txt")
DUMPS = os.path.join(ROOT, "v2", "data", "acu_dumps")
OUT_DIR = os.path.join(ROOT, "v2", "out", "capture_session")
OUT = os.path.join(OUT_DIR, "mapping.json")
MAX_ENTRIES = 150

VAULT_RE = re.compile(r"^xx_Parkour_(Run|Idle)_.*(SkipOver|SlideOver|Vault)")
# (group label, matcher). First match wins; the groups are disjoint by prefix.
GROUPS = [
    ("LandingFwd", lambda n: n.startswith("xx_Parkour_LandingFwd_")),
    ("Run/Idle vault", lambda n: VAULT_RE.match(n) is not None),
    ("Muraille_Exit", lambda n: n.startswith("xx_Parkour_Muraille_Exit_")),
    ("Leaping", lambda n: n.startswith("xx_Parkour_Leaping_")),
]


def is_hex(s):
    return len(s) > 0 and all(c in "0123456789abcdefABCDEF" for c in s)


def dumped_handles():
    """Upper-case hex ids (no 0x) that already have a dump file named <HEX>_<...>."""
    done = set()
    if not os.path.isdir(DUMPS):
        print("make_capture_mapping: no dump folder at %s, nothing excluded" % DUMPS)
        return done
    for fn in os.listdir(DUMPS):
        head = fn.split("_", 1)[0]
        if "_" in fn and is_hex(head):
            done.add(head.upper())
    return done


def main():
    if not os.path.isfile(SRC):
        print("make_capture_mapping: source not found: %s" % SRC)
        return 1

    done = dumped_handles()
    group_of = {}          # (HEX, name) -> group label
    entries = {}           # HEX -> name (one entry per distinct id)
    skipped_dumped = 0
    with open(SRC, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            hex_id, name = parts[0].strip(), parts[1].strip()
            if hex_id.lower().startswith("0x"):
                hex_id = hex_id[2:]
            if not is_hex(hex_id):
                continue
            label = None
            for g_label, matches in GROUPS:
                if matches(name):
                    label = g_label
                    break
            if label is None:
                continue
            key = hex_id.upper()
            if key in done:
                skipped_dumped += 1
                continue
            if key in entries:
                continue
            entries[key] = name
            group_of[key] = label

    # Same name with different ids: every id stays (keys are ids, not names).
    clips = [
        {
            "handle": "0x" + hid,
            "name": name,
            "source": "",
            "enabled": True,
            "dump_only": True,
        }
        for hid, name in entries.items()
    ]
    clips.sort(key=lambda c: (c["name"], c["handle"]))

    total = len(clips)
    if total > MAX_ENTRIES:
        print("WARNING: %d matching clips, capping at %d (first %d by name)"
              % (total, MAX_ENTRIES, MAX_ENTRIES))
        clips = clips[:MAX_ENTRIES]

    kept_groups = {}
    for c in clips:
        g = group_of[c["handle"][2:]]
        kept_groups[g] = kept_groups.get(g, 0) + 1

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"clips": clips}, f, indent=1)
        f.write("\n")

    print("make_capture_mapping: wrote %d entries to %s" % (len(clips), OUT))
    print("  already dumped (skipped): %d" % skipped_dumped)
    for g_label, _ in GROUPS:
        print("  %-16s %d" % (g_label, kept_groups.get(g_label, 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
