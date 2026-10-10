"""Build the clip-name table the WD2Vaults plugin loads at startup.

Input : acu_data/anim_clip_names.txt   (tab separated, no header: hex id, name, size, container)
Output: v2/data/clipnames_parkour.tsv  (tab separated: hex id without 0x, name)

Only the clips the flight recorder cares about are kept (parkour, navigation, dodge and fall clips).
Prefix matching is case-sensitive, like the plugin's trigger regex.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "acu_data", "anim_clip_names.txt")
DST = os.path.join(ROOT, "v2", "data", "clipnames_parkour.tsv")
PREFIXES = ("xx_Parkour_", "xx_Nav_", "acu_nav", "xx_Fight_Dodge", "xx_Fall")


def is_hex(s):
    return len(s) > 0 and all(c in "0123456789abcdefABCDEF" for c in s)


def main():
    if not os.path.isfile(SRC):
        print("make_clipnames: source not found: %s" % SRC)
        return 1
    seen = set()
    rows = []
    with open(SRC, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            hex_id, name = parts[0].strip(), parts[1].strip()
            if not is_hex(hex_id) or not name.startswith(PREFIXES):
                continue
            key = hex_id.upper()
            if key in seen:
                continue
            seen.add(key)
            rows.append((hex_id, name))
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8", newline="\n") as f:
        for hex_id, name in rows:
            f.write("%s\t%s\n" % (hex_id, name))
    print("make_clipnames: wrote %d lines to %s" % (len(rows), DST))
    return 0


if __name__ == "__main__":
    sys.exit(main())
