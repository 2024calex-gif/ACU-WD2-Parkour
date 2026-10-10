# WD2 → AC Unity vault port (v2)

Tools for retargeting Watch Dogs 2 parkour/vault animations onto Assassin's Creed Unity's
parkour clips, plus the analysis pipeline used to fit them to how ACU actually plays its vaults.

**No game data is included.** Extracted/decoded game files (ACU clip dumps, WD2 `.mab` files,
baked patches, in-game recordings) are personal-use only and are git-ignored. To run most of the
pipeline you need your own extracted data in the locations listed in `pipeline/config.py`.

## Layout
| Path | What |
|---|---|
| `pipeline/` | Python pipeline + tests (`pipeline/README.md` has module docs and the patch/mapping formats) |
| `tools/` | helper scripts (clip-name table, capture mapping) |
| `CAPTURE_FORMAT.md` | JSON format of the in-game flight-recorder captures |
| `data/wd2_events*.json` | hand-annotated WD2 clip events (takeoff, hand plants, landing) |
| `build.ps1`, `package.ps1` | build/package scripts for the companion ACU plugin (WD2Vaults, ACUFixes plugin loader; source not in this repo) |
| `out/**/*.md` | analysis reports |

## Key modules (pipeline/)
- `mab.py`, `wd2.py` – Watch Dogs 2 `.mab` animation decoding (rotations)
- `skel.py`, `retarget.py` – skeleton loading and WD2 → ACU retargeting
- `fk.py`, `ik.py`, `blend.py` – forward/inverse kinematics, blending
- `acuclip.py`, `contacts.py`, `dumps.py` – ACU clip dumps, markers, contact detection
- `capture.py`, `matchclip.py`, `slideover_model.py`, `exit_model.py` – fit in-game recordings to a model of how the game plays clips
- `bake2.py`, `speedvault.py` – baking WD2 motion into ACU clip patches (kong / speed vaults)
- `wd2_path.py` – reconstruct a WD2 body path from contacts (WD2 pelvis translation is not decoded)
- `aux_tracks.py` – foot-plant IK targets and inverse-Reference tracks for ACU clips
- `dive*.py` – the (abandoned) dive-roll experiments, kept for reference
- `wd2_eval.py` – scoring WD2 clips against ACU vault targets

## Tests
```
python -m pytest pipeline/tests -q -p no:cacheprovider
```
Many tests need the local game data and are expected to fail or skip without it.
