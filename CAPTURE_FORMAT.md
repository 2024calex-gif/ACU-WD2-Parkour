# WD2Vaults capture format v1 (written by the plugin's flight recorder, read by `pipeline/capture.py`)

One JSON file per recorded vault: `<plugin dir>\WD2Vaults\captures\cap_<yyyyMMdd_HHmmss>_<n>.json`.

```json
{
  "format": "wd2vaults-capture",
  "version": 1,
  "plugin_version": "0.6.0",
  "patching_enabled": false,
  "variant": "",
  "trigger": "xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm",
  "pose_ok": true,
  "pose_bones_missing": [],
  "bones_space": "entity",
  "frames": [
    {
      "t": 0.0,
      "entity_matrix": [16 floats, exactly as stored in memory (Entity::mainTransform)],
      "bones": { "<boneId decimal>": [px, py, pz, qx, qy, qz, qw] }
    }
  ],
  "clips": [
    { "t": 0.0, "ptr": "0x1A2B3C", "handle": "0x296F0C7D9", "name": "xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm",
      "length": 2.1, "patched": false }
  ]
}
```

- `t`: seconds since the capture started (high-resolution timer). One frame entry per game frame.
- `entity_matrix`: 4x4 float matrix in memory order. Translation (world position, metres) = elements 12, 13, 14
  (`BaseEntity::GetPosition()` reads offset 0x30). Elements 0-2, 4-6, 8-10 are the three basis axes.
  In ACU clip data the character moves along entity **+Y** (forward), Z is up.
- `bones`: optional per frame, and may be partial. Keys = bone IDs = CRC32 of the bone name, as decimal strings.
  Bones: Reference, Hips, Spine2, Head, LeftHand, RightHand, LeftFoot, RightFoot, LeftFootThumb1, RightFootThumb1.
  A bone is left out of a frame when the game's bone table had no valid calc index for it. A frame whose self-check
  fails has no `bones` key at all.
  Values: position + rotation in the space named by `bones_space` ("entity" = relative to the entity matrix).
- `variant` (v0.5): the patch variant active when the capture started: the folder name under `WD2Vaults\variants\`, or
  `""` for the top-level files.
- `pose_ok` (v0.5): true only when every frame has all 10 bones. `pose_bones_missing` (v0.5): the bone names missing in at
  least one frame (union over the frames; all 10 when there are no frames). Captures from 0.4.x have neither field:
  readers treat `pose_bones_missing` as all bones when `pose_ok` is false.
- `pool_choice` (v0.9, optional, only on clip records of a pool start): `"original"`, or the source file of the alternative that was
  written when the clip started. `patched` is true when that alternative was not the original.
- `clips`: every animation the player's animation graph STARTS during the capture, in order. `handle` may be null when
  unknown; `name` may be null (resolve offline from `acu_data\anim_clip_names.txt` by handle).
  `patched` = this Animation object held WD2 tracks when the clip STARTED (v0.6). For a gated clip (mapping `only_after`)
  that is the ARMED state at the start: its tracks are installed only while armed, so `patched` is false while it rests.
- Capture starts when the player starts any clip whose name matches `xx_Parkour_(Sprint|Run|Idle)_.*(SkipOver|SlideOver|Vault)`
  or `xx_Parkour_LandingFwd_LandRoll` (or manually from the menu), includes the last 0.5 s before the trigger
  (ring buffer), and stops 1.5 s after the last parkour clip started, or after 8 s max.
