# WD2 -> ACU vault pipeline (v2 package)

Reuses the proven v1 modules in `..\..\tools\` (unchanged logic) and replaces the hard-coded paths with `config.py`.

## Modules

| File | What it does |
|---|---|
| `config.py` | Every path (WD2 data, skeletons, ACU clip index, dumps, output). Computed from this file's location. |
| `mab.py` | Decoder for Watch Dogs 2 `.mab` rotation clips (copy of `tools\mab.py`). |
| `skel.py` | Loads the WD2 (XML) and ACU (binary) bind-pose skeletons, bone-name hashes (copy of `tools\skel.py`, paths from config). |
| `retarget.py` | WD2 -> ACU rotation retargeting, `Rig` and `retarget_clip` (copy of `tools\retarget.py`, no logic change). |
| `dumps.py` | Reads the original ACU clip dumps in `..\data\acu_dumps\`: lookup by handle or name, track access, rotation and position sampling, bone-name table. Replaces `tools\dumpread.py`. |
| `wd2.py` | Finds and loads the unpacked WD2 `.mab` files (`mab_index()`, `load(stem)`). |
| `clipnames.py` | ACU clip name <-> hex id index from `acu_data\anim_clip_names.txt`. |
| `tests\` | pytest suite (see below). |

## Running the tests

```
python -m pytest "pipeline\tests" -q
```

Needs Python 3.12, numpy and pytest (`python -m pip install pytest`). The tests only read files; they never touch the game folder.

Known data facts the tests record:
- 398 `.mab` files, 397 parse (`b02-054-02_layer-flowerbasket-lland` fails in `mab.parse`).
- WD2 skeleton has 75 bones, ACU skeleton 109; 52 WD2 bones map onto ACU bones.
- 53 clip names appear twice in `anim_clip_names.txt`; the name map keeps the last id.

## Flight-recorder captures (`capture.py`, `capture_report.py`)

Reads the capture JSON written by the plugin's flight recorder (format: `..\CAPTURE_FORMAT.md`, version 1).

| File | What it does |
|---|---|
| `capture.py` | `load(path)` -> `Capture`; `entity_axes` / `to_world` / `bone_world` (matrix reading set by `ROW_MAJOR_BASIS`, optional `row_major=` override); `check_convention(cap)` (which reading puts entity +Y along the movement); `clip_segments`; `obstacle_from_clip(cap, dumps)` (obstacle edges in world space from the trigger clip's dump markers); `events(cap, obstacle)` (hand_plant, takeoff, landing). |
| `capture_report.py` | `python capture_report.py <capture.json or folder> [--out-dir DIR]`: prints the summary and writes `..\out\captures\<name>.png` (top view, side view, clip timeline). |
| `tests\synth_capture.py` | Writes a synthetic capture (sprint at 6 m/s, 1 m obstacle placed from the SkipOver dump, hands planted at 1.05 s) and returns its ground truth. |

`ROW_MAJOR_BASIS = True` is unconfirmed against real data: run `check_convention` on a real capture first. Its verdict is printed by the CLI, with a warning when it disagrees with the constant.

Run the tests with `python -m pytest "...\pipeline\tests" -q -p no:cacheprovider`.

## Added in T5 (measure side: original ACU vault clips)

| File | What it does |
|---|---|
| `fk.py` | Vectorised forward kinematics `fk(bones, local_rot, local_pos, root_rot, root_pos)` (batch dims allowed, parents must precede children) and quaternion helpers `qmul, qconj, qnormalize, qrot, slerp`. Same conventions as `skel.py` / `retarget.py`. |
| `acuclip.py` | `PoseSeq` and `pose_sequence(dump, step)` (ACU bone transforms per frame, quaternion sign-continuous), `entity_transforms` / `entity_positions` (Reference track applied once as root), `markers(dump, frames)`, `world_positions(seq, dump)` (entity + root displacement TrackID 0), `obstacle(dump)` (world box from the edge markers), `vault_dumps()`, `parse_name()`. |
| `contacts.py` | Contact intervals of hands and feet/toes against the obstacle top or the ground (`point_masks`, `runs`, module thresholds) and summary events `events(seq, dump)`: `hand_plant`, `takeoff`, `foot_on_top`, `landing`. |
| `acuclip_report.py` | `python acuclip_report.py`: prints one row per vault clip, writes `..\out\acu_events.json` and `..\out\sheets\acu_<clip>.png`. |
| `tests\test_fk.py`, `tests\test_acuclip.py` | FK bind-pose check, batch / root-composition checks, marker constancy, height/depth vs clip name, plausibility checks, synthetic contact test. |

Track facts found in T5 (all 20 vault dumps):
- TrackID `0` position = root displacement along +Y (0 to about 10 m); world = entity + this.
- `2307824085` near edge (z = height), `277334127` far edge (far.y - near.y = depth), both constant in world space within 1e-6 m.
- `1736505593` = ground landing point (z = 0), 0.63 to 1.18 m beyond the far edge; the first lead-foot ground contact after the far edge is within 0.40 m of it in y (19 of 20 clips within 0.25 m).
- `2246150967` and `3215835963` are the **inverse of the Reference transform** (position = -conj(R_ref) * P_ref, rotation = conj(R_ref) up to sign). They are not skeleton bones.
- Helper bones with no skin role: `ChildOfNeckSmthOnGroundLevel` (reaches 0.58 m inside the wall box, excluded from body checks), `*Pos` deform joints, `A_BloodFX`, `PistolInHandLeft`, `RightHand_Weapon_Ref`.
- Contacts use hand/foot/toe joint centres (no palm, sole or mesh offset).

## Added in T6 (IK and blending: `ik.py`, `blend.py`)

Conventions: same as `fk.py`. `frame_index` / `a, b` frame arguments are indices into `seq.frames` (0 .. F-1),
times are seconds, `dst_frames` are 60 Hz frame numbers. `root_rot` / `root_pos` is an extra transform applied in
front of the whole skeleton (to bone 0 `Reference`, as `fk.fk` does); with identity and the TrackID-0 displacement
it gives the same world positions as `acuclip.world_positions`.

| File | What it does |
|---|---|
| `ik.py` | `CHAINS` (LeftHand, RightHand, LeftFoot, RightFoot -> upper/lower/end bone names), `bone_index(bones, name)`, `chain_indices(bones, effector)`, `two_bone_ik(bones, local_rot, local_pos, chain, target, pole=None, root_rot, root_pos, max_reach=0.999, keep_end_world_rot=True)` (one pose, returns a new `local_rot`; only upper and lower change, plus the end when `keep_end_world_rot`), `solve_sequence(seq, root_rot, root_pos, targets, poles=None, ...)` (targets: effector -> list of (frame_index, world_target, weight); weight slerps the IK result in). |
| `blend.py` | `crossfade(seq_a, seq_b, f0, f1, curve='smoothstep')` (slerp + lerp, sign-continuous output), `fade_weights(n_frames, intervals, ramp)` (1 inside each [start, end], linear fall to 0 over `ramp` frames outside), `time_warp(seq, src_events, dst_events, dst_frames)` (piecewise-linear time map, slerp sampling). |
| `tests\test_ik.py`, `tests\test_blend.py` | Reachable / unreachable / pole / no-pole IK on the real dump pose `xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm`, chain checks, `solve_sequence` weights and root transform; crossfade endpoints, sign-flip continuity, fade weights, time-warp identity and doubled segment. |

Chains (checked against the ACU skeleton, all are direct parent chains):
- LeftHand: LeftHand <- LeftForeArm <- LeftArm <- LeftClavicle <- Spine2 <- Spine1 <- Spine <- Reference
- RightHand: RightHand <- RightForeArm <- RightArm <- RightClavicle <- Spine2 <- Spine1 <- Spine <- Reference
- LeftFoot: LeftFoot <- LeftLeg <- LeftUpLeg <- Hips <- Reference
- RightFoot: RightFoot <- RightLeg <- RightUpLeg <- Hips <- Reference

IK notes:
- The solve works on the ancestors of the end bone only (a few quaternion products per call), so `solve_sequence` is cheap.
- A target beyond `max_reach * (upper + lower)` extends the chain along the target direction; a target closer than
  |upper - lower| is clamped to that distance. Both give finite results.
- Raw dump steps: every vault dump already has frame-to-frame rotation steps of 20 to 38 degrees on some bone
  (LeftFoot and hand tracks are keyed every 2 frames). A "< 10 degrees per frame" check therefore cannot hold on the
  raw clip. The sign-flip test checks instead that the output adds no step beyond the input's own largest step.

## Added in T9a (root offset solver: `rootsolve.py`)

Solves a smooth per-frame translation `Delta(t)` [F, 3] (X always 0; Y forward, Z up, entity/world space) for the
original Reference position track, so that contacts are met. Input `P` is [F, K, 3] world positions of K tracked
points without the offset. Constraint frames are indices into P's first axis, not frame numbers.

| Name | What it does |
|---|---|
| `Height(point, frames, z, weight)` | point z should equal `z` on the frames (equality, weighted least squares). |
| `Above(point or 'all', frames, surface)` | point z must stay on or above the surface at its y. `surface` is None (ground z = 0), a `Box(height, near_y, far_y)` (top `height` inside the y band, ground outside), or any callable `y -> z`. Inequality, solved as an active set (see below). |
| `Pin(frames, value=0, weight)` | Delta equals `value` (scalar, or a (dy, dz) pair) on the frames. |
| `solve_offset(P, constraints, lam_smooth=50, lam_small=(5.0, 0.1), ineq_weight=1000, clearance=0.01, tol=1e-3, max_iter=20)` | returns `(delta, info)`. |
| `apply_offset(ref_pos_track, delta)` | Reference position track plus the offset. |
| `contact_constraints_from_events(obstacle, frames, contacts, point_index, pin_first=3, pin_last=6)` | builds Height constraints from contact intervals (`contacts.events()['contacts']` format, frame numbers, `'top'`/`'ground'`), one `Above('all')` over the obstacle box, and Pins at the first and last frames. Work in world space. |

Objective: `sum w^2 r^2` over Height and Pin rows, `+ lam_smooth ||D2 Delta||^2` (both axes), `+ lam_Y ||Delta_Y||^2 + lam_Z ||Delta_Z||^2`.
Joint offsets above the skin contact: `HAND_OFFSET` 0.03, `FOOT_OFFSET` 0.08, `TOE_OFFSET` 0.03 (`FootThumb1`).

Active set for `Above`: solve; each (point, frame) pair below its surface by more than 1 mm is frozen at
surface + 1 cm; re-solve. Per frame the one equality is the largest required lift over that frame's active pairs,
since all points share one translation. Stops when no new violations appear or after `max_iter` solves.

`info`: `iterations`, `max_residual` (`Height`, `Pin`, `Above_eq` = how far an active pair is still below its target),
`max_penetration` (m), `n_violations`, `converged`, `active` (sorted (point, frame) pairs).

Real-data finding (`xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm`, offset table as above): the solver does not
move the body sideways (`max |dy|` about 0), but vertical offset reaches about 5.5 cm. The clip's own measured contact
heights (foot ~0.10 m, toe ~0.00 m, hand top ~1.04 m) differ from the table, and foot and toe are both
ground contacts on the same frames, so one translation cannot satisfy both. The solver does not reach zero on this
clip; the test for that is an xfail (strict) until the contact model is fixed.

Tests: `tests\test_rootsolve.py` (synthetic Height/Pin, penetration upward-only, apply, helper targets, real clip).

## Added in T7 (WD2 clips: contact candidates from rotations, `wd2events.py`)

WD2 clips carry rotations only. `wd2events.py` retargets a clip (`retarget.Rig`, `target_locals`), runs `fk.fk` with
identity root and the Reference (pelvis) at the origin, and works on pelvis-relative effector traces at 30 Hz.

| Function / CLI | What it does |
|---|---|
| `effector_traces(stem)` | Pelvis-relative positions of LeftHand, RightHand, LeftFoot, RightFoot, LeftFootThumb1, RightFootThumb1; forward velocity `vf` (central differences, [1,2,1] smoothing) and height `z` per effector; joint positions for the stick figure; forward axis and body speed. |
| `forward_axis(heading, t)` | Horizontal part of the pelvis heading (Reference rotation applied to +X), median over the first 0.5 s. |
| `estimate_body_speed(traces)` | Per foot, the most negative sustained run (>= 3 ticks, v_f < 0) in the first second, as a median; body speed = minus the median over feet. 0 when no foot moves backward. |
| `candidate_contacts(traces, speed=None)` | Intervals (s) per effector where \|v_f + s\| <= 0.35 s for >= 3 ticks. |
| `suggest_events(contacts)` | takeoff (end of the last foot contact before the hand plant), hand_plant_start / end (first merged run of hand contacts), landing (start of the first foot contact after the hand plant). |
| `plot_sheet(...)` | Writes `..\out\sheets\wd2_<stem>.png`: side-view stick strip (top), v_f and z per effector with shaded candidates and dashed events. |
| `python wd2events.py <stem> ... [--speed M/S] [--strip-speed M/S]` | Prints per clip the duration, forward axis, body speed, contact intervals and suggested events; writes the sheet. |

Facts found while building it (T7):
- Forward axis = ACU +X. Bind-pose toes point along +X (ACU toe - foot = +0.13 m in X), the mapped WD2 toe direction is also +X, and the pelvis heading at t = 0 is +X in all four test clips.
- The run-up pelvis-relative speed of the feet in the test clips is small (roughly 0 to 1.4 m/s in the first second, not 5-7 m/s): the rotations do not encode the sprint speed, so `estimate_body_speed` returns 1.7 to 3.4 m/s from leg swings. The contact candidates therefore also match backward leg swings.
- Pelvis-relative velocity alone does not separate planted limbs from swinging ones. Treat `candidate_contacts` as a starting list for a human, not as ground truth. `--speed` overrides the body speed (for example with the ACU run-up speed from a dump).
- The stick strip uses the spec's 1.0 m/s offset, so figures 0.1 s apart overlap. `--strip-speed 10` spreads them.

Tests: `tests\test_wd2events.py` (synthetic traces for body speed, the candidate band, min run length, suggested events, and one real-clip shape check).

## Added in T9b (bake2: WD2 catpass onto the ACU SkipOver clips, `bake2.py`)

Bakes the WD2 `catpass_kong` motion (`..\data\wd2_events.json`, entry `catpass_kong`) onto the ten `xx_Parkour_Sprint_SkipOver_*` clips. Rotation tracks only; the Reference position track is not written (patch_translation false).

| Function | What it does |
|---|---|
| `time_map(acu_ev, wd2_ev, clip_len)` | Clip time -> WD2 time, piecewise linear through takeoff / hand plant / landing anchors, slope 1 outside, clamped to [static_until, duration]. ACU hand plant missing -> placed by the WD2 fraction of takeoff..landing (`acu_hand_plant`). |
| `wd2_pose_at(rig, m, t_wd2)` | `rig.target_locals(rig.source_locals(m, t*30))` -> (locals, world rotations). |
| `compensated_rotations(rig, rot, frame, wt, patched)` | Local rotations of the patched bones so that dump Reference + locals reproduce `R.A_REF * Wt` in world space (same formula as `tools\bake.py`). Walks the skeleton with the actual parent orientation. Hips (no track) and Reference are never patched. |
| `bake_clip(rig, m, dump, acu_s, wd2_ev, o, window)` | 30 Hz keys over the clip, then `crossfade_to_original` (smoothstep, 0.10 s in, 0.15 s out, sign-continuous, exact original outside the window). |
| `reduce_keys` | Greedy 0.15 deg key reduction (checks every skipped key against the kept interpolation; bake.py checked only the successor). |
| `write_patch`, `mapping_entry` | Plugin format (`Length`, `Tracks` of codec 1 / infotype 0 with integer 60 Hz `Frame`) and mapping entries (prebaked true, patch_translation false). |
| `validate_clip`, `penetration_parts` | Composed pose in world space (dump Reference + patched locals + TrackID-0 displacement): wall-box and ground penetration, hand height above the top during the mapped plant, angular steps, key-reduction error. |
| `plot_clip`, `write_report`, `write_diagnostic` | PNG per clip, `report.md`, `diagnostic.md`. `report.md` appends `..\out\bake2\observations.md` when that file exists. |
| `load_play_model`, `play_for` | Per-clip `o` / `window` from `..\out\match\play_model.json` when present; otherwise o = 0 and the window from `tools\measured_play.json`, else 0.38 * length. |

Run (writes patches, mapping, PNGs, report):

```
python "pipeline\bake2.py"
python "pipeline\bake2.py" --diagnostic   # take-off..landing window, writes diagnostic.md only
```

Outputs: `..\out\patches\<HEX>.baked.json` x10 + `mapping.json` (10 entries only); `..\out\bake2\<clip>.png`, `report.md`, `diagnostic.md`, `observations.md` (hand-written, see there). The play model used is printed into `report.md`.

Tests: `tests\test_bake2.py` (time map anchors / monotonic / slope 1 / clamps / fallback; crossfade endpoints synthetic and real; compensation FK round-trip; reduction; penetration metric; written patch format and TrackIDs vs the dump; mapping entry; real outputs when present).

Limits found in T9b (details in ..\outake2\observations.md): 9 of 10 clips fail the wall/ground penetration check with the play model (the motion itself penetrates 1.5 m walls and dips toes below ground; no root offset and no IK in this step); the --diagnostic run shows the whole vault.

## Added in T10 (play model from the captures: `matchclip.py`, `matchclip_report.py`)

Fits, per flight-recorder capture, the time map of its trigger clip in the game and the pelvis offset against the
entity's rise. Uses the original dump of the trigger clip (`dumps.load_by_handle`, falls back to the name) and the
entity-space bones of the capture (captures with any recorded bone; partial poses load, missing bones are NaN per frame).

| Function / CLI | What it does |
|---|---|
| `match_window(cap, clip_index)` | Window from the clip's start to the first later start of a `xx_Parkour_Muraille_Exit*`, `xx_Parkour_Landing*`, `xx_Parkour_Leaping*`, `xx_Nav*` or `acu_nav*` clip, capped at the clip length and the capture end. |
| `fit_time_map(cap, dump, window)` | clip_time = o + r * (t - t_start), yaw about Z. Compares FIT_BONES (LeftHand, RightHand, LeftFoot, RightFoot, Head, Spine2) relative to Hips, in-game vs clip. Grid o [-0.3, 1.5] step 0.01, r [0.6, 1.6] step 0.02, yaw [-180, 180] step 5, then a shrinking local refinement. Returns o, r, yaw, RMS (cm), RMS of the r = 1 fit, the naive RMS (o = 0, r = 1, best yaw) and the unfitted baseline (o = 0, r = 1, yaw = 0). |
| `fit_landscape(cap, dump, window, yaw)` | RMS over the (o, r) grid at a fixed yaw (used for the PNG). |
| `pelvis_correction(cap, dump, fit)` | In-game Hips minus the clip's Hips (entity space, yaw applied) against the entity's z rise since the clip start; linear fit dz = a * rise + b with R2. |
| `python matchclip_report.py [--captures DIR] [--out-dir DIR] [--exclude-poor]` | Table (stdout and `summary.txt`), one PNG per capture, and `play_model.json`. Default folders: `..\data\captures\session1_20261008` and `..\out\match\`. |

Outputs (`..\out\match\`): `cap_<name>.png` per capture (hand/foot z relative to Hips, in-game vs predicted; Hips z
offset vs rise with the fit; RMS landscape over (o, r)), `summary.txt`, `play_model.json`:
`{"per_clip": {clip: {"o", "r", "window", "n"}}, "median": {"o", "r", "yaw"}, "pelvis": {"a", "b"}}`.
`per_clip` values are medians over the captures of the clip (`window` = median window length in seconds), `median`
and `pelvis` are medians over all fitted captures. By default every fitted capture is used.

Facts from the 19 session-1 captures (`session1_20261008`):
- Captures 1-3 have no pose (`pose_ok` false): skipped. 16 fit.
- The best-fitting group is the `SkipOver_Fd010cm-Up200cm` and `Fd010cm-Up100cm` captures (9-13): o about 0.36 to 0.59 s,
  r about 1.0, RMS 12-14 cm (naive 44-56 cm). With the in-game take-off about 0.1 s after the clip start (PLAN section 4),
  o of about 0.38 s says the game starts the clip about 0.38 s in, so the clip-to-game gap is a start offset, not a
  rate change, for this group.
- r is poorly identified for most captures: the landscape is a diagonal valley in (o, r), and for most captures the
  r = 1 fit is within about 1.5 cm RMS of the free fit (capture 4 is the exception at 8 cm, capture 7 is 3 cm). Captures 4, 6
  and 7 end at r about 0.47 to 0.56, below the coarse grid (0.6), so their r is not trustworthy.
- No capture fits within `POOR_RMS_CM` = 5 cm. The game does not replay the stored clip shape exactly: capture 6 shows
  hands about 0.7 m above Hips at the peak against 0.27 m in the clip. The game probably adapts the pose to the real
  obstacle height (a 1.25 m wall for an Up100 clip), so a time map cannot close the gap.
- Pelvis: the entity-local Hips z hardly follows the entity rise (median a -0.10, median R2 0.24; for captures 8-13
  a between -0.02 and -0.09 with the rise up to 1.7 m). The Hips do not drop to compensate the rise. That conflicts
  with the PLAN section 4 note that the Reference z drops, and should be checked before the pelvis model is used.
- Idle and LandRoll captures (15, 17-19) have no rise (max rise +0.03 m for 15, -0.04 to -0.22 m for 17-19: the entity
  stays flat or drops), so their a is not meaningful (a of +2.1 and -5.6 with R2 0.25 and 0.14).

Tests: `tests\test_matchclip.py` (synthetic capture from a real dump: window cases, time map and yaw recovery, landscape
minimum, pelvis rise recovery, one real-capture smoke test, play-model medians, poor-fit reasons).

## Added in T9c (root offset and hand IK on the bake: `bake2.py`, section 7)

Run `python bake2.py` (writes `..\out\patches\` with `mapping.json`, `..\out\bake2\<clip>.png` and `report.md`).
`python bake2.py --t9b` reproduces the T9b rotation-only bake, and `--diagnostic` is unchanged.

| Function | What it does |
|---|---|
| `measure_contact_offsets(clips)` | Median height of Hand, Foot and FootThumb1 joints above their support over the contact frames of the original dumps. |
| `bake_t9c(...)` | Per clip: patch rotations, blend, root solve on the window (`rootsolve.solve_offset`), hand IK on the un-faded patch (`ik.solve_sequence`), then the tracks. Returns the tracks, `delta` (zero outside the window), targets and solver info. |
| `root_constraints(...)` | Height for both hands on the plant, Height for the lower of Foot and FootThumb1 on each ACU contact frame, Above('all') against the box, Pin(0) on the first and last 4 window frames minus the plant. |
| `rotation_track` / `translation_track` | Original keys outside [window start - 1, window end + 1] copied; inside rebuilt at 60 Hz and reduced (0.15 degree, 2 mm). |
| `validate_t9c(...)` | Metrics on the composed pose exactly as the tracks play it: penetration, hand error to target, Delta change per frame, exactness outside the window. |
| `write_patch_t9c`, `mapping_entry_t9c` | Patch with the rotation tracks (codec 1) and the Reference translation track (codec 27, infotype 3); `patch_translation: true`. |
| `bake_all_t9c(write)` | All ten SkipOver clips; PASS sets `enabled`. |

Settings are `T9C_*` constants. Two diagnostic switches exist and are off in the shipped run: `T9C_PLANT_SOURCE = 'acu'`
(plant = the original clip's own hand-on-top frames) and `T9C_SOLVE_EXCLUDE` (bone names left out of the root solve only).

Measured offsets over the original SkipOver dumps: hand 0.037 m, foot 0.103 m, toe -0.002 m.

Known limits found in T9c (details in `..\out\bake2\observations.md`): the fixed hand target cannot hold for the whole
WD2 plant at ACU's root speed (about 4.5 m/s, arm about 0.55 m); the hard Above rows make Delta jump per frame; finger
joints in the wall band drive large lifts. Those limits fail every clip at the spec settings, so every patch ships
with `enabled: false`.

Tests: `tests\test_bake2.py`, T9c section (Delta exactly zero outside the window, translation and rotation keys outside
the window equal the original, reduction tolerances, hand held on target over the first 8 plant frames of
Fd100cm-Up100cm, and a PASS-clip hand check that skips while no clip passes).

## Added in T9d (default method: root lift, hand IK, foot clamp; `bake2.py` section 8)

`python bake2.py` now runs T9d (`bake_all_t9d`). `--t9c` runs the T9c least-squares root solve, `--t9b` the T9b rotation-only bake.

| Function | What it does |
|---|---|
| `finger_bones(rig, body)` | The 30 hand finger joints (excluded from the lift and from PASS, reported separately). |
| `lift_need`, `smooth_envelope`, `lift_taper` | Required lift per window frame (non-finger points of the blended pose, clearance 0.01 m), numpy Gaussian upper envelope (sigma 5 frames, 40 iterations; scipy is not installed), smoothstep taper over 6 frames at each window end (not applied in an end zone with a need). |
| `_ik_weights` | Hand IK weight: 3-frame ramp in before the plant, 1 while the shoulder-to-target distance is at most 0.95 x arm length, 3-frame fade out from the first frame out of reach or after the plant. |
| `bake_t9d` | Lift, hand IK to fixed targets (z = top + Hand offset), foot clamp IK on the lower of Foot / FootThumb1, window crossfade, tracks. |
| `touch_frames` | Consecutive plant frames from the first one with each hand within 5 cm of its target (PASS needs at least 4). |
| `source_steps` | Largest angle per 60 Hz frame per patched bone: final patch vs the retargeted WD2 source through the time map (no lift, IK or crossfade). |
| `validate_t9d`, `write_report_t9d`, `bake_all_t9d` | Metrics, report (T9b before, T9c, T9d tables, jaggedness), mapping (enabled only for PASS). |

Diagnostic switches (off in the shipped run): `T9D_REACH` (reach gate), `T9D_LIFT_NO_FEET` (feet left out of the lift).

Known limits (see `..\out\bake2\observations.md`): the lift follows the take-off toe, and the 1 cm clearance rule
keeps the lift above zero at the window start, so the taper never applies there; the hand target snaps to top + 0.037 m,
which can lie far below a lifted hand; the crossfade over 0.10 s and 0.15 s produces most of the per-frame angle steps.

Tests: `tests\test_bake2.py`, T9d section (finger count, envelope and taper, Delta zero outside the window, keys outside the
window equal the original, touch on each enabled clip).

## Added in T9e (default method now; `bake2.py` section 9)

`python bake2.py` runs T9e (`bake_all_t9e`): the lift sees the non-finger body without Foot / FootThumb1 (feet are handled
by the foot clamp IK only) and is signed (not clamped at 0); an upper bound U(f) on the plant keeps both shoulders within
0.95 x arm length of their hand targets (`reach_upper`, exact); `d0_rule` gives the preferred lift, `solve_lift_t9e`
smooths it (floor L, bound U where feasible) and tapers it at the window ends. Fades per clip: `fades_for` (0.20 s / 0.25 s,
capped at 0.30 / 0.35 of the window). Each clip also gets `<clip>.gif` (`write_gif_t9e`).
`--t9d` runs the T9d method (`bake_t9d`), `--t9c` the T9c solve, `--t9b` the T9b bake. Tests: `tests\test_bake2.py`, T9e section.

## mapping.json entry fields (plugin WD2Vaults 0.9)

The plugin reads `WD2Vaults\mapping.json` (`{"clips": [ ... ]}`). Fields per entry:

| Field | Default | Meaning |
|---|---|---|
| `handle` | (required) | Animation handle, hex string, e.g. `"0x296F0C7D9"` |
| `name` | `""` | Clip name (also used to match `only_after`) |
| `source` | (required) | Patch JSON file (in `WD2Vaults\`, or the active variant folder) |
| `enabled` | `true` | Entry on/off |
| `prebaked` | `false` | Source keys are already in clip time (60 Hz frames), no retiming |
| `dump_only` | `false` | Only export the original clip, never patch it |
| `src_start`, `src_end` | `0`, `0` | Source window (seconds) mapped into the played part of the clip |
| `play_seconds`, `play_fraction`, `extra_blend` | measured / `0.40` / `0.10` | Played part of the clip |
| `patch_translation` | `false` | Also replace LocalTranslation (infotype 3) tracks |
| `only_after` | none | **Gated (v0.6):** array of exact clip names. The entry is patched only while its clip follows one of them (a single string is also accepted). |
| `only_after_regex` | none | **Gated (v0.6):** ECMAScript regex over the clip name, same meaning as `only_after`. An invalid regex is ignored and logged. |
| `arm_seconds` | `2.0` | **Gated (v0.6):** arm window in seconds, from the last trigger; `<= 0` means 2.0 |
| `pool_weight` | absent | **Pool (v0.9):** float 0..1. Absent = the normal behaviour (permanent patch, or gated). Present = a POOL entry: each player start of the clip picks one alternative at random. Entries that share a `handle` form one group; P(ORIGINAL) = max(0, 1 - sum of the weights). A sum above 1 is normalised (logged). |

**Pool mode (v0.9).** Several entries with the same `handle` and a `pool_weight` each give one WD2 alternative (their own `source`).
Per player start of that clip, one alternative is picked at random (the rest of the probability is the ORIGINAL animation). The
alternatives are built once per Animation object (and again on force, re-patch or variant change); the clip rests with the
ORIGINAL pointers until the first player start. The choice is made in the animation-start hook and kept until the next player
start of that clip. NPC starts do not choose: they play the set that is in place. A pool handle that is also used by a permanent or
gated entry is ignored (logged). Logs: `[pool] <clip> -> <source or ORIGINAL> (roll r)`. Menu "Pool": weight sliders (in memory only),
pick counts, force buttons. Example, two WD2 alternatives for one clip, ORIGINAL about 40 % of the time:

```json
{ "handle": "0x<HANDLE>", "name": "xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm", "source": "<HANDLE>_a.baked.json", "pool_weight": 0.35 },
{ "handle": "0x<HANDLE>", "name": "xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm", "source": "<HANDLE>_b.baked.json", "pool_weight": 0.25 }
```

An entry with `only_after` or `only_after_regex` is gated: its patched tracks are built once per Animation object and
installed only while armed (the clip rests with the original tracks). Example, the exit clip of a dive roll that follows a SlideOver:

```json
{ "handle": "0x<HANDLE>", "name": "xx_Parkour_Muraille_Exit_Fd100cm-Dn100cm", "source": "<HANDLE>.baked.json",
  "enabled": true, "prebaked": true, "patch_translation": true,
  "only_after": ["xx_Parkour_Sprint_SlideOver_Fd100cm-Up100cm"], "arm_seconds": 2.0 }
```

Verify in game: `WD2Vaults-log.log` has a `gate: [name] ARMED by '<trigger>'` line per arm and a
`gate: [name] DISARMED: <reason> (gated clip played since arming ...)` line per disarm. The menu lists each gated entry with
its state, arm count and last trigger. Captures record the gated clip's `patched` as the armed state at its start.
`<HANDLE>` is a placeholder: take the real handle from `clipnames_parkour.tsv`, and the clip names from the same file.

### Patch JSON format (`<HANDLE>.baked.json`, plugin 0.7)

A patch file is `{"Length": <seconds>, "Tracks": [ ... ]}`. Each track is one of:

| `infotype` | Track | Codec written | Keys | Needs `Slot` |
|---|---|---|---|---|
| `0` | rotation | the slot's own codec (`1` in the pipeline) | `{"Frame": n, "LocalRotation": [x, y, z, w]}` | no |
| `3` | translation (LocalTranslation) | `27` | `{"Frame": n, "LocalPosition": [x, y, z]}` | no; needs `patch_translation: true` |
| `5` | event (IntegerSignal) | the slot's own codec (`57` or `59` in ACU 1.5.1) | `{"Frame": n, "Int": v}` | **yes** |

Rotation and translation tracks are matched to the clip by `TrackID` (the first slot with that TrackID and the same
information type). **Event tracks (infotype 5) must carry `"Slot": <index>`**, the index into the clip's `rawTracks`
(the `Tracks` index in the clip's dump, `WD2Vaults\dump\<HANDLE>_<name>.json`), because TrackIDs repeat inside a clip.
The plugin replaces that slot only when all of these hold: `Slot` < `rawTracks` size, `AnimTrackDataMapping_[Slot].TrackID`
equals the track's `TrackID`, and the slot's codec has information type 5. Otherwise the track is skipped and the log says why.

```json
{"TrackID": "4294967264", "infotype": 5, "Slot": 12,
 "Keyframes": [{"Frame": 0, "Int": 8388612}, {"Frame": 48, "Int": 4194305}]}
```

- `Int` is an unsigned 32-bit value (for example `4160749570` = `0xF8000002`); a value outside `-2^31 .. 2^32-1` skips the track.
- Event values are held: each value applies until the next key. Keys are retimed like the rotation keys (`prebaked: true` = the
  frames are already in clip time). Frames must be strictly increasing; a key that is not after the previous one is moved to
  previous + 1 and logged as a fix-up.
- The self-check decodes every new key back through the codec (frame and value) and compares it with the encoded key. A
  mismatch means the track is not installed and the original stays; the log line says `self-check MISMATCH`.
  A match logs `event track slot <n> (TrackID <id>, codec <c>): <keys> keys, self-check match`.
- Gated entries (`only_after`) swap event slots with the other slots: armed installs them, disarmed and restore put the originals back.
- The log line `patched N rotation + M translation + K event tracks` counts the event tracks separately.
