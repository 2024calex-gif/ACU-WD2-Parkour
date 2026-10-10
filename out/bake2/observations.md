These are the author's notes on the T9e run above (lift without feet, two-sided lift with a reach bound, foot clamp,
per-clip fades). Written from the table, the three PNGs (Fd100cm-Up100cm, Fd100cm-Up150cm, Fd010cm-Up100cm), the
jaggedness table and the crossfade-overlap numbers. Nothing was installed and nothing was checked in game.

**1. Result: 1 of 10 clips pass. Fd200cm-Up100cm is enabled; the other nine are `enabled: false`.** The PASS clip has
non-finger penetration 0.3 cm, a Delta step of 2.6 cm per frame, and a touch of 5 frames (left) and 4 frames (right).
After the touch its hands are up to 176 cm from their targets, which the rule allows ("hands may be anywhere after
release"). Its window is short (0.47 s, fades 0.14 s and 0.16 s).

**2. Why the nine others fail.**
- Hand touch 0 (no touch): Fd010cm-Up100cm, Fd200cm-Up150cm, Fd200cm-Up050cm (the touch counts from the first plant
  frame). The IK is correct there: on Fd010cm-Up100cm the IK output hits the target exactly at the first plant frame
  (1.037 m), but the crossfade weight at that frame is 0.69, so the final pose is 69 % IK and 31 % original, and the
  hand is 41 cm off. The 0.20 s fade-in ends at 0.79 s, after the plant starts at 0.72 s.
- Non-finger penetration above 5 cm where the reach bound wins over the penetration: Fd100cm-Up100cm 9.2 cm (box),
  Fd100cm-Up150cm 7.0 cm, Fd100cm-Up050cm 17.7 cm, Fd200cm-Up050cm 9.9 cm, Fd100cm-Up150cm_02 15.0 cm. These clips have
  8 to 14 conflict frames (upper bound below the lift the body needs), where the rule lets penetration win.
- Delta step above 3 cm per frame: Fd100cm-Up150cm 7.9, Fd200cm-Up150cm 12.0, Fd300cm-Up150cm 16.1, Fd100cm-Up150cm_02
  10.3, Fd010cm-Up200cm 18.3 (lift up to +86 cm, the body must go up the full wall height), Fd200cm-Up050cm 3.3.

**3. The reach bound and the lift.** With feet out of the lift the signed lift goes negative on several clips (the body
is lowered: down to -35.6 cm on Fd300cm-Up150cm, -15.5 cm on Fd200cm-Up050cm, -10.3 cm on Fd010cm-Up200cm), because the
upper bound says the shoulders can only reach the hand targets with the body lower. On Fd200cm-Up050cm the lowered body
goes with 9.9 cm of ground penetration, so there the two rules pull against each other; on Fd300cm-Up150cm the same
lowering gives no penetration (0.3 cm). The large Delta steps are on the 1.5 m and 2.0 m walls, where the lift must go up
(up to +41.8 cm on Fd300cm-Up150cm and +85.7 cm on Fd010cm-Up200cm).

**4. Fades and where the plant sits.** fi = min(0.20 s, 0.30 w), fo = min(0.25 s, 0.35 w). The hand plant overlaps the
crossfade on 8 of 10 clips, and the IK only holds the hand when the crossfade weight is 1 (table: pure WD2 part per clip).
Weight at the first plant frame: Fd010cm-Up100cm 0.69, Fd200cm-Up150cm 0.88, Fd200cm-Up050cm 0.81; weight at the last
plant frame: Fd100cm-Up100cm 0.85, Fd200cm-Up100cm 0.06, Fd200cm-Up150cm 0.00, Fd300cm-Up150cm 0.00, Fd100cm-Up050cm 0.85,
Fd100cm-Up150cm_02 0.28. A longer fade-in is what the orchestrator asked for, but it moves the plant start into the
blend, so the touch needs either the fade shorter than the plant lead-in or the IK target taken from the blended pose.

**5. Jaggedness (largest angle per 60 Hz frame).** Patch 18.5 to 32.9 degrees, WD2 source 11.6 to 20.7, original dump
17.3 to 38.3. The patch is above the WD2 source on all ten clips, by 2 to 21 degrees. Against the original dump it is
higher on four clips (Fd100cm-Up150cm +4.7, Fd010cm-Up100cm +9.4, Fd100cm-Up050cm +2.3, Fd010cm-Up200cm +9.4), about equal on
two (Fd100cm-Up100cm, Fd200cm-Up100cm) and lower on four (Fd200cm-Up150cm -5.7, Fd300cm-Up150cm -1.5, Fd200cm-Up050cm -9.4,
Fd100cm-Up150cm_02 -4.2). The decomposition from T9d (raw patch about 13 degrees, crossfade alone about 28 to 32) was not
repeated for T9e, so the crossfade being the main source of the steps is carried over from T9d and not re-measured.

**6. The three PNGs.**
- Fd100cm-Up100cm (FAIL, non-finger penetration 9.2 cm): the hands sit on or just above the top through most of the plant
  (median +4 cm), the pelvis path stays close to the original, and the lift is limited by the reach bound. The box
  penetration is where the lift was not allowed to go up enough.
- Fd100cm-Up150cm (FAIL, penetration 7.0 cm, Delta step 7.9): the solved pelvis path is close to the original with a visible
  kink where the lift changes most; hands are +1 to +7 cm above the top and the hand target sits on the top edge.
- Fd010cm-Up100cm (FAIL, touch 0): the hands are on the top (median +4 cm, max +25 cm) and the body follows the original
  path; the hand target cross is on the top edge, but the hands are 41 cm off it at the plant start because of the
  crossfade weight (point 2). The patched limbs still show crossfade snaps at the window start.

**7. GIFs.** One GIF per clip (`<clip>.gif`, side view, world space, original grey and patch colour, 20 fps over
[o - 0.2, o + window + 0.2] clamped to the clip). Sizes 59 to 109 KB each. Not checked frame by frame.

**8. Not done.** The touch vs crossfade conflict (point 4), the reach bound vs penetration conflict (point 2), and
the wall-height Delta step on the 1.5 m and 2.0 m walls need a decision: for example the hand target from the blended
pose, a shorter fade-in tied to the plant start, or a lower lift cap. Diagnostics were not run for this turn.
