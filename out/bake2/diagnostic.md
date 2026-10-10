# T9b diagnostic: the same patch over take-off..landing

Not the shipped patch. Same bake, but the crossfade window is 0.05 s before take-off to 0.30 s after landing, so the whole vault is inside the patch. Shows whether the motion itself penetrates the wall, independent of the play window.

| clip | window (s) | pen box / ground (cm) | hand above top min / med / max (cm) | status |
|---|---|---|---|---|
| xx_Parkour_Sprint_SkipOver_Fd100cm-Up100cm | 0.45 to 1.48 | 15.3 / 9.9 | -3.6 / +2.2 / +11.0 | FAIL |
| xx_Parkour_Sprint_SkipOver_Fd100cm-Up150cm | 0.42 to 1.48 | 40.0 / 16.8 | -27.4 / -23.1 / -16.2 | FAIL |
| xx_Parkour_Sprint_SkipOver_Fd200cm-Up100cm | 0.42 to 1.37 | 18.1 / 23.5 | -10.1 / +8.9 / +16.9 | FAIL |
| xx_Parkour_Sprint_SkipOver_Fd010cm-Up100cm | 0.57 to 1.63 | 4.7 / 1.9 | +0.1 / +9.5 / +13.2 | PASS |
| xx_Parkour_Sprint_SkipOver_Fd200cm-Up150cm | 0.62 to 1.88 | 25.5 / 13.7 | -6.4 / +0.6 / +8.5 | FAIL |
| xx_Parkour_Sprint_SkipOver_Fd300cm-Up150cm | 0.45 to 1.90 | 36.8 / 19.5 | +23.2 / +31.3 / +43.6 | FAIL |
| xx_Parkour_Sprint_SkipOver_Fd100cm-Up050cm | 0.43 to 1.17 | 6.6 / 1.8 | +9.1 / +10.5 / +16.4 | FAIL |
| xx_Parkour_Sprint_SkipOver_Fd200cm-Up050cm | 0.40 to 1.27 | 0.0 / 9.4 | +19.0 / +21.6 / +30.7 | FAIL |
| xx_Parkour_Sprint_SkipOver_Fd100cm-Up150cm_02 | 0.55 to 1.77 | 27.1 / 15.1 | -9.6 / -1.0 / +7.1 | FAIL |
| xx_Parkour_Sprint_SkipOver_Fd010cm-Up200cm | 0.43 to 1.77 | 5.0 / 40.7 | -73.5 / -5.5 / +14.2 | FAIL |
