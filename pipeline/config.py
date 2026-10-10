"""All file locations used by the WD2 -> ACU vault pipeline.

Every path is computed from this file's location: pipeline -> v2 -> project ROOT.
Nothing here creates directories or touches the game folder.
"""
import os

PIPELINE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(PIPELINE)
ROOT = os.path.dirname(V2)

# Watch Dogs 2 source data
WD2_DATA = os.path.join(ROOT, 'wd2_data')                                  # unpacked .mab files (recursive)
WD2_SKEL_XML = os.path.join(ROOT, 'wd2_skel_xml', 'pelvis_ref.xml')        # WD2 bind-pose skeleton

# Assassin's Creed Unity data
ACU_SKEL_BIN = os.path.join(ROOT, 'acu_clips', 'ACU_BaseSkeleton_Male.bin')  # ACU bind-pose skeleton
ACU_CLIP_NAMES = os.path.join(ROOT, 'acu_data', 'anim_clip_names.txt')     # hex id, name, size, container (TSV)
ACU_DUMPS = os.path.join(V2, 'data', 'acu_dumps')                          # 30 original clip dumps <HANDLE>_<name>.json
ACUFIXES_ANIM_EDITOR = os.path.join(                                       # source with bone-name table
    ROOT, 'plugin', 'ACUFixes-151-src', 'ACUFixes', 'src', 'AnimationTools', 'AnimationEditor.cpp')

# Outputs
OUT = os.path.join(V2, 'out')
PATCHES = os.path.join(OUT, 'patches')
