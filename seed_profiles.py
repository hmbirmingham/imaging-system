"""
seed_profiles.py — One-shot seeder for the MLS organism × plate-type roster.

Writes profiles/_generic.yaml, profiles/_plate_types.yaml, and one YAML per
organism in the lab's list.

IMPORTANT — what this seeder does and does NOT fill in:

  Filled (safe, textbook-unambiguous facts):
    - display_name (corrected binomial + the lab's descriptive note)
    - gram, cell_morphology
    - swarming  (only Proteus mirabilis = true)
    - incubation window  (standard ranges; fastidious organisms widened)
    - plate_types  (standard primary-isolation media — workflow defaults, editable)

  NOT filled (deferred to in-app PhD validation):
    - per-plate appearance: hemolysis, lactose reaction, colony size, colour.
      These start at the generic `default` block and every profile is written
      with validation.validated = false. Dr. Ryberg fills/confirms them in the
      validation UI, which flips the flag.

Re-running is safe: it overwrites the seed files. Profiles a human has already
validated should NOT be re-seeded — pass --skip-validated to preserve them.

Run:  python seed_profiles.py
"""

from __future__ import annotations

import argparse

from profiles import ProfileStore, PLATE_TYPES_SEED, _GENERIC_DEFAULT
import yaml


# (profile_id, display_name, gram, morphology, swarming, incub_min, incub_max, plates)
# gram: + / -    morphology: rod | cocci | coccobacillus | curved_rod
ORGANISMS = [
    ("escherichia_coli_lf",        "Escherichia coli (classic LF, pink donuts)",        "-", "rod",           False, 18, 24, ["BAP", "MAC", "CHOC"]),
    ("escherichia_coli_bhem",      "Escherichia coli (β-hemolytic, NLF, dimorphic)",     "-", "rod",           False, 18, 24, ["BAP", "MAC"]),
    ("salmonella_enterica",        "Salmonella enterica",                                "-", "rod",           False, 18, 24, ["BAP", "MAC", "XLD"]),
    ("campylobacter_jejuni",       "Campylobacter jejuni",                               "-", "curved_rod",    False, 48, 72, ["CAMPY"]),
    ("streptococcus_pneumoniae",   "Streptococcus pneumoniae",                           "+", "cocci",         False, 18, 24, ["BAP", "CHOC"]),
    ("staphylococcus_aureus_mrsa", "Staphylococcus aureus (MRSA)",                       "+", "cocci",         False, 18, 24, ["BAP", "MRSA", "CNA"]),
    ("staphylococcus_epidermidis", "Staphylococcus epidermidis",                         "+", "cocci",         False, 18, 24, ["BAP", "CNA"]),
    ("staphylococcus_lugdunensis", "Staphylococcus lugdunensis (coag-neg, PYR+)",        "+", "cocci",         False, 18, 24, ["BAP", "CNA"]),
    ("staphylococcus_saprophyticus","Staphylococcus saprophyticus",                      "+", "cocci",         False, 18, 24, ["BAP", "CNA"]),
    ("streptococcus_pyogenes",     "Streptococcus pyogenes (Group A)",                   "+", "cocci",         False, 18, 24, ["BAP", "CNA"]),
    ("streptococcus_agalactiae",   "Streptococcus agalactiae (Group B)",                 "+", "cocci",         False, 18, 24, ["BAP", "CNA"]),
    ("viridans_streptococci",      "Viridans group streptococci",                        "+", "cocci",         False, 18, 24, ["BAP", "CNA"]),
    ("streptococcus_gallolyticus", "Streptococcus gallolyticus",                         "+", "cocci",         False, 18, 24, ["BAP", "CNA"]),
    ("enterococcus_faecalis",      "Enterococcus faecalis",                              "+", "cocci",         False, 18, 24, ["BAP", "CNA"]),
    ("aerococcus_urinae",          "Aerococcus urinae",                                  "+", "cocci",         False, 18, 24, ["BAP", "CNA"]),
    ("moraxella_catarrhalis",      "Moraxella catarrhalis",                              "-", "cocci",         False, 18, 24, ["BAP", "CHOC"]),
    ("neisseria_gonorrhoeae",      "Neisseria gonorrhoeae",                              "-", "cocci",         False, 24, 48, ["CHOC", "MTM"]),
    ("haemophilus_influenzae",     "Haemophilus influenzae",                             "-", "coccobacillus", False, 18, 24, ["CHOC", "QUAD_XV"]),
    ("haemophilus_parainfluenzae", "Haemophilus parainfluenzae",                         "-", "coccobacillus", False, 18, 24, ["CHOC", "QUAD_XV"]),
    ("eikenella_corrodens",        "Eikenella corrodens",                                "-", "rod",           False, 24, 48, ["BAP", "CHOC"]),
    ("listeria_monocytogenes",     "Listeria monocytogenes",                             "+", "rod",           False, 18, 24, ["BAP"]),
    ("yersinia_enterocolitica",    "Yersinia enterocolitica",                            "-", "rod",           False, 18, 48, ["BAP", "MAC"]),
    ("serratia_marcescens",        "Serratia marcescens",                                "-", "rod",           False, 18, 24, ["BAP", "MAC"]),
    ("proteus_mirabilis",          "Proteus mirabilis (classic swarming)",               "-", "rod",           True,  18, 24, ["BAP", "MAC"]),
    ("proteus_vulgaris",           "Proteus vulgaris (citrate+, non-swarming variant)",  "-", "rod",           False, 18, 24, ["BAP", "MAC"]),
    ("citrobacter_freundii",       "Citrobacter freundii (citrate/H2S variable)",        "-", "rod",           False, 18, 24, ["BAP", "MAC"]),
    ("enterobacter_cloacae",       "Enterobacter cloacae",                               "-", "rod",           False, 18, 24, ["BAP", "MAC"]),
    ("enterobacter_aerogenes",     "Enterobacter aerogenes (Klebsiella aerogenes)",      "-", "rod",           False, 18, 24, ["BAP", "MAC"]),
    ("bacillus_cereus",            "Bacillus cereus",                                    "+", "rod",           False, 18, 24, ["BAP"]),
    ("providencia_stuartii",       "Providencia stuartii (urease+ variant)",             "-", "rod",           False, 18, 24, ["BAP", "MAC"]),
    ("morganella_morganii",        "Morganella morganii",                                "-", "rod",           False, 18, 24, ["BAP", "MAC"]),
    ("pasteurella_multocida",      "Pasteurella multocida",                              "-", "coccobacillus", False, 18, 24, ["BAP", "CHOC"]),
    ("klebsiella_pneumoniae",      "Klebsiella pneumoniae",                              "-", "rod",           False, 18, 24, ["BAP", "MAC"]),
    ("klebsiella_oxytoca",         "Klebsiella oxytoca",                                 "-", "rod",           False, 18, 24, ["BAP", "MAC"]),
    ("corynebacterium_striatum",   "Corynebacterium striatum",                           "+", "rod",           False, 18, 48, ["BAP"]),
    ("pseudomonas_aeruginosa",     "Pseudomonas aeruginosa",                             "-", "rod",           False, 18, 24, ["BAP", "MAC"]),
    ("burkholderia_cepacia",       "Burkholderia cepacia",                               "-", "rod",           False, 24, 48, ["BAP", "MAC"]),
    ("aeromonas_hydrophila",       "Aeromonas hydrophila",                               "-", "rod",           False, 18, 24, ["BAP", "MAC"]),
    ("micrococcus_luteus",         "Micrococcus luteus",                                 "+", "cocci",         False, 18, 24, ["BAP"]),
]

GRAM = {"+": "positive", "-": "negative"}


def build_profile(pid, name, gram, morph, swarming, imin, imax, plates) -> dict:
    return {
        "profile_id":   pid,
        "display_name": name,
        "plate_types":  plates,
        "instrument": {
            "plate_diameter_mm": 90,
            "px_per_mm_expected": None,
            "exposure_hint_us":  20000,
            "gain_hint":         1.0,
            "locked":            False,
        },
        "biology": {
            "gram":                  GRAM[gram],
            "cell_morphology":       morph,
            "swarming":              swarming,
            "incubation_time_h_min": imin,
            "incubation_time_h_max": imax,
            # per-plate appearance deliberately left to the generic default until
            # validated in-app; only declare the keys, leave values unknown.
            "plates": {
                "default": {
                    "colony_size":    "medium",
                    "size_tolerance": "loose",
                    "hemolysis":      "unknown",
                    "lactose":        "unknown",
                    "colony_color":   None,
                    "morphology":     None,
                    "notes":          "",
                },
            },
        },
        "validation": {
            "validated":    False,
            "validated_by": None,
            "validated_at": None,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-validated", action="store_true",
                    help="Preserve any profile already marked validated:true")
    args = ap.parse_args()

    store = ProfileStore()

    # organism generic fallback
    (store.organisms_dir / "_generic.yaml").write_text(
        yaml.safe_dump(_GENERIC_DEFAULT, sort_keys=False, allow_unicode=True))
    # plate-type profiles (one file per medium)
    store.plate_types()   # seeds plate_types/*.yaml from PLATE_TYPES_SEED if empty

    written = skipped = 0
    for row in ORGANISMS:
        pid = row[0]
        if args.skip_validated and store.exists(pid):
            existing = store.get(pid)
            if existing.get("validation", {}).get("validated"):
                skipped += 1
                continue
        prof = build_profile(*row)
        (store.organisms_dir / f"{pid}.yaml").write_text(
            yaml.safe_dump(prof, sort_keys=False, allow_unicode=True))
        written += 1

    print(f"Seeded {written} organism profiles "
          f"({skipped} validated preserved) → {store.organisms_dir}/")
    print(f"Plate types: {len(PLATE_TYPES_SEED)} → {store.plates_dir}/")
    print("All per-plate appearance fields are unvalidated (validated: false).")


if __name__ == "__main__":
    main()
