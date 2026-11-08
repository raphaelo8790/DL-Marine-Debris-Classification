G10 Marine Debris - the tile labelling problem (found 7 Nov 2026, fixed in v3)
==============================================================================

FILES IN THIS FOLDER
  1_label_problem_before.jpg     A real SeaClear test photo (Bistrina, frame 2466): a fishing net
                                 (red outline), cut into our 6 tiles. Every tile was labelled CLEAN.
  2_label_fix_before_after.jpg   Four photos (net, tire, cement tube, rope) with each tile's label
                                 under the old rule ("old:") and the new v3 rule ("new:").


WHAT HAPPENED
  The datasets are labelled correctly: SeaClear's annotators drew an outline around every object.
  The problem came from OUR preparation step. To keep small objects visible at 320x320 we cut each
  1920x1080 underwater photo into 6 tiles (3x2) and had to decide which objects each tile contains.

  Old rule: a tile gets an object's label only if >= 30% of that object's box is inside the tile.

  That works for small objects. It fails for BIG objects: a net, rope or tire that spans several
  tiles puts less than 30% of itself in every tile, so NO tile gets the label, even though the object
  is clearly visible in several of them. In the example photo the net's slices are 17%, 17%, 3%,
  29.9%, 29.9% and 5%: all six tiles were labelled "clean".


HOW BIG THE PROBLEM WAS (audit of all 50,724 SeaClear tiles against the original annotations)
  - 73.5% of SeaClear tiles were labelled clean. Of those "clean" tiles:
      60.9%  truly empty
      19.9%  showed a clearly visible piece of trash (>= 2% of the tile)   <- wrong label
       4.9%  showed only a tiny sliver of trash
      17.9%  showed only animals / ROV parts / plants (correctly clean)
       1.9%  showed an "unknown" object
  - The same rule was used for train, validation AND test, so the wrong labels both taught the
    model the wrong thing (training) and graded it wrongly (val/test): a model that correctly said
    "net" on such a tile was counted as a false alarm.
  - Other datasets: BePLi only 4.7% (small objects, so the rule rarely fails); UCWD/MADOS unaffected.
  - Symptom that led us here: underwater "is there any trash?" AUC was only 0.65 (CNN) / 0.75
    (MobileNetV2), while each underwater class on its own scored ~0.84.


THE FIX (v3, SeaClear only)
  1. Measure on the object's real outline (polygon), not its box.
  2. An object counts for a tile if >= 30% of it is inside OR its visible part covers >= 2% of the tile.
  3. Ambiguous tiles are dropped (neither "trash" nor "clean"): tiles whose only content is a sliver
     of trash (0.1-2% of the tile) or an "unknown" object. Dropped: 2,253 train / 425 val / 541 test.
  4. Same rule for train, val and test. The tile images and the train/val/test split are unchanged
     (checked: the old and new code give an identical split).

  Effect on training labels (tiles containing the class, train split):
      net_plastic    620 -> 2,271  (x3.7)       tire_rubber  2,104 -> 3,351
      tube_cement    841 -> 1,505               rope         1,574 -> 2,069
      metal_debris   307 ->   401               small classes: almost unchanged
      Tiles with trash: 9,270 -> 13,067; clean share 74% -> 61%.


WHAT IT MEANS FOR THE RESULTS
  - v1/v2 results were measured with the old labels; v3 results use the new labels, so compare
    versions on the v3 test labels (re-evaluate old models with 04_evaluate_final if needed).
  - Expected gain: underwater AUC +0.02 to +0.05 (honest estimate, not a promise). Removing the
    ambiguous test tiles alone raised trash-vs-clean AUC by only +0.015; the real gain should come
    from training on correct labels.
  - Not fixed by this: both models also score tiles with fish/urchins/ROV parts as high as trash
    (they learned "object = trash"). That is a model limitation, discussed separately.


ONE-LINE VERSION FOR A SLIDE
  "Cutting detection images into tiles with a 30% overlap rule silently labelled 1 in 5 'clean'
   underwater tiles that actually showed trash (big nets/ropes/tires split across tiles). We
   re-labelled using object outlines and a visible-area rule, and retrained both models."
