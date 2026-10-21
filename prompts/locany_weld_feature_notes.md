# Weld defect visual cues for prompt experiments

These are observations from a small set of fold0 validation GT/prediction panels, not universal physical definitions. The GT labels take priority when wording a dataset-specific prompt. The V6 text uses concise cues because V5 often copied longer class definitions into its output.

| Label | Visible pattern in reviewed GT boxes | Frequent model confusion |
| --- | --- | --- |
| `pore` | Small, isolated void or dark pinhole, generally with a tight local box. | The original model uses `pore` for many unrelated dark spots and for boxes labeled `deposit` or `stain` by GT. Darkness alone is inadequate. |
| `deposit` | Compact localized spot or small irregular cluster on the bead. Some reviewed GT examples are dark round spots with a bright center or rim. | Those round GT deposits are frequently predicted as `pore`; V4/V5 often omit the entire class. A grayscale image does not reliably show physical height, so “raised material” alone is a weak visual instruction. |
| `discontinuity` | Larger, often elongated or edge-connected interruption of the continuous bead region; reviewed boxes are frequently wider than isolated dots. | Small pieces get labeled `pore`; the original model also produces whole-image boxes for this class. |
| `stain` | Irregular local patch or texture/tone change, sometimes elongated and sometimes containing several small marks. | A small dark component inside a stain is often predicted as `pore`; the model may miss the larger GT patch. |

Reviewed examples include `25_400.jpg`, `25_396.jpg`, `25_615.jpg`, `25_612.jpg`, `26_278.jpg`, `26_318.jpg`, `26_288.jpg`, and `26_540.jpg`. Several boxes are visually ambiguous, and these cues should be treated as testable hypotheses. Compare class-aware and class-agnostic matching on the same validation images; do not use the held-out test set to tune prompt wording.
