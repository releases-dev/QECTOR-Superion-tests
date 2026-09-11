/-
  QectorProofs.Phi — phi-kernel and LLR identities.
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic

set_option linter.style.header false

namespace QectorProofs.Phi

open Real

noncomputable def phi (x : ℝ) : ℝ :=
  if x = 0 then 0 else -Real.log (Real.tanh (x / 2))

noncomputable def llr (p : ℝ) : ℝ :=
  Real.log ((1 - p) / p)

lemma llr_zero_half : llr 0.5 = 0 := by
  unfold llr
  have h : (1 - (0.5 : ℝ)) / 0.5 = 1 := by norm_num
  rw [h, Real.log_one]

lemma llr_pos_of_lt_half {p : ℝ} (hp0 : 0 < p) (hp1 : p < 0.5) : 0 < llr p := by
  unfold llr
  have h1 : 1 < (1 - p) / p := by
    rw [one_lt_div hp0]
    linarith
  exact Real.log_pos h1

lemma llr_neg_of_gt_half {p : ℝ} (hp0 : p < 1) (hp1 : 0.5 < p) : llr p < 0 := by
  unfold llr
  have h1 : 0 < (1 - p) / p := by
    apply div_pos <;> linarith
  have h2 : (1 - p) / p < 1 := by
    rw [div_lt_one (by linarith)]
    linarith
  exact Real.log_neg h1 h2

end QectorProofs.Phi
