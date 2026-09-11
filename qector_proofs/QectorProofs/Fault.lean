/-
  QectorProofs.Fault — Fault-distance and correctability.
-/
import QectorProofs.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Finset.Basic

set_option linter.style.header false

namespace QectorProofs.Fault

open Matrix QectorProofs

variable {m n : ℕ}

def weight (e : Fin n → ZMod 2) : ℕ :=
  (Finset.univ.filter (fun i => e i = 1)).card

lemma weight_zero_iff {e : Fin n → ZMod 2} : weight e = 0 ↔ e = 0 := by
  constructor
  · intro h
    funext i
    have hi : i ∉ Finset.univ.filter (fun j => e j = 1) := by
      intro hm
      have hcard : (Finset.univ.filter (fun j => e j = 1)).card = 0 := h
      have hmem2 : Finset.univ.filter (fun j => e j = 1) = ∅ := Finset.card_eq_zero.mp hcard
      rw [hmem2] at hm
      exact Finset.notMem_empty i hm
    simp [Finset.mem_filter] at hi
    have h01 : ∀ a : ZMod 2, a = 0 ∨ a = 1 := by decide
    rcases h01 (e i) with h0 | h1
    · exact h0
    · exact absurd h1 hi
  · intro h
    rw [h]
    simp [weight]

def CorrectForWeight (H : Matrix (Fin m) (Fin n) (ZMod 2))
    (decode : (Fin m → ZMod 2) → Option (Fin n → ZMod 2)) (w : ℕ) : Prop :=
  ∀ (e : Fin n → ZMod 2), weight e ≤ w →
    ∀ s, s = H.mulVec e →
      match decode s with
      | some c => H.mulVec c = s
      | none => False

theorem correctForWeight_zero (H : Matrix (Fin m) (Fin n) (ZMod 2))
    (decode : (Fin m → ZMod 2) → Option (Fin n → ZMod 2))
    (h : decode 0 = some 0) :
    CorrectForWeight H decode 0 := by
  intro e he s hs
  have h0 : weight e = 0 := by omega
  have he0 : e = 0 := weight_zero_iff.mp h0
  rw [he0, Matrix.mulVec_zero] at hs
  rw [hs, h]
  simp [Matrix.mulVec_zero]

end QectorProofs.Fault
