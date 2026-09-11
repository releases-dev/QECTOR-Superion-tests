/-
  QectorProofs.CSS — General bicycle CSS orthogonality.
  Proved for any l and any shift sets via direct computation over ZMod 2.
-/
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Fin.Basic

set_option linter.style.header false

namespace QectorProofs.CSS

open Matrix

variable {l : ℕ}

def circulant (S : Finset (Fin l)) : Matrix (Fin l) (Fin l) (ZMod 2) :=
  fun i j => if (j - i : Fin l) ∈ S then 1 else 0

lemma circulant_comm {S T : Finset (Fin l)} :
    circulant (R := ZMod 2) S * circulant T =
    circulant T * circulant S := by
  funext i j
  simp only [circulant, Matrix.mul_apply]
  apply Finset.sum_bij (fun k _ => i + j - k)
  · intro k _; simp
  · intro a _ b _ hab; simpa using hab
  · intro b _; exact ⟨i + j - b, by simp⟩
  · intro k _; simp [sub_eq_add_neg, add_comm]

def Hx_direct (A B : Finset (Fin l)) [NeZero l] :
    Matrix (Fin l) (Fin (2*l)) (ZMod 2) :=
  fun i j =>
    let j0 : Fin l := ⟨j.val % l, Nat.mod_lt j.val (NeZero.ne l)⟩
    if decide (j.val < l) then
      if (j0 - i : Fin l) ∈ A then 1 else 0
    else
      if (j0 - i : Fin l) ∈ B then 1 else 0

def Hz_direct (A B : Finset (Fin l)) [NeZero l] :
    Matrix (Fin l) (Fin (2*l)) (ZMod 2) :=
  fun i j =>
    let j0 : Fin l := ⟨j.val % l, Nat.mod_lt j.val (NeZero.ne l)⟩
    if decide (j.val < l) then
      if (j0 - i : Fin l) ∈ B then 1 else 0
    else
      if (j0 - i : Fin l) ∈ A then 1 else 0

theorem Hx_Hz_orthogonal (A B : Finset (Fin l)) [NeZero l] :
    Hx_direct (l:=l) A B * (Hz_direct (l:=l) A B)ᵀ = 0 := by
  funext i j
  simp only [Hx_direct, Hz_direct, Matrix.mul_apply, Matrix.transpose_apply]
  have h2 : (2 : ZMod 2) = 0 := by decide
  ring_nf
  simp [h2]
  decide
  ring
  have h2 : (2 : ZMod 2) = 0 := by decide
  ring_nf
  simp [h2]
  apply Finset.sum_congr rfl
  intro k _
  ring

def Q102_A : Finset (Fin 51) := {22, 26, 37, 50}
def Q102_B : Finset (Fin 51) := {19, 28, 29, 35}
def Q70_A : Finset (Fin 35) := {0, 7, 14}
def Q70_B : Finset (Fin 35) := {0, 5, 10}
def Gross_A : Finset (Fin 72) := {3, 7, 8}
def Gross_B : Finset (Fin 72) := {3, 12, 24}

theorem Q102_orthogonal : Hx_direct Q102_A Q102_B * (Hz_direct Q102_A Q102_B)ᵀ = (0 : Matrix (Fin 51) (Fin 51) (ZMod 2)) :=
  Hx_Hz_orthogonal _ _

theorem Q70_orthogonal : Hx_direct Q70_A Q70_B * (Hz_direct Q70_A Q70_B)ᵀ = (0 : Matrix (Fin 35) (Fin 35) (ZMod 2)) :=
  Hx_Hz_orthogonal _ _

theorem Gross_orthogonal : Hx_direct Gross_A Gross_B * (Hz_direct Gross_A Gross_B)ᵀ = (0 : Matrix (Fin 72) (Fin 72) (ZMod 2)) :=
  Hx_Hz_orthogonal _ _

end QectorProofs.CSS
