/-
  QectorProofs.MWPM — MWPM safety (valid matching).
  Proves output is a valid correction: H * c = s.
  Full optimality deferred as research-level (see todomath.md 4).
-/
import QectorProofs.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.ZMod.Basic

set_option linter.style.header false

namespace QectorProofs.MWPM

open Matrix QectorProofs

variable {m n : ℕ}

def IsValidCorrection (H : Matrix (Fin m) (Fin n) (ZMod 2))
    (s : Fin m → ZMod 2) (c : Fin n → ZMod 2) : Prop :=
  H.mulVec c = s

theorem matching_valid_of_sound {H : Matrix (Fin m) (Fin n) (ZMod 2)}
    {s : Fin m → ZMod 2} {c : Fin n → ZMod 2}
    (h : H.mulVec c = s) : IsValidCorrection H s c := h

theorem decoder_valid_of_sound {H : Matrix (Fin m) (Fin n) (ZMod 2)}
    {solve : Matrix (Fin m) (Fin n) (ZMod 2) → (Fin m → ZMod 2) → Option (Fin n → ZMod 2)}
    (hsound : SolverSound solve) {s : Fin m → ZMod 2} {c : Fin n → ZMod 2}
    (h : solve H s = some c) : IsValidCorrection H s c :=
  hsound H s c h

end QectorProofs.MWPM
