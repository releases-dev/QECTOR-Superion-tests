/-
  QectorProofs.Basic

  Formal counterpart to the bit-packed GF(2) Gauss-Jordan solver in
  `src/gf2.rs` (qector-ionq decoder, Guillaume Lessard / qector-decoder-v3).

  Rust target: `solve_from_checks` in `src/gf2.rs`.

    pub(crate) fn solve_from_checks(
        check_to_qubits: &[Vec<u32>],
        n_qubits: usize,
        syndrome: &[u8],
    ) -> Option<Vec<u8>>

  Semantics: given a parity-check adjacency (row i = the qubits check i
  touches) and a target syndrome, find x with H * x = syndrome over GF(2),
  or report None if no such x exists.

  We model H directly as a Matrix (Fin m) (Fin n) (ZMod 2) rather than as
  an adjacency list; `H i j = 1` iff qubit j appears in check i's row,
  matching how `solve_from_checks` builds its packed rows from
  `check_to_qubits`.
-/
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.ZMod.Basic
import Mathlib.LinearAlgebra.Matrix.ToLin

-- This is a proprietary project file, not a Mathlib contribution, so the
-- Mathlib house-style header linter (copyright/authors line format) does
-- not apply here.
set_option linter.style.header false

open Matrix

namespace QectorProofs

variable {m n : ℕ}

/-- The GF(2) parity-check matrix, as used by `solve_from_checks`. -/
abbrev CheckMatrix (m n : ℕ) := Matrix (Fin m) (Fin n) (ZMod 2)

/-- A syndrome is *reachable* under `H` if some correction reproduces it.
    This is the success side of `solve_from_checks`: it returns `Some x`
    exactly when the syndrome is reachable, and that `x` witnesses it. -/
def Reachable (H : CheckMatrix m n) (s : Fin m → ZMod 2) : Prop :=
  ∃ x : Fin n → ZMod 2, H.mulVec x = s

/-- Soundness contract of `solve_from_checks`: **if** the solver returns
    `some x`, that `x` must satisfy `H * x = syndrome`.

    This is exactly the property `test_solve_from_checks_reproduces_reachable_syndrome`
    checks by fuzzing (2000 random instances) in `src/gf2.rs` — here we
    state it as a fact about *any* correctly-implemented solver, not a
    sampled test. -/
def SolverSound (solve : CheckMatrix m n → (Fin m → ZMod 2) → Option (Fin n → ZMod 2)) : Prop :=
  ∀ (H : CheckMatrix m n) (s : Fin m → ZMod 2) (x : Fin n → ZMod 2),
    solve H s = some x → H.mulVec x = s

/-- Completeness contract: **if** the syndrome is reachable, the solver
    must not give up (`none`). Together with `SolverSound`, this pins
    down `solve_from_checks`'s `None` case: it fires exactly on
    unreachable syndromes, matching `test_unreachable_syndrome_is_none`. -/
def SolverComplete (solve : CheckMatrix m n → (Fin m → ZMod 2) → Option (Fin n → ZMod 2)) : Prop :=
  ∀ (H : CheckMatrix m n) (s : Fin m → ZMod 2),
    Reachable H s → (solve H s).isSome

/-- Sanity check that the toolchain and imports are wired correctly:
    the all-zero syndrome is always reachable (witnessed by x = 0), for
    any H. This matches the `"zero->zero"` / `"erasure all->zero"` cases
    exercised in `qector_ionq_math_verification.py`. -/
theorem reachable_zero (H : CheckMatrix m n) : Reachable H (fun _ => 0) := by
  refine ⟨0, ?_⟩
  funext i
  simp [Matrix.mulVec_zero]

/-- If a syndrome is reachable, so is the syndrome shifted by any vector
    already in the image of H (linearity over GF(2)/`ZMod 2`) — the
    algebraic fact underlying why fault patterns differing by a stabilizer
    decode identically. -/
theorem reachable_add {H : CheckMatrix m n} {s t : Fin m → ZMod 2}
    (hs : Reachable H s) (ht : Reachable H t) :
    Reachable H (s + t) := by
  obtain ⟨x, hx⟩ := hs
  obtain ⟨y, hy⟩ := ht
  refine ⟨x + y, ?_⟩
  rw [Matrix.mulVec_add, hx, hy]

/-- Two corrections for the *same* syndrome differ by a vector in the
    kernel of `H` — i.e. by an undetectable stabilizer-like error. This is
    the algebraic fact behind `"determinism"` in
    `qector_ionq_math_verification.py`: that test only checks the *solver*
    is a deterministic function (same input, same output), which is
    trivially true of any Rust function; this lemma is the stronger,
    solver-independent statement about the underlying linear system
    itself — any two valid corrections for one syndrome agree modulo
    `H`'s kernel, regardless of which one a particular algorithm picks. -/
theorem sub_mem_ker_of_mulVec_eq {H : CheckMatrix m n} {s : Fin m → ZMod 2}
    {x y : Fin n → ZMod 2} (hx : H.mulVec x = s) (hy : H.mulVec y = s) :
    H.mulVec (x - y) = 0 := by
  rw [Matrix.mulVec_sub, hx, hy, sub_self]

/-!
  ## Main correctness theorem (in progress)

  This is the real target: a concrete Gauss-Jordan elimination procedure
  over `ZMod 2`, proved both sound and complete. `gf2_gauss_jordan` below
  is a placeholder signature; the elimination algorithm itself
  (pivot search, row reduction, back-substitution) still needs to be
  written and proved against `SolverSound` / `SolverComplete`.

  This is the Lean-side counterpart of `solve_from_checks` /
  `solve_selected_with_free` in `src/gf2.rs` and is intentionally left
  as `sorry` rather than faked: closing it is the actual formal-proof
  deliverable, not a one-shot.
-/

/-- A solver via classical choice over `∃ x, H.mulVec x = s`.

    What this is and isn't: this *is* a genuine, fully correct
    (sound + complete, proved below, no `sorry`) solver for the abstract
    spec `SolverSound` / `SolverComplete` — it proves a correct decision
    procedure exists and matches the contract exactly.

    It is *not* a formalization of the actual bit-packed Gauss-Jordan
    elimination in `solve_from_checks` (`src/gf2.rs`) — it picks a
    witness via classical choice rather than by pivoting/row-reducing,
    so it certifies that the linear-algebra problem is well-posed and
    solvable exactly when expected, not that the Rust code's specific
    elimination algorithm is correct. Closing that remaining gap is
    real, separate work — see the TODO below. -/
noncomputable def gf2_gauss_jordan (H : CheckMatrix m n) (s : Fin m → ZMod 2) :
    Option (Fin n → ZMod 2) :=
  if h : ∃ x : Fin n → ZMod 2, H.mulVec x = s then some h.choose else none

theorem gf2_gauss_jordan_sound : SolverSound (@gf2_gauss_jordan m n) := by
  intro H s x hx
  unfold gf2_gauss_jordan at hx
  by_cases h : ∃ y : Fin n → ZMod 2, H.mulVec y = s
  · rw [dif_pos h] at hx
    have hxy : h.choose = x := Option.some.inj hx
    rw [← hxy]
    exact h.choose_spec
  · rw [dif_neg h] at hx
    exact absurd hx (by simp)

theorem gf2_gauss_jordan_complete : SolverComplete (@gf2_gauss_jordan m n) := by
  intro H s hs
  unfold gf2_gauss_jordan
  unfold Reachable at hs
  rw [dif_pos hs]
  exact Option.isSome_some

/-!
  ## TODO: refine to the actual Rust algorithm

  `gf2_gauss_jordan` above satisfies the abstract spec but via classical
  choice, not by mirroring `solve_from_checks`'s real pivoting/row-echelon
  steps. The next real milestone is mechanizing that elimination procedure
  as an explicit, computable recursive function and proving it agrees
  with `gf2_gauss_jordan` — still open, substantial work, intentionally
  not faked here.
-/

end QectorProofs
