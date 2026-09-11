/-
  QectorProofs.Elimination — Explicit Gauss-Jordan over GF(2), proved sound & complete.

  Mirrors `solve_from_checks` in `src/gf2.rs` as a clean mathematical spec.
-/
import QectorProofs.Basic

set_option linter.style.header false

namespace QectorProofs

variable {m : ℕ}

def gf2_solve : ∀ (n : ℕ) {m : ℕ}, Matrix (Fin m) (Fin n) (ZMod 2) →
    (Fin m → ZMod 2) → Option (Fin n → ZMod 2)
  | 0, _, _, s => if s = 0 then some Fin.elim0 else none
  | (k + 1), m, H, s =>
      let col0 : Fin m → ZMod 2 := fun i => H i 0
      let H' : Matrix (Fin m) (Fin k) (ZMod 2) := fun i j => H i j.succ
      if h : ∃ i, col0 i = 1 then
        let p := h.choose
        let hp : col0 p = 1 := h.choose_spec
        let H'' : Matrix (Fin m) (Fin k) (ZMod 2) :=
          fun i j => if i = p then 0 else H' i j + col0 i * H' p j
        let s'' : Fin m → ZMod 2 := fun i => if i = p then 0 else s i + col0 i * s p
        match gf2_solve k H'' s'' with
        | some x' => some (Fin.cons (s p + H' p ⬝ᵥ x') x')
        | none => none
      else
        match gf2_solve k H' s with
        | some x' => some (Fin.cons (0 : ZMod 2) x')
        | none => none

private lemma mulVec_fin_zero {m : ℕ} (H : Matrix (Fin m) (Fin 0) (ZMod 2))
    (x : Fin 0 → ZMod 2) : H.mulVec x = 0 := by
  funext i
  simp [Matrix.mulVec, dotProduct]

private lemma mulVec_succ_split {m k : ℕ}
    (H : Matrix (Fin m) (Fin (k + 1)) (ZMod 2))
    (y : Fin (k + 1) → ZMod 2) (i : Fin m) :
    H.mulVec y i = H i 0 * y 0 +
      Matrix.mulVec (fun i' (j : Fin k) => H i' j.succ) (Fin.tail y) i := by
  simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Fin.tail]

private lemma zmod2_row_iff {a b c d e : ZMod 2} :
    a * (b + c) + d = e ↔ d + a * c = e + a * b := by
  revert a b c d e; decide

private lemma zmod2_iso1_iff {a b c : ZMod 2} :
    a + b = c ↔ a = c + b := by
  revert a b c; decide

private lemma zmod2_eq_zero_or_one : ∀ a : ZMod 2, a = 0 ∨ a = 1 := by
  decide

theorem gf2_solve_correct : ∀ (n : ℕ) {m : ℕ}
    (H : Matrix (Fin m) (Fin n) (ZMod 2)) (s : Fin m → ZMod 2),
    (∀ x, gf2_solve n H s = some x → H.mulVec x = s) ∧
    (Reachable H s → (gf2_solve n H s).isSome) := by
  intro n
  induction n with
  | zero =>
    intro m H s
    constructor
    · intro x hx
      unfold gf2_solve at hx
      by_cases h : s = 0
      · rw [if_pos h] at hx
        have hxe : Fin.elim0 = x := Option.some.inj hx
        rw [← hxe, mulVec_fin_zero, h]
      · rw [if_neg h] at hx
        exact absurd hx (by simp)
    · rintro ⟨x, hx⟩
      have hs0 : s = 0 := by rw [← hx, mulVec_fin_zero]
      unfold gf2_solve
      rw [if_pos hs0]
      exact Option.isSome_some
  | succ k ih =>
    intro m H s
    -- split on whether first column has a pivot
    by_cases hp : ∃ i, (fun i => H i 0) i = 1
    · -- pivot exists
      let p := hp.choose
      have hp1 : H p 0 = 1 := hp.choose_spec
      let H' : Matrix (Fin m) (Fin k) (ZMod 2) := fun i j => H i j.succ
      let H'' : Matrix (Fin m) (Fin k) (ZMod 2) :=
        fun i j => if i = p then 0 else H' i j + H i 0 * H' p j
      let s'' : Fin m → ZMod 2 := fun i => if i = p then 0 else s i + H i 0 * s p
      have heq : gf2_solve (k+1) H s =
          (gf2_solve k H'' s'').map (fun x' => Fin.cons (s p + H' p ⬝ᵥ x') x') := by
        unfold gf2_solve
        rw [dif_pos hp]
      rw [heq]
      have hsplit : ∀ (y : Fin (k+1) → ZMod 2) (i : Fin m),
          H.mulVec y i = H i 0 * y 0 + H'.mulVec (Fin.tail y) i := by
        intro y i
        exact mulVec_succ_split H y i
      have hH''_spec : ∀ (i : Fin m) (x' : Fin k → ZMod 2),
          H''.mulVec x' i = if i = p then 0 else H'.mulVec x' i + H i 0 * H'.mulVec x' p := by
        intro i x'
        by_cases hip : i = p
        · simp [H'', hip]
        · simp only [H'', hip, ite_false]
          simp [Matrix.mulVec, dotProduct]
          rw [← Finset.sum_add_distrib]
          congr 1; funext j; ring
      have hiff : ∀ (x0 : ZMod 2) (x' : Fin k → ZMod 2),
          H.mulVec (Fin.cons x0 x') = s ↔
            (x0 = s p + H' p ⬝ᵥ x' ∧ H''.mulVec x' = s'') := by
        intro x0 x'
        constructor
        · intro heq
          have hrow : ∀ i, H i 0 * x0 + H'.mulVec x' i = s i := by
            intro i; have hi := congrFun heq i; rw [hsplit (Fin.cons x0 x') i] at hi; exact hi
          have hrp := hrow p
          rw [hp1, one_mul] at hrp
          have hx0 : x0 = s p + H' p ⬝ᵥ x' := by
            have := zmod2_iso1_iff (a:=x0) (b:=H' p ⬝ᵥ x') (c:=s p)
            exact this.mp hrp
          refine ⟨hx0, ?_⟩
          funext i
          by_cases hip : i = p
          · subst hip; simp [H'', s'']
          · have hri := hrow i
            rw [hx0] at hri
            have hmul := hH''_spec i x'
            simp only [hip, ite_false] at hmul
            rw [hmul] at hri ⊢
            have hs'' : s'' i = s i + H i 0 * s p := by simp [s'', hip]
            rw [hs''] at hri ⊢
            exact (zmod2_row_iff (a:=H i 0) (b:=s p) (c:=H' p ⬝ᵥ x') (d:=H'.mulVec x' i) (e:=s i)).mp hri
        · rintro ⟨hx0, heq''⟩
          funext i
          have hrow : H i 0 * x0 + H'.mulVec x' i = s i := by
            by_cases hip : i = p
            · subst hip
              rw [hx0, hp1, one_mul]
              -- (s p + H' p ⬝ᵥ x') + H' p ⬝ᵥ x' = s p  in ZMod 2 (b+b=0)
              have h1 : (s p + H' p ⬝ᵥ x') + H' p ⬝ᵥ x' = s p := by
                have h2 : (H' p ⬝ᵥ x' : ZMod 2) + H' p ⬝ᵥ x' = 0 := by
                  have : (2 : ZMod 2) = 0 := by decide
                  calc H' p ⬝ᵥ x' + H' p ⬝ᵥ x' = 2 * (H' p ⬝ᵥ x') := by ring
                    _ = 0 * (H' p ⬝ᵥ x') := by rw [this]
                    _ = 0 := by ring
                calc (s p + H' p ⬝ᵥ x') + H' p ⬝ᵥ x'
                    = s p + (H' p ⬝ᵥ x' + H' p ⬝ᵥ x') := by ring
                  _ = s p + 0 := by rw [h2]
                  _ = s p := by ring
              exact h1
            · have hrow'' := congrFun heq'' i
              have hmul := hH''_spec i x'
              simp only [hip, ite_false] at hmul
              rw [hmul] at hrow''
              have hs'' : s'' i = s i + H i 0 * s p := by simp [s'', hip]
              rw [hs''] at hrow''
              rw [hx0]
              exact (zmod2_row_iff (a:=H i 0) (b:=s p) (c:=H' p ⬝ᵥ x') (d:=H'.mulVec x' i) (e:=s i)).mpr hrow''
          rw [hsplit (Fin.cons x0 x') i]
          exact hrow
      obtain ⟨ih_sound, ih_complete⟩ := ih H'' s''
      constructor
      · intro x hx
        rw [Option.map_eq_some_iff] at hx
        obtain ⟨x', hx', rfl⟩ := hx
        exact (hiff _ _).mpr ⟨rfl, ih_sound x' hx'⟩
      · rintro ⟨x, hx⟩
        have hx_eq : x = Fin.cons (x 0) (Fin.tail x) := by
          funext j; cases j using Fin.cases <;> simp [Fin.cons, Fin.tail]
        rw [hx_eq] at hx
        have hpair := (hiff (x 0) (Fin.tail x)).mp hx
        have hr' : Reachable H'' s'' := ⟨Fin.tail x, hpair.2⟩
        have hsome := ih_complete hr'
        rw [Option.isSome_map]; exact hsome
    · -- no pivot, first column all zero
      let H' : Matrix (Fin m) (Fin k) (ZMod 2) := fun i j => H i j.succ
      have heq : gf2_solve (k+1) H s = (gf2_solve k H' s).map (Fin.cons 0) := by simp [gf2_solve, hp]
      rw [heq]
      have hp' : ∀ i, H i 0 ≠ 1 := by push Not at hp; exact hp
      have hzero : ∀ i, H i 0 = 0 := by
        intro i
        have h1 := hp' i
        have h2 := zmod2_eq_zero_or_one (H i 0)
        rcases h2 with h0 | h1'
        · exact h0
        · exact absurd h1' h1
      have hsplit : ∀ (y : Fin (k+1) → ZMod 2) (i : Fin m),
          H.mulVec y i = H'.mulVec (Fin.tail y) i := by
        intro y i
        have h := mulVec_succ_split H y i
        rw [hzero i, zero_mul, zero_add] at h
        exact h
      obtain ⟨ih_sound, ih_complete⟩ := ih H' s
      constructor
      · intro x hx
        rw [Option.map_eq_some_iff] at hx
        obtain ⟨x', hx', rfl⟩ := hx
        funext i
        rw [hsplit]
        exact congrFun (ih_sound x' hx') i
      · rintro ⟨x, hx⟩
        have hr' : Reachable H' s := by
          refine ⟨Fin.tail x, ?_⟩
          funext i
          have hi : H.mulVec x i = s i := congrFun hx i
          have hx_eq : x = Fin.cons (x 0) (Fin.tail x) := by
            funext j
            cases j using Fin.cases
            · simp
            · simp [Fin.tail]
          rw [hx_eq] at hi
          have hi2 : H.mulVec (Fin.cons (x 0) (Fin.tail x)) i = H'.mulVec (Fin.tail x) i :=
            hsplit (Fin.cons (x 0) (Fin.tail x)) i
          rw [hi2] at hi
          exact hi
        have hsome := ih_complete hr'
        rw [Option.isSome_map]; exact hsome

end QectorProofs
