-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee
--
-- Impossibility kernel for reference-free IMU mount calibration.
--
-- An IMU on a bone reports orientation  M = D * C * R, the product of the drifting
-- world frame D, the clean bone pose C (what we want), and the constant sensor-to-bone
-- mount R.  This file proves the factorisation is GAUGE-INVARIANT: for any A B, the
-- triple (D*A⁻¹, A*C*B, B⁻¹*R) produces the same M.  So drift D and mount R are not a
-- function of the orientation M alone — no algorithm inverts M ↦ (D,C,R).
--
-- This is the algebraic core of the ~17° mount floor.  The accelerometer pins 2 of the
-- 3 left-gauge DOF (leaving global yaw); the bone's motion excitation pins the right
-- gauge only along axes it actually rotates about (leaving the unexcited mount DOF).
-- Those analytic refinements — and which interventions remove the floor — live in
-- docs/gaps/impossibility-floor.md; the non-identifiability they rest on is proved here.
namespace Sinew.Impossibility

-- A minimal group (the repo carries no mathlib).  SO(3) is the intended instance; the
-- proofs use only the group axioms, so the result holds for any orientation group.
class Grp (G : Type u) where
  mul : G → G → G
  one : G
  inv : G → G
  mul_assoc : ∀ a b c : G, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a : G, mul one a = a
  mul_one : ∀ a : G, mul a one = a
  inv_mul : ∀ a : G, mul (inv a) a = one
  mul_inv : ∀ a : G, mul a (inv a) = one

namespace Grp
variable {G : Type u} [Grp G]

local infixl:70 " ⋆ " => Grp.mul
local postfix:max "⁻¹" => Grp.inv

@[simp] theorem inv_mul_cancel_left (a b : G) : a⁻¹ ⋆ (a ⋆ b) = b := by
  rw [← mul_assoc, inv_mul, one_mul]

@[simp] theorem mul_inv_cancel_left (a b : G) : a ⋆ (a⁻¹ ⋆ b) = b := by
  rw [← mul_assoc, mul_inv, one_mul]

theorem mul_left_cancel {a b c : G} (h : a ⋆ b = a ⋆ c) : b = c := by
  have h2 := congrArg (fun x => a⁻¹ ⋆ x) h
  simpa using h2

-- The IMU orientation measurement.
def meas (D C R : G) : G := D ⋆ C ⋆ R

-- Lemma 1 — gauge invariance: the left gauge A folds into the drift, the right gauge B
-- into the mount, and the measured orientation is unchanged.
theorem gauge (D C R A B : G) :
    meas (D ⋆ A⁻¹) (A ⋆ C ⋆ B) (B⁻¹ ⋆ R) = meas D C R := by
  simp only [meas, mul_assoc, inv_mul_cancel_left, mul_inv_cancel_left]

-- Specialisation used below: a pure drift/clean gauge (B = 1) shifts the recovered drift
-- D ↦ D*g⁻¹ and clean C ↦ g*C while leaving the orientation identical.
theorem gauge_drift (D C R g : G) : meas (D ⋆ g⁻¹) (g ⋆ C) R = meas D C R := by
  simp only [meas, mul_assoc, inv_mul_cancel_left]

-- Theorem — non-identifiability.  No function recovers (drift, clean, mount) from the
-- orientation measurement, as soon as the orientation group is nontrivial (∃ g ≠ 1):
-- the gauge orbit collapses distinct (drift, clean) pairs onto one measurement.
theorem not_identifiable
    (recover : G → G × G × G)
    (hrec : ∀ D C R, recover (meas D C R) = (D, C, R))
    (D C R g : G) (hg : g ≠ one) : False := by
  have h1 : recover (meas D C R) = (D, C, R) := hrec D C R
  have h2 : recover (meas (D ⋆ g⁻¹) (g ⋆ C) R) = (D ⋆ g⁻¹, g ⋆ C, R) := hrec _ _ _
  rw [gauge_drift, h1] at h2                      -- (D, C, R) = (D⋆g⁻¹, g⋆C, R)
  have hD : D = D ⋆ g⁻¹ := congrArg Prod.fst h2
  -- D = D⋆g⁻¹  ⟹  1 = g⁻¹  ⟹  g = 1, contradicting hg
  have hone : one = g⁻¹ := mul_left_cancel (a := D) (by rw [mul_one]; exact hD)
  have hg1 : g = one := by
    have hgi := mul_inv g                          -- g ⋆ g⁻¹ = 1
    rw [← hone, mul_one] at hgi                     -- g ⋆ 1 = 1  ⟹  g = 1
    exact hgi
  exact hg hg1

end Grp
end Sinew.Impossibility
