-- SPDX-License-Identifier: MIT
-- SinewMountDrift — the calibration theory (mountdrift cluster): the
-- machine-checked impossibility proof (gauge non-identifiability of M = D·C·R,
-- the reference-free mount floor).  Self-contained: no dependencies.
import Lake
open Lake DSL

package "SinewMountDrift" where
  version := v!"0.1.0"

lean_lib SinewMountDrift where
  srcDir := "."
  globs  := #[Glob.one `Sinew.Impossibility]
