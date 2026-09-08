# Third-Party Notices

This repository vendors self-contained source snapshots needed by the reproduction so setup
does not clone other repositories. Some snapshots intentionally retain adjacent upstream
modules to preserve their import and configuration contracts. Vendored files retain upstream
copyright headers.

| Component | Source revision | License | Location |
|---|---|---|---|
| RoboMeter policy learning | `d5fd0b3` plus audited adapter changes | MIT | `robometer_policy_learning/` |
| RoboMeter | `a669dff` | MIT | `vendor/robometer/` |
| RLinf runtime subset | `a3816b5` | Apache-2.0 | `vendor/rlinf/` |
| OpenPI runtime | `b0e3cfb` | Apache-2.0 | `vendor/openpi/` |
| LIBERO | `87edbd1` | MIT | `vendor/libero/` |

License texts supplied by the upstream checkouts are available in `LICENSES/`. The RoboMeter
checkout did not contain a standalone license file at the pinned revision; its README and
package metadata both declare MIT. Model checkpoints and simulator assets are not redistributed
and remain subject to their upstream licenses.
