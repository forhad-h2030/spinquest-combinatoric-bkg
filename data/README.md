## Output branches saved (selected dimuon: μ⁺/μ⁻ and associated tracks)

These branches correspond to the **two muon tracks (μ⁺ and μ⁻)** of the **selected reconstructed dimuon**
candidate in the event.

 ### 1) Target-evaluated muon momenta (from the dimuon object)

These quantities are the **μ⁺ and μ⁻ three-momenta evaluated at the target plane**, as stored in the `SRecDimuon` object.

In this analysis, the target-level muon momenta are accessed directly via:
- `dim->p_pos_target`  &nbsp;— μ⁺ momentum at the target  
- `dim->p_neg_target`  &nbsp;— μ⁻ momentum at the target  

The **dimuon momentum at the target** is then constructed by summing the two:
```cpp
dd.mom_target = dim->p_pos_target + dim->p_neg_target;
```

| Branch name | Type | Meaning |
|---|---:|---|
| `rec_dimu_mu_pos_px` | `double` | μ⁺ target momentum \(p_x\) (from `p_pos_target.Px()`) |
| `rec_dimu_mu_pos_py` | `double` | μ⁺ target momentum \(p_y\) (from `p_pos_target.Py()`) |
| `rec_dimu_mu_pos_pz` | `double` | μ⁺ target momentum \(p_z\) (from `p_pos_target.Pz()`) |
| `rec_dimu_mu_neg_px` | `double` | μ⁻ target momentum \(p_x\) (from `p_neg_target.Px()`) |
| `rec_dimu_mu_neg_py` | `double` | μ⁻ target momentum \(p_y\) (from `p_neg_target.Py()`) |
| `rec_dimu_mu_neg_pz` | `double` | μ⁻ target momentum \(p_z\) (from `p_neg_target.Pz()`) |
---

### 2) Track state at Station 1 (from the associated reconstructed tracks)

These quantities come from the reconstructed track objects associated with the dimuon (μ⁺ and μ⁻),
evaluated at **Station 1**.

| Branch name | Type | Meaning |
|---|---:|---|
| `rec_track_pos_x_st1`  | `double` | μ⁺ track x-position at Station 1 |
| `rec_track_neg_x_st1`  | `double` | μ⁻ track x-position at Station 1 |
| `rec_track_pos_px_st1` | `double` | μ⁺ track \(p_x\) at Station 1 |
| `rec_track_neg_px_st1` | `double` | μ⁻ track \(p_x\) at Station 1 |

---

### 3) Track vertex position (from the associated reconstructed tracks)

These are the reconstructed vertex coordinates of the μ⁺ and μ⁻ tracks.

| Branch name | Type | Meaning |
|---|---:|---|
| `rec_track_pos_vx` | `double` | μ⁺ track vertex x |
| `rec_track_pos_vy` | `double` | μ⁺ track vertex y |
| `rec_track_pos_vz` | `double` | μ⁺ track vertex z |
| `rec_track_neg_vx` | `double` | μ⁻ track vertex x |
| `rec_track_neg_vy` | `double` | μ⁻ track vertex y |
| `rec_track_neg_vz` | `double` | μ⁻ track vertex z |

