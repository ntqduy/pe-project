# 11. Data lineage

```text
raw CTPA/report/EHR
  -> eligibility + integrity + official patient split audit
  -> manifests + normalized CT cache
  -> segmentation masks + preview/QC
  -> semantic ROI + rigid ROI8 + preview/QC
  -> DAPT -> C0 -> C_silver/C_diagnosis
  -> diagnosis/prognosis predictions
  -> validation threshold -> locked internal/external test metrics
```

Định danh data là patient/study/split/path; silver thêm report/hash/target; ROI thêm
code/name/source; model thêm experiment/config/checkpoint/source hashes. EHR preprocessing
fit train-only, prognosis có một index study/patient và test không dùng để tune.

Để tái lập cần giữ raw version, manifest/hash, preprocessing fingerprint, resolved config,
git/environment, seed, checkpoint+sidecar và predictions. `result.json` một mình không đủ.
