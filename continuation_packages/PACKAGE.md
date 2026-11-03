# VERDE field-adaptation and training-continuation savepoint

Created on 2026-11-03 from the consolidated local savepoint. This package preserves both workstreams in one ZIP:

1. Fixed-camera field-adaptation dataset, review sheets, scripts, and controlled baseline reports.
2. Paused Yellowing training dataset, logs, run settings, scripts, `best.pt`, and resumable `last.pt`.

## Integrity

- ZIP size: 2,801,934,768 bytes
- ZIP SHA-256: `c04c23025cd9d10e7dc9f9a3bb3dcc96ca2e2b8238940377c04e22b5dc0a21f1`
- ZIP entries: 4,917
- Parts: two Git LFS objects listed in `SHA256SUMS.txt`

## Restore

After cloning the repository and running `git lfs pull`:

```powershell
python continuation_packages/restore_continuation_savepoint.py --output-dir C:/VERDE-restore
```

The script verifies both parts, reconstructs and verifies the ZIP, checks for unsafe archive paths, and extracts the savepoint.

Read `STATUS.md` and `TRAINING_STATE.md` inside the restored folder before continuing. Training was stopped when this package was created. The recorded morning run contains 11 completed result rows out of an intended 20 epochs; resume from its copied `last.pt`, not the historical epoch-7 resume script.

The contained validation measurements are controlled fixed-camera development results. They are not proof of general model accuracy or deployment readiness.
