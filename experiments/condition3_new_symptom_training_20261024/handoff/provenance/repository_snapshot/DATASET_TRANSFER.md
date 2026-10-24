# Dataset retrieval

The private repository contains the required source folders as split Git LFS
ZIP archive parts under `dataset_sources/`:

1. `verde_v2.zip.part001` and `part002` - five-class YOLO source split,
   labels, and metadata.
2. `verde_v2_anylabeling.zip.part001` and `part002` - decoded JPEG
   counterparts used when a source image is HEIC.

The archives are split because GitHub Free limits each Git LFS object to
2 GB. Do not rename or edit the parts.

## Retrieve and restore

Install Git LFS before cloning. After cloning, retrieve the full LFS objects:

```powershell
git lfs install
git lfs pull
```

Verify, join, and extract both archives:

```powershell
python src\restore_dataset_archives.py --dest "D:\VERDE_AI_DATA"
```

This produces:

```text
D:\VERDE_AI_DATA\
|-- _archives\
|-- verde_v2\
`-- verde_v2_anylabeling\
```

Use `--remove-archives` if disk space is limited and the reconstructed ZIP
files should be deleted after successful extraction. The LFS parts remain in
the cloned repository.

## Recorded source checks

| Folder | Files | Bytes |
| --- | ---: | ---: |
| `verde_v2` | 3,175 | 2,308,825,646 |
| `verde_v2_anylabeling` | 3,153 | 2,675,903,637 |

The restore script checks every part and reconstructed archive against
`dataset_sources/SHA256SUMS.txt`. These checks validate transfer completeness;
they do not prove annotation quality or independent plant identities.

## Prepared three-class dataset

`src/prepare_dataset.py` creates an isolated three-class dataset. It excludes
images containing Harvest-Ready or Pest Detected labels instead of guessing
those objects into Healthy. Expected image counts:

| Split | Images |
| --- | ---: |
| Train | 760 |
| Validation | 207 |
| Test | 200 |

Keep the generated `dataset/` folder local. It is ignored by Git.
