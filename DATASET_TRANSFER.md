# Dataset transfer

The repository does not contain the VERDE images. On the original computer,
the required source folders are under:

`C:\Users\josephemmanuel\Desktop\VERDE AI\D_training\`

Transfer these folders to the new device using an external drive or another
private large-file transfer method:

1. `verde_v2` - five-class YOLO source split, labels, and metadata.
2. `verde_v2_anylabeling` - decoded JPEG counterparts used when a source image
   is HEIC.

Recorded source-package checks:

| Folder | Files | Bytes |
| --- | ---: | ---: |
| `verde_v2` | 3,175 | 2,308,825,646 |
| `verde_v2_anylabeling` | 3,153 | 2,675,903,637 |

After copying, confirm the file counts and byte totals on the new device before
running `src/prepare_dataset.py`. These totals validate transfer completeness;
they do not prove annotation quality or independent plant identities.

The preparation script reads the source folders and creates an isolated
three-class dataset. It excludes any image containing Harvest-Ready or Pest
Detected labels instead of guessing those objects into Healthy. The expected
prepared counts are:

| Split | Images |
| --- | ---: |
| Train | 760 |
| Validation | 207 |
| Test | 200 |

Keep the generated `dataset/` folder local. It is ignored by Git and must not be
committed unless image ownership, licensing, privacy, and Git hosting limits
have been reviewed separately.

