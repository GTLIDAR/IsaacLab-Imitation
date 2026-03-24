# Local Data Layout

This directory is for local and generated motion assets. Most of it remains git-ignored on purpose.

Tracked manifests and templates live under `source/isaaclab_imitation/isaaclab_imitation/manifests/`:

- `g1_default_manifest.json`: tracked 3-motion debug manifest
- `g1_dance102_manifest.json`: tracked single-motion `dance_102` manifest
- `g1_lafan1_manifest.template.json`: tracked template for a full local G1 LAFAN1 manifest

Expected local paths:

- `data/lafan1/raw/g1/`: downloaded or source CSV motions
- `data/lafan1/npz/g1/`: converted G1 NPZ motions
- `data/lafan1/manifests/g1_lafan1_manifest.json`: full local manifest
- `data/lafan1/manifests/g1_debug_manifest.json`: smaller local subset manifest
- `data/dance_102/G1_Take_102.bvh_60hz.npz`: local `dance_102` NPZ

Common flows:

1. Download the Hugging Face G1 dataset and prepare NPZ plus a full manifest:

```bash
./scripts/download_g1_lafan1_data.sh
```

Equivalent lower-level Python command:

```bash
conda run -n SkillLearning python scripts/setup_lafan1_dataset.py \
    --prepare-npz --headless
```

2. If NPZ files already exist, regenerate the full manifest:

```bash
conda run -n SkillLearning python scripts/write_lafan1_npz_manifest.py \
    --npz_dir data/lafan1/npz/g1 \
    --manifest_path data/lafan1/manifests/g1_lafan1_manifest.json
```

3. If you want to hand-edit a manifest instead of generating one, copy the tracked template:

```bash
mkdir -p data/lafan1/manifests
cp source/isaaclab_imitation/isaaclab_imitation/manifests/g1_lafan1_manifest.template.json \
   data/lafan1/manifests/g1_lafan1_manifest.json
```

After copying, replace the placeholder motion names and paths with your local NPZ files.
