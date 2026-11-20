# Codex working instructions

## AIHC training

- Read `deploy/aihc/README.md` before working on cloud training.
- Development is CPU-only. Allocate GPUs through AIHC training jobs; do not resize the development machine to add GPUs.
- Local settings are `deploy/aihc/site.local.json` and `profiles.local.json`. These are ignored by Git. Credentials are outside the repository at `~/.aihc/nyx.yaml`; never print, copy into CFS/images, or commit them.
- Default training topology is **one worker with 8 GPUs**, using torchrun. Eight worker replicas would request eight machines and is incorrect for this setup.
- Code and the Python environment live on the development machine SSD at `/root/workspace/HumanoidSkill`. Datasets, pretrained weights, caches and run outputs live under `/mnt/cfs/usr/nyx`.
- See `deploy/aihc/DATA.md` for the user's residual_rl datasets and downloaded checkpoint catalog. The derived keypoint NPZ files are not SuperMimic motion-library inputs; do not fill the SuperMimic data profile with them. Keep archived checkpoint provenance unchanged and use CFS paths in runtime profiles.
- Use `python3 deploy/aihc/task.py doctor PROFILE` and `prepare PROFILE` before submitting. Preparation is local and does not allocate GPUs.
- When the user explicitly asks to submit/start a training task, that authorizes `submit RUN_DIRECTORY`; do not ask again if profile, inputs, resources and parameters are already established. A request to configure or inspect does not itself start training.
- If required data, checkpoints or image information is missing, report the exact missing fields and continue any independent setup. Do not invent paths or run a dummy GPU job.
- Never retry a timed-out create automatically. Inspect `submission-attempt.json`, `submission.log`, and AIHC jobs with the exact run name first. Report the task ID, state and CFS output directory after submission.
- Run `python3 -m unittest discover -s deploy/aihc -p 'test_*.py'` after changing the submission tool. Validate training arguments with the corresponding module's `--help` in the installed environment. GPU execution requires a real allocated task and must not be claimed from CPU checks.
- Keep original `can-pnp` and `supermimic-reproduced` environment lockfiles intact. AIHC has a separate CUDA 13 environment in `deploy/aihc/environment`. When dependencies change, rebuild and verify the image before submission.
- Query image builds with `task.py image-status PACK_ID`: CLI 1.6.5 incorrectly labels API state 3 as failed. Official states are 3 building, 4 pushing, 5 complete, 30 failed. Do not restart a development machine merely because SSH pauses during an image build.
- Use the explicitly configured `job_api_pool` for job commands; the physical pool ID is used for pool/queue inspection. CLI list commands need `q\n` on stdin to terminate their interactive pager.

## Repository scope

The historical `docs/CODEX_REORG_TASK.md` is not a standing instruction to refactor the repository. Only perform that work when requested.
