"""Container entrypoint. Run with the image's AIHC venv Python."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

MODULES = {
    "dynamic-fm": ("Humanoid-Diffusion", "train"),
    "supermimic": ("supermimic-reproduced", "supermimic.scripts.train"),
    "can-pnp-dsrl": (".", "experiments.train_dsrl"),
    "can-pnp-v3": (".", "experiments.train_v3"),
    "fivepoint-flashsac": (".", "experiments.train_fivepoint"),
}


def training_args(profile, settings, output):
    if profile == "dynamic-fm":
        comparison = (["--compare-pretraining", settings["comparison_checkpoint"]]
                      if settings.get("comparison_checkpoint") else [])
        return ["--config", "../Dynamic-FM/can_pnp.yaml", "--output", str(output),
                "--set", "dataset.path=" + settings["dataset"],
                "--set", "model.vision_checkpoint=" + settings["vision_checkpoint"],
                "--set", "dataset.cache_dir=" + str(output.parent / "image_cache"),
                *comparison, *settings.get("args", [])]
    if profile == "supermimic":
        data = (["--data.library", settings["library"]] if settings.get("library")
                else ["--data.npz-dir", settings["npz_dir"]])
        return data + ["--log-root", str(output), "--run-name", "train"] + settings.get("args", [])
    if profile == "fivepoint-flashsac":
        # The script adds FM_DIR/inference_source and TRACKER/source to sys.path itself.
        return ["--fm", settings["fm"], "--tracker-release", settings["tracker_release"],
                "--out-dir", str(output), "--auto-resume"] + settings.get("args", [])
    return ["--tracker-checkpoint", settings["tracker"],
            "--diffusion-checkpoint", settings["diffusion"],
            "--diffusion-repo", settings["diffusion_repo"],
            "--out-dir", str(output)] + settings.get("args", [])


def main():
    run = Path(sys.argv[1])
    manifest = json.loads((run / "manifest.json").read_text())
    root = Path(__file__).resolve().parents[2]
    image_lock = Path("/root/workspace/HumanoidSkill/deploy/aihc/environment/uv.lock")
    ready = image_lock.with_name(".ready")
    if (not image_lock.exists() or not ready.exists()
            or ready.read_text().strip() != manifest["environment_sha256"]
            or hashlib.sha256(image_lock.read_bytes()).hexdigest() != manifest["environment_sha256"]):
        raise SystemExit("Image environment differs from this run; rebuild the training image.")
    os.environ["PYTHONPATH"] = os.pathsep.join([
        str(root), str(root / "hskill/src"), str(root / "tasks/can_pnp/src"), str(root / "supermimic-reproduced/src"),
        os.environ.get("PYTHONPATH", "")])
    os.environ.setdefault("MUJOCO_GL", "disable")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ["PYTHONUNBUFFERED"] = "1"
    # Platform WORLD_SIZE may count worker containers; torchrun owns these values.
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK"):
        os.environ.pop(key, None)
    import torch
    count = torch.cuda.device_count()
    expected = manifest["gpu_count"]
    if count != expected:
        raise SystemExit(f"Expected {expected} allocated GPUs, found {count}; refusing to train.")
    # Preserve any scheduler mask. SuperMimic otherwise scans host /proc GPU entries.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(count)))
    profile = manifest["profile"]
    if profile == "dynamic-fm":
        os.environ["PYTHONPATH"] = str(root / "Dynamic-FM") + os.pathsep + os.environ["PYTHONPATH"]
    directory, module = MODULES[profile]
    output = run / "outputs"
    output.mkdir(exist_ok=True)
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--node-rank=0",
               f"--nproc-per-node={expected}", "--master-addr=127.0.0.1", "--master-port=29531",
               "-m", module, *training_args(profile, manifest["settings"], output)]
    print(json.dumps({"command": command, "gpus": count, "torch": torch.__version__}, ensure_ascii=False), flush=True)
    # Keep both platform stdout and a persistent CFS copy, preserving the exit code.
    with (run / "train.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=root / directory, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b""):
            log.write(line)
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
        sys.exit(process.wait())


if __name__ == "__main__":
    main()
