#!/usr/bin/env python3
"""Prepare immutable training inputs locally; submit only through explicit `submit`."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import uuid
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REMOTE_ROOT = "/root/workspace/HumanoidSkill"
PROFILES = ("supermimic", "can-pnp-dsrl", "can-pnp-v3", "fivepoint-flashsac")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def ssh(site, command, **kwargs):
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-p",
                str(site["ssh_port"]), site["ssh_host"], command], **kwargs)


def aihc(site, args):
    binary = os.environ.get("AIHC_BIN") or shutil.which("aihc") or str(Path.home() / ".local/bin/aihc")
    return [binary, *args, "-C", str(Path(site["aihc_config"]).expanduser())]


def input_paths(profile, settings):
    if profile == "supermimic":
        return [("library", "file")] if settings.get("library") else [("npz_dir", "dir")]
    if profile == "fivepoint-flashsac":
        # The FM's frozen inference_source sits beside it; the release holds source/ and the ONNX export.
        return [("fm", "file"), ("tracker_release", "dir")]
    return [("tracker", "file"), ("diffusion", "file"), ("diffusion_repo", "dir")]


RESOURCE_KEYS = ("gpu_count", "cpu", "memory_gib", "shared_memory_gib")
PRIORITIES = ("high", "normal", "low")


def effective_site(site, settings):
    """Site settings with this profile's optional `resources` and `priority` overrides."""
    merged = dict(site)
    merged.update({k: v for k, v in settings.get("resources", {}).items() if k in RESOURCE_KEYS})
    if "priority" in settings:
        merged["priority"] = settings["priority"]
    return merged


def blockers(site, profile, settings):
    errors = []
    overrides = settings.get("resources", {})
    if not isinstance(overrides, dict) or set(overrides) - set(RESOURCE_KEYS):
        errors.append(f"resources may only override {', '.join(RESOURCE_KEYS)}")
    site = effective_site(site, settings if isinstance(overrides, dict) else {})
    if site.get("priority", "normal") not in PRIORITIES:
        errors.append(f"priority must be one of {', '.join(PRIORITIES)}")
    for key in ("image", "gpu_resource_name", "pool", "queue", "cfs_instance", "cfs_mount_point"):
        if not site.get(key) or "REPLACE" in site[key]:
            errors.append(f"site.local.json: missing {key}")
    if site.get("image") and (site["image"].endswith(":latest") or ":" not in site["image"].rsplit("/", 1)[-1]):
        errors.append("Use a versioned training image tag or digest, not latest.")
    if site.get("gpu_resource_name") and not re.fullmatch(r"[a-z0-9.-]+/[a-zA-Z0-9_.-]+", site["gpu_resource_name"]):
        errors.append("gpu_resource_name must be the API resource key, not a GPU display name.")
    for key in RESOURCE_KEYS:
        if type(site.get(key)) is not int or site[key] <= 0:
            errors.append(f"{key} must be a positive integer")
    if isinstance(site.get("gpu_count"), int) and site["gpu_count"] > 8:
        errors.append("This launcher supports one node with at most 8 GPUs.")
    if site.get("shared_memory_gib", 0) > site.get("memory_gib", 0):
        errors.append("shared_memory_gib must not exceed memory_gib")
    for key, _ in input_paths(profile, settings):
        value = settings.get(key, "")
        if not value or not value.startswith("/mnt/cfs/") or ".." in Path(value).parts:
            errors.append(f"{profile}: {key} must be an existing absolute CFS path")
    args = settings.get("args", [])
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        errors.append("args must be a JSON array of strings")
    else:
        reserved = ("--log-root", "--run-name", "--out-dir", "--tracker-checkpoint",
                    "--diffusion-checkpoint", "--diffusion-repo", "--data.library", "--data.npz-dir",
                    "--fm", "--tracker-release")
        if any(x.split("=", 1)[0] in reserved for x in args):
            errors.append("Set data/checkpoint/output options through profile fields, not args.")
    return errors


def job_spec(site, manifest):
    remote = manifest["remote_run"]
    code = f"/tmp/{manifest['name']}/code"
    python = REMOTE_ROOT + "/deploy/aihc/environment/.venv/bin/python"
    q = shlex.quote
    command = "\n".join([
        "set -euo pipefail", "mountpoint -q /mnt/cfs",
        f"cd {q(remote)}", f"echo {q(manifest['source_sha256'] + '  source.tar.gz')} | sha256sum -c -",
        f"mkdir -p {q(code)}", f"tar -xzf source.tar.gz -C {q(code)}",
        f"export NYX_CFS_ROOT={q(site['cfs_root'])}",
        f"source {q(code + '/deploy/aihc/cache.sh')}",
        f"exec {q(python)} {q(code + '/deploy/aihc/launch.py')} {q(remote)}",
    ])
    resources = [(site.get("gpu_resource_name") or "UNSET_GPU_RESOURCE", site["gpu_count"]),
                 ("cpu", site["cpu"]), ("memory", site["memory_gib"]),
                 ("sharedMemory", site["shared_memory_gib"])]
    return {"name": manifest["name"], "queue": site["queue"], "jobFramework": "pytorch",
            "jobSpec": {"command": "bash -lc " + q(command), "image": site.get("image", ""),
                        "replicas": 1, "resources": [{"name": k, "quantity": v} for k, v in resources],
                        "envs": [{"name": "AIHC_USE_EHCA", "value": "true"}],
                        "enableRDMA": False, "hostNetwork": False, "privileged": False},
            "priority": site.get("priority", "normal"), "faultTolerance": False, "visibleScope": 0,
            "datasources": [{"type": "cfs", "sourcePath": "/", "mountPath": "/mnt/cfs", "name": "",
                             "options": {"readOnly": False, "cfsInstanceId": site["cfs_instance"],
                                         "cfsMountPoint": site["cfs_mount_point"]}}]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def api_get(site, url, version=None):
    args = ["curl", url, "-X", "GET"]
    if version:
        args += ["-H", "version: " + version]
    response = run(aihc(site, args), capture_output=True, timeout=45).stdout
    start = response.find("{")
    if start < 0:
        raise ValueError("AIHC returned no JSON response")
    result = json.loads(response[start:])
    if result.get("code"):
        raise ValueError(f"AIHC request failed: {result['code']}")
    return result


def image_status(site, pack_id, site_path, activate=False):
    query = urlencode({"action": "DescribeDevInstanceImagePackJob",
                       "devInstanceId": site["dev_id"], "imagePackJobId": pack_id})
    detail = api_get(site, "/?" + query, "v2")["devInstanceImagePackJob"]
    # Official API enum. CLI 1.6.5 incorrectly labels state 3 as failed.
    states = {1: "initializing", 2: "running", 3: "building", 4: "pushing", 5: "complete", 30: "failed"}
    state = detail["status"]
    address = f"{detail['registry']}/{detail['namespace']}/{detail['imageName']}:{detail['imageTag']}"
    print(json.dumps({"pack_id": pack_id, "state": states.get(state, "unknown"),
                      "raw_state": state, "image": address}, indent=2))
    if activate:
        if state != 5:
            raise ValueError("Image is not complete; training image setting has not been changed")
        record = read_json(HERE / "image.local.json")
        if (record.get("image") != address or record.get("image_pack_job_id") != pack_id
                or record.get("environment_sha256") != digest(HERE / "environment/uv.lock")):
            raise ValueError("Image build record does not match this environment; verify before activation")
        site["image"] = address
        write_json(site_path, site)
        record["state"] = "complete"
        write_json(HERE / "image.local.json", record)
        print("Training image configured.")


def list_jobs(site, name="nyx-"):
    # CLI 1.6.5 can ignore --name on serverless queues; filter API results locally.
    page = 1
    matches = {}
    while True:
        query = urlencode({"resourcePoolId": site.get("job_api_pool", site["pool"]),
                           "queueID": site["queue"], "queue": site["queue"],
                           "pageNo": page, "pageSize": 100})
        result = api_get(site, "/api/v1/aijobs?" + query)
        if "result" not in result:
            raise ValueError(f"AIHC list failed: {result.get('code', 'unknown error')}")
        result = result["result"]
        for job in result["jobs"]:
            if name in job["name"]:
                matches[job["jobId"]] = {k: job.get(k) for k in ("name", "jobId", "status", "createdAt")}
        if page * 100 >= result["total"]:
            break
        page += 1
        if page > 100:
            raise ValueError("List exceeded 10,000 jobs; narrow the queue before reconciling")
    print(json.dumps(list(matches.values()), indent=2, ensure_ascii=False))


def prepare(site, profile, settings):
    name = "nyx-" + profile + "-" + datetime.datetime.now(datetime.timezone.utc).strftime("%m%d%H%M%S") + "-" + uuid.uuid4().hex[:6]
    destination = HERE / "runs" / name
    destination.mkdir(parents=True)
    files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).split(b"\0")
    with tarfile.open(destination / "source.tar.gz", "w:gz") as archive:
        for raw in sorted(set(files)):
            if not raw:
                continue
            relative = Path(os.fsdecode(raw))
            path = ROOT / relative
            if not path.exists():
                continue  # deleted working-tree file
            if path.is_symlink():
                raise ValueError(f"Source symlink needs review: {relative}")
            if path.is_file():
                if path.stat().st_size > 50 * 1024 * 1024:
                    raise ValueError(f"Large source file; put datasets on CFS: {relative}")
                archive.add(path, arcname=str(relative), recursive=False)
    lock = HERE / "environment/uv.lock"
    site = effective_site(site, settings)
    manifest = {"name": name, "profile": profile, "settings": settings, "gpu_count": site["gpu_count"],
                "remote_run": site["cfs_root"] + "/work/HumanoidSkill/runs/" + name,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": digest(destination / "source.tar.gz"),
                "environment_sha256": digest(lock) if lock.exists() else "",
                "site": site}
    errors = blockers(site, profile, settings)
    if not lock.exists():
        errors.append("AIHC environment/uv.lock is missing; finish setup_environment.sh first")
    manifest["blockers_at_prepare"] = errors
    write_json(destination / "manifest.json", manifest)
    write_json(destination / "job.json", job_spec(site, manifest))
    print(destination)
    print("Prepared locally; no upload or GPU allocation.")
    for error in errors:
        print("BLOCKED:", error)
    return destination


def submit(destination):
    destination = Path(destination).resolve()
    manifest = read_json(destination / "manifest.json")
    site = manifest["site"]
    errors = blockers(site, manifest["profile"], manifest["settings"])
    if not manifest["environment_sha256"]:
        errors.append("Environment lock missing; prepare again after installation")
    if digest(destination / "source.tar.gz") != manifest["source_sha256"]:
        errors.append("Source archive changed after preparation")
    if read_json(destination / "job.json") != job_spec(site, manifest):
        errors.append("job.json changed after preparation; edit site/profile and prepare again")
    if (destination / "submission-attempt.json").exists():
        errors.append("This run already had a submission attempt; reconcile job list before retrying")
    if errors:
        raise ValueError("\n".join(errors))
    checks = ["set -euo pipefail", "mountpoint -q /mnt/cfs"]
    for key, kind in input_paths(manifest["profile"], manifest["settings"]):
        checks.append(f"test -{'f' if kind == 'file' else 'd'} {shlex.quote(manifest['settings'][key])}")
    ssh(site, "bash -lc " + shlex.quote("\n".join(checks)), timeout=30)
    # mkdir without -p on the run itself prevents overwriting an existing run.
    remote = manifest["remote_run"]
    ssh(site, "mkdir -p " + shlex.quote(str(Path(remote).parent)) + " && mkdir " + shlex.quote(remote), timeout=30)
    run(["rsync", "-az", "-e", f"ssh -o BatchMode=yes -p {site['ssh_port']}",
         str(destination) + "/", site["ssh_host"] + ":" + remote + "/"], timeout=300)
    write_json(destination / "submission-attempt.json", {"name": manifest["name"], "state": "sending"})
    # Never retry create automatically: a timeout may occur after the server accepted it.
    try:
        result = subprocess.run(aihc(site, ["job", "create", "-f", str(destination / "job.json"),
                                            "-p", site.get("job_api_pool", site["pool"]), "-q", site["queue"]]),
                                input="q\n", text=True, capture_output=True, timeout=120)
        (destination / "submission.log").write_text(result.stdout + result.stderr)
        print(result.stdout, result.stderr)
        write_json(destination / "submission-attempt.json", {"name": manifest["name"], "returncode": result.returncode,
                   "state": "response_received", "job_ids": re.findall(r"\bjob-[a-zA-Z0-9]+\b", result.stdout)})
        if result.returncode:
            raise ValueError("AIHC returned an error; inspect submission.log and job list before retrying")
        if not re.search(r"\bjob-[a-zA-Z0-9]+\b", result.stdout):
            raise ValueError("No job ID in response; reconcile by run name before treating submission as successful")
    except subprocess.TimeoutExpired:
        raise ValueError("Submission outcome unknown. Query AIHC by the run name; do not resubmit.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=HERE / "site.local.json")
    sub = parser.add_subparsers(dest="command", required=True)
    for action in ("doctor", "prepare"):
        p = sub.add_parser(action)
        p.add_argument("profile", choices=PROFILES)
        p.add_argument("--profiles", type=Path, default=HERE / "profiles.local.json")
    p = sub.add_parser("submit")
    p.add_argument("run_directory", type=Path)
    p = sub.add_parser("status")
    p.add_argument("job_id")
    p = sub.add_parser("image-status")
    p.add_argument("image_pack_job_id")
    p.add_argument("--activate", action="store_true", help="Configure a completed, recorded image for training")
    p = sub.add_parser("list")
    p.add_argument("--name", default="nyx-", help="Substring matched locally across queue pages")
    args = parser.parse_args()
    if args.command == "submit":
        submit(args.run_directory)
        return
    site = read_json(args.site)
    if args.command in ("doctor", "prepare"):
        settings = read_json(args.profiles)[args.profile]
        if args.command == "prepare":
            prepare(site, args.profile, settings)
        else:
            errors = blockers(site, args.profile, settings)
            if not (HERE / "environment/uv.lock").exists():
                errors.append("Missing AIHC environment lock")
            print("\n".join(errors) if errors else "Local config complete. Submission also checks remote CFS inputs.")
            sys.exit(bool(errors))
    elif args.command == "list":
        list_jobs(site, args.name)
    elif args.command == "image-status":
        image_status(site, args.image_pack_job_id, args.site, args.activate)
    else:
        cmd = ["job", "get", args.job_id]
        run(aihc(site, cmd + ["-p", site.get("job_api_pool", site["pool"]), "-q", site["queue"]]), input="q\n", timeout=45)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
