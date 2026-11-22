"""No cloud access or GPU allocation in these tests."""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import task
from launch import training_args


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.site = task.read_json(task.HERE / "site.example.json")
        self.site.update(image="registry.example/nyx/train:v1", gpu_resource_name="example.com/gpu",
                         pool="aihc-test", queue="aihcq-test", cfs_instance="cfs-test",
                         cfs_mount_point="cfs.example", ssh_host="root@example")
        self.settings = {"library": "/mnt/cfs/usr/nyx/data/library.pt", "args": []}

    def test_missing_settings_cannot_submit(self):
        self.assertEqual(task.blockers(self.site, "supermimic", self.settings), [])
        broken = copy.deepcopy(self.site)
        broken.update(image="", gpu_resource_name="NVIDIA RPBZZZ6", gpu_count=24)
        self.assertGreaterEqual(len(task.blockers(broken, "supermimic", self.settings)), 3)

    def test_official_morphmimic_uses_single_gpu_and_own_entrypoint(self):
        settings = task.read_json(task.HERE / "profiles.example.json")["morphmimic-five"]
        self.assertEqual(task.blockers(self.site, "morphmimic-five", settings), [])
        site = task.effective_site(self.site, settings)
        manifest = {"name": "nyx-test", "remote_run": "/mnt/cfs/test", "source_sha256": "abc",
                    "profile": "morphmimic-five", "settings": settings}
        job = task.job_spec(site, manifest)["jobSpec"]
        self.assertEqual(job["resources"][0]["quantity"], 1)
        self.assertIn("train_morphmimic_five.sh", job["command"])
        self.assertIn(settings["human_root"], job["command"])

    def test_output_override_and_local_data_rejected(self):
        bad = {"library": "/tmp/private.pt", "args": ["--log-root=/tmp/logs"]}
        self.assertEqual(len(task.blockers(self.site, "supermimic", bad)), 2)

    def test_8_gpus_means_one_worker(self):
        manifest = {"name": "nyx-test", "remote_run": "/mnt/cfs/usr/nyx/work/test",
                    "source_sha256": "abc"}
        spec = task.job_spec(self.site, manifest)
        self.assertEqual(spec["jobSpec"]["replicas"], 1)
        self.assertEqual(spec["jobSpec"]["resources"][0]["quantity"], 8)
        self.assertFalse(spec["jobSpec"]["privileged"])
        self.assertEqual(spec["visibleScope"], 0)

    def test_incomplete_run_never_contacts_cloud(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            (dest / "source.tar.gz").write_bytes(b"fixture")
            site = dict(self.site, image="")
            manifest = {"name": "nyx-test", "remote_run": "/mnt/cfs/test", "profile": "supermimic",
                        "settings": self.settings, "site": site, "environment_sha256": "abc",
                        "source_sha256": task.digest(dest / "source.tar.gz")}
            task.write_json(dest / "manifest.json", manifest)
            task.write_json(dest / "job.json", task.job_spec(site, manifest))
            with patch.object(task, "ssh") as remote, patch.object(task.subprocess, "run") as create:
                with self.assertRaisesRegex(ValueError, "missing image"):
                    task.submit(dest)
                remote.assert_not_called()
                create.assert_not_called()

    def test_existing_attempt_blocks_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            (dest / "source.tar.gz").write_bytes(b"fixture")
            manifest = {"name": "nyx-test", "remote_run": "/mnt/cfs/test", "profile": "supermimic",
                        "settings": self.settings, "site": self.site, "environment_sha256": "abc",
                        "source_sha256": task.digest(dest / "source.tar.gz")}
            task.write_json(dest / "manifest.json", manifest)
            task.write_json(dest / "job.json", task.job_spec(self.site, manifest))
            (dest / "submission-attempt.json").write_text('{}')
            with patch.object(task, "ssh") as remote:
                with self.assertRaisesRegex(ValueError, "already had a submission"):
                    task.submit(dest)
                remote.assert_not_called()

    def test_profile_arguments_remain_separate(self):
        s = {"tracker": "/mnt/cfs/a b.pt", "diffusion": "/mnt/cfs/d.pt",
             "diffusion_repo": "/mnt/cfs/Humanoid-Diffusion", "args": ["--iterations", "2"]}
        args = training_args("can-pnp-dsrl", s, Path("/mnt/cfs/run/outputs"))
        self.assertEqual(args[1], "/mnt/cfs/a b.pt")
        self.assertEqual(args[-2:], ["--iterations", "2"])

    def test_fivepoint_flashsac_profile(self):
        s = {"fm": "/mnt/cfs/ckpt/5points_FM/fivepoints_fm.pt", "tracker_release": "/mnt/cfs/ckpt/morphmimic-v0.1",
             "args": ["--steps", "10"]}
        self.assertEqual(task.blockers(self.site, "fivepoint-flashsac", s), [])
        args = training_args("fivepoint-flashsac", s, Path("/mnt/cfs/run/outputs"))
        self.assertEqual(args[:6], ["--fm", s["fm"], "--tracker-release", s["tracker_release"],
                                    "--out-dir", "/mnt/cfs/run/outputs"])
        self.assertEqual(args[-2:], ["--steps", "10"])
        local = dict(s, tracker_release="downloads/morphmimic-v0.1", args=["--fm=/tmp/x.pt"])
        self.assertEqual(len(task.blockers(self.site, "fivepoint-flashsac", local)), 2)

    def test_profile_resource_and_priority_overrides(self):
        settings = dict(self.settings, resources={"gpu_count": 4, "cpu": 24, "memory_gib": 250,
                                                  "shared_memory_gib": 125}, priority="high")
        self.assertEqual(task.blockers(self.site, "supermimic", settings), [])
        site = task.effective_site(self.site, settings)
        spec = task.job_spec(site, {"name": "nyx-test", "remote_run": "/mnt/cfs/x", "source_sha256": "abc"})
        self.assertEqual([r["quantity"] for r in spec["jobSpec"]["resources"]], [4, 24, 250, 125])
        self.assertEqual((spec["priority"], spec["jobSpec"]["replicas"]), ("high", 1))
        self.assertEqual(task.effective_site(self.site, self.settings)["gpu_count"], self.site["gpu_count"])
        bad = dict(self.settings, resources={"gpu_count": 16, "image": "x"}, priority="urgent")
        self.assertEqual(len(task.blockers(self.site, "supermimic", bad)), 3)

    def test_name_filter_checks_later_pages(self):
        first = {"result": {"total": 101, "jobs": [{"name": "someone-else", "jobId": "job-a"}]}}
        second = {"result": {"total": 101, "jobs": [{"name": "nyx-later", "jobId": "job-b"}]}}
        responses = [subprocess.CompletedProcess([], 0, 'HTTP Status: 200\n' + json.dumps(x))
                     for x in (first, second)]
        with patch.object(task, "run", side_effect=responses) as runner, patch("builtins.print") as output:
            task.list_jobs(self.site)
            self.assertEqual(runner.call_count, 2)
            results = json.loads(output.call_args.args[0])
            self.assertEqual([j["jobId"] for j in results], ["job-b"])

    def test_building_image_cannot_be_activated(self):
        detail = {"devInstanceImagePackJob": {"status": 3, "registry": "registry.example",
                  "namespace": "nyx", "imageName": "train", "imageTag": "v1"}}
        with patch.object(task, "api_get", return_value=detail), patch.object(task, "write_json") as writer:
            with patch("builtins.print"), self.assertRaisesRegex(ValueError, "not complete"):
                task.image_status(self.site, "tk-test", Path("site.json"), activate=True)
            writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
