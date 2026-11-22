import json
import os
import unittest
from unittest import mock

from imagent import config
from imagent.pipeline.locking import AlreadyRunning, is_running, single_instance
from tests.support import IsolatedHome


class ConfigFileTests(IsolatedHome):
    def test_saved_settings_read_back(self):
        config.save_config({"ollama_host": "http://localhost:11434"})
        self.assertEqual(config.load_config(), {"ollama_host": "http://localhost:11434"})
        self.assertEqual(list(self.home.glob("*.tmp")), [])

    def test_a_crash_while_saving_leaves_the_old_file_intact(self):
        config.save_config({"ollama_host": "old"})
        with mock.patch("imagent.config.os.replace", side_effect=OSError("power cut")):
            with self.assertRaises(OSError):
                config.save_config({"ollama_host": "new"})
        self.assertEqual(json.loads(config.CONFIG_PATH.read_text()), {"ollama_host": "old"})

    def test_a_damaged_or_missing_file_reads_as_empty(self):
        self.assertEqual(config.load_config(), {})
        self.home.mkdir(parents=True)
        config.CONFIG_PATH.write_text('{"ollama_host": "http://loc')  # cut off
        self.assertEqual(config.load_config(), {})
        config.CONFIG_PATH.write_text("[1, 2]")
        self.assertEqual(config.load_config(), {})


class TemporaryDeviceSettingsTests(IsolatedHome):
    def test_the_settings_change_for_the_block_and_the_options_sent_to_ollama_with_them(self):
        with config.temporary_device_settings(num_ctx=4096, use_mmap=True, image_long_edge=512, timeout_seconds=900):
            self.assertEqual((config.OLLAMA_NUM_CTX, config.OLLAMA_USE_MMAP, config.IMAGE_MAX_LONG_EDGE, config.OLLAMA_TIMEOUT),
                             (4096, True, 512, 900))
            self.assertEqual((config.LLM_OPTIONS["num_ctx"], config.LLM_OPTIONS["use_mmap"], config.EMBED_OPTIONS["use_mmap"]),
                             (4096, True, True))

    def test_only_what_is_named_changes(self):
        before = config.device_settings()
        with config.temporary_device_settings(image_long_edge=512):
            self.assertEqual(config.device_settings(), {**before, "image_long_edge": 512})

    def test_the_earlier_settings_come_back_even_when_the_block_fails(self):
        before, options = config.device_settings(), dict(config.LLM_OPTIONS)
        with self.assertRaises(RuntimeError), config.temporary_device_settings(num_ctx=2048, use_mmap=not before["use_mmap"]):
            raise RuntimeError("boom")
        self.assertEqual(config.device_settings(), before)
        self.assertEqual(config.LLM_OPTIONS, options)


class LockTests(IsolatedHome):
    def test_only_one_holder_at_a_time(self):
        with single_instance("indexing", "Indexing"):
            with self.assertRaisesRegex(AlreadyRunning, "Indexing is already running"):
                with single_instance("indexing", "Indexing"):
                    self.fail("the second holder must not get in")

    def test_the_lock_is_free_again_afterwards_and_other_names_are_independent(self):
        with single_instance("indexing", "Indexing"):
            with single_instance("other", "Something else"):
                pass
        with single_instance("indexing", "Indexing"):
            pass

    def test_a_crash_inside_the_block_still_releases_it(self):
        with self.assertRaises(RuntimeError):
            with single_instance("indexing", "Indexing"):
                raise RuntimeError("boom")
        with single_instance("indexing", "Indexing"):
            pass

    def test_a_running_job_can_be_told_from_a_free_lock_without_taking_it(self):
        self.assertFalse(is_running("indexing"))
        with single_instance("indexing", "Indexing"):
            self.assertTrue(is_running("indexing"))
        self.assertFalse(is_running("indexing"))
        with single_instance("indexing", "Indexing"):  # the check left it free
            pass


if __name__ == "__main__":
    unittest.main()


class OllamaProgramTests(IsolatedHome):
    def install_program(self, *parts):
        program = config.OLLAMA_DIR.joinpath("program", *parts)
        program.parent.mkdir(parents=True, exist_ok=True)
        program.write_bytes(b"x")
        return str(program)

    MANIFEST = "\n".join(f"{i:x}" * 64 + f"  ./{name}" for i, name in enumerate((
        "install.sh", "Ollama.dmg", "ollama-darwin.tgz", "Ollama-darwin.zip", "ollama-linux-amd64.tar.zst", "ollama-linux-amd64-rocm.tar.zst",
        "ollama-linux-amd64-mlx.tar.zst", "ollama-linux-arm64.tar.zst", "ollama-linux-arm64-jetpack6.tar.zst", "ollama-windows-amd64.zip",
        "ollama-windows-amd64-rocm.zip", "ollama-windows-arm64.zip")))

    def downloads(self, system, cpu, extras=()):
        with mock.patch.object(config.platform, "system", return_value=system), mock.patch.object(config.platform, "machine", return_value=cpu):
            return config.ollama_downloads(self.MANIFEST, extras)

    def test_each_os_and_cpu_gets_its_own_build_and_unknown_ones_get_none(self):
        for system, cpu, file in (("Darwin", "arm64", "ollama-darwin.tgz"), ("Darwin", "x86_64", "ollama-darwin.tgz"),
                                  ("Linux", "x86_64", "ollama-linux-amd64.tar.zst"), ("Linux", "aarch64", "ollama-linux-arm64.tar.zst"),
                                  ("Windows", "AMD64", "ollama-windows-amd64.zip"), ("Windows", "ARM64", "ollama-windows-arm64.zip")):
            [(url, name, sha)] = self.downloads(system, cpu)
            self.assertEqual(name, file)
            self.assertEqual(url, f"https://github.com/ollama/ollama/releases/download/v{config.OLLAMA_VERSION}/{file}")
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
        for system, cpu in (("Linux", "riscv64"), ("FreeBSD", "x86_64")):
            self.assertEqual(self.downloads(system, cpu), [])
            with mock.patch.object(config.platform, "system", return_value=system), mock.patch.object(config.platform, "machine", return_value=cpu):
                self.assertIsNone(config.ollama_build_prefix())

    def test_the_sums_come_from_the_release_list_and_not_from_the_code(self):
        self.assertEqual(config.ollama_checksums(self.MANIFEST)["ollama-darwin.tgz"], "2" * 64)
        self.assertEqual(config.ollama_checksums("short  ./a\n" + "A" * 64 + " *b.zip\nnot a line"), {"b.zip": "a" * 64})

    def test_an_extra_file_follows_the_build_and_one_the_release_lacks_is_left_out(self):
        names = lambda found: [name for _, name, _ in found]
        self.assertEqual(names(self.downloads("Linux", "x86_64", ("rocm",))), ["ollama-linux-amd64.tar.zst", "ollama-linux-amd64-rocm.tar.zst"])
        self.assertEqual(names(self.downloads("Windows", "x86_64", ("rocm",))), ["ollama-windows-amd64.zip", "ollama-windows-amd64-rocm.zip"])
        self.assertEqual(names(self.downloads("Linux", "arm64", ("jetpack6",))), ["ollama-linux-arm64.tar.zst", "ollama-linux-arm64-jetpack6.tar.zst"])
        self.assertEqual(names(self.downloads("Linux", "arm64", ("jetpack5",))), ["ollama-linux-arm64.tar.zst"])  # not in this release
        self.assertEqual(names(self.downloads("Linux", "x86_64", ("mlx",))), ["ollama-linux-amd64.tar.zst", "ollama-linux-amd64-mlx.tar.zst"])

    def test_a_release_without_the_build_gives_nothing_even_with_extras(self):
        manifest = "\n".join(line for line in self.MANIFEST.splitlines() if not line.endswith("ollama-linux-amd64.tar.zst"))
        with mock.patch.object(config.platform, "system", return_value="Linux"), mock.patch.object(config.platform, "machine", return_value="x86_64"):
            self.assertEqual(config.ollama_downloads(manifest, ("rocm",)), [])

    def test_the_program_is_found_where_each_os_unpacks_it(self):
        self.assertIsNone(config.private_ollama_binary())
        with mock.patch.object(config.platform, "system", return_value="Linux"):
            linux = self.install_program("bin", "ollama")
            self.assertEqual(config.private_ollama_binary(), linux)
        with mock.patch.object(config.platform, "system", return_value="Windows"):
            self.assertIsNone(config.private_ollama_binary())  # no ollama.exe
            windows = self.install_program("ollama.exe")
            self.assertEqual(config.private_ollama_binary(), windows)

    def test_imagents_own_copy_wins_over_the_one_on_path(self):
        with mock.patch.object(config.shutil, "which", return_value="/usr/bin/ollama"):
            self.assertEqual(config.ollama_binary(), "/usr/bin/ollama")
            own = self.install_program("ollama") if os.name != "nt" else self.install_program("ollama.exe")
            self.assertEqual(config.ollama_binary(), own)

    def test_the_server_gets_the_models_folder_only_when_the_program_is_imagents_own(self):
        self.assertIsNone(config.ollama_environment(None))
        self.assertEqual(config.ollama_environment("localhost")["OLLAMA_HOST"], "localhost")
        self.assertNotIn("OLLAMA_MODELS", config.ollama_environment("localhost"))
        self.install_program("ollama" if os.name != "nt" else "ollama.exe")
        env = config.ollama_environment("localhost")
        self.assertEqual(env["OLLAMA_MODELS"], str(config.OLLAMA_DIR / "models"))
        self.assertEqual(env["OLLAMA_HOST"], "localhost")
        self.assertNotIn("OLLAMA_HOST", config.ollama_environment(None))  # the address is only set when there is one to set

    def test_imagents_own_server_keeps_every_file_it_writes_inside_its_folder(self):
        self.install_program("ollama" if os.name != "nt" else "ollama.exe")
        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": "/somewhere/else"}):
            env = config.ollama_environment(None)
        home = str(config.OLLAMA_DIR / "home")
        self.assertEqual((env["HOME"], env["USERPROFILE"]), (home, home))  # its key pair and cache go under ~/.ollama
        self.assertEqual(env["OLLAMA_MODELS"], str(config.OLLAMA_DIR / "models"))  # not the user's own setting
        with mock.patch.object(config, "private_ollama_binary", return_value=None):
            self.assertEqual(config.ollama_environment("localhost")["HOME"], os.environ.get("HOME"))  # a system Ollama keeps the user's own home

    def test_tools_are_listed_from_the_file_each_one_carries(self):
        self.assertEqual(config.installed_tools(), [])
        (self.tools / "stray").mkdir(parents=True)
        (self.tools / "damaged").mkdir()
        (self.tools / "damaged" / config.TOOL_INFO_FILE).write_text("{")
        (self.tools / "ollama").mkdir()
        config.save_tool_info(self.tools / "ollama", "Ollama", "Runs models.", "1.2.3")
        self.assertEqual(config.installed_tools(), [{"name": "Ollama", "description": "Runs models.", "version": "1.2.3", "path": self.tools / "ollama"}])

    def test_the_tools_folder_sits_beside_the_data_folder_not_inside_it(self):
        self.assertEqual(config.TOOLS_DIR.parent, config.IMAGENT_HOME.parent)
        self.assertNotEqual(config.TOOLS_DIR, config.IMAGENT_HOME)

