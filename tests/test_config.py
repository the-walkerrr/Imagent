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
