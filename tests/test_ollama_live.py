"""The check to run before releasing with a new Ollama version: the pinned Ollama really downloads, installs, and runs Imagent's pipeline.

Every other test uses a stand-in for Ollama. These use the real one, so they are skipped unless asked for (they download gigabytes):

    IMAGENT_LIVE_OLLAMA=1 IMAGENT_LIVE_VLM=<vision model> IMAGENT_LIVE_EMBEDDING=<embedding model> IMAGENT_LIVE_TEXT=<text model> \\
        python -m unittest tests.test_ollama_live -v

Nothing of the real Imagent folder or the real Ollama is used: the pinned Ollama is installed into a temporary folder, listens on
a spare port, keeps its models there, and everything is deleted afterwards. The models are downloaded again on every run.
"""

import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from imagent import config
from imagent.commands import setup
from imagent.image_processing import sample_pictures
from imagent.text_processing.formatters import build_search_text
from imagent.vlm.analyzer import analyse_image, embed_search_text, understand_query
from imagent.vlm.client import load_model, other_models_running, server_version, unload_model
from tests.support import IsolatedHome

ENABLED = os.environ.get("IMAGENT_LIVE_OLLAMA") == "1"
MODEL_VARIABLES = ("IMAGENT_LIVE_VLM", "IMAGENT_LIVE_EMBEDDING", "IMAGENT_LIVE_TEXT")
NOT_ENABLED = "a live check: set IMAGENT_LIVE_OLLAMA=1 to run it (see the top of this file)"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@unittest.skipUnless(ENABLED, NOT_ENABLED)
class PinnedDownloadTests(unittest.TestCase):
    def test_the_release_list_is_the_pinned_one_and_every_file_chosen_from_it_matches_its_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / config.OLLAMA_MANIFEST
            url = config.ollama_release_url(config.OLLAMA_MANIFEST)
            self.assertEqual(setup._download_file(url, manifest), config.OLLAMA_MANIFEST_SHA256, f"{url} is not the file the checksum in config.py is for")
            text = manifest.read_text()
            # Every OS and CPU, each with every extra file setup can ask for.
            wanted = set()
            for system, cpu in (("Darwin", "arm64"), ("Darwin", "x86_64"), ("Linux", "x86_64"), ("Linux", "aarch64"),
                                ("Windows", "AMD64"), ("Windows", "ARM64")):
                with mock.patch.object(config.platform, "system", return_value=system), mock.patch.object(config.platform, "machine", return_value=cpu):
                    found = config.ollama_downloads(text, ("rocm", "jetpack5", "jetpack6"))
                    self.assertTrue(found, f"the release has no build for {system} on {cpu}")
                    wanted.update((name, sha) for _, name, sha in found)
            for file_name, expected in sorted(wanted):  # the macOS builds for both CPUs are one file
                with self.subTest(file=file_name):
                    target = Path(tmp) / file_name
                    self.assertEqual(setup._download_file(config.ollama_release_url(file_name), target), expected)
                    target.unlink()


@unittest.skipUnless(ENABLED, NOT_ENABLED)
class PinnedOllamaTests(IsolatedHome):
    def test_the_pinned_ollama_installs_and_runs_the_pipeline_on_a_picture(self):
        missing = [name for name in MODEL_VARIABLES if not os.environ.get(name)]
        if missing:
            self.skipTest(f"set {', '.join(missing)} to the model names to check")
        vlm, embedding, text_model = (os.environ[name] for name in MODEL_VARIABLES)
        host = f"http://127.0.0.1:{free_port()}"
        patchers = (mock.patch.object(config, "OLLAMA_HOST", host),
                    mock.patch.object(setup, "fail", side_effect=lambda message: self.fail(message)))  # a failed install ends the test, not the run
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        # What setup does when the computer has no Ollama: install the pinned build for this OS, checked against its checksum.
        setup.install_ollama()
        binary = config.private_ollama_binary()
        self.assertIsNotNone(binary, "the install left no `ollama` program")
        expected = tuple(int(part) for part in config.OLLAMA_VERSION.split("."))
        self.assertEqual(setup._ollama_command_version(binary), expected)

        # What setup does with the models: download, keep a fixed copy, check what each can do, measure the computer.
        settings, problems = setup.download_and_verify(host, vlm, embedding, text_model)
        self.assertEqual(problems, {})
        self.assertIsNotNone(settings, "setup did not accept the models")
        config.apply_device_settings(settings["device"])

        folder = self.work / "pictures"
        sample_pictures.unpack(folder)
        picture = sorted(path for path in folder.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png"))[0]

        with setup.ollama_client(host) as client:
            self.assertIsNotNone(client, "the pinned Ollama did not start")
            self.assertEqual(server_version(host), expected, "the server runs a different version than the one pinned")

            # What indexing does with a picture: the vision model's reply checked against the schema, then its embedding.
            analysis = analyse_image(client, settings["vlm_model"], str(picture))
            self.assertTrue(analysis.reply, "the vision model sent an empty reply")
            search_text = build_search_text(analysis.reply, analysis.ocr_text, config.SUMMARY_FIELD)
            self.assertTrue(search_text.strip())
            vector = embed_search_text(client, settings["embedding_model"], search_text)
            self.assertEqual(len(vector), settings["embedding_dim"])

            # What searching does with a question.
            understood = understand_query(client, settings["judge_model"], "a dog on a beach")
            self.assertTrue(understood.normalized_query.strip())

            # What Imagent does to make room: say which models are in memory and free them.
            load_model(client, settings["vlm_model"])
            self.assertIn(settings["vlm_model"], other_models_running(client, []))
            unload_model(client, settings["vlm_model"])
            for _ in range(20):  # Ollama frees the memory a moment after it is asked to
                if settings["vlm_model"] not in other_models_running(client, []):
                    break
                time.sleep(0.5)
            self.assertNotIn(settings["vlm_model"], other_models_running(client, []))


if __name__ == "__main__":
    unittest.main()
