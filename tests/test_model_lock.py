"""The models are locked: a changed model stops the run, setup keeps a fixed copy of each, and a removed copy is made again."""

import contextlib
import unittest
from unittest import mock

from imagent import config
from imagent.commands import setup
from imagent.db.operations import set_device_settings, set_model_settings
from imagent.vlm import client as vlm_client
from imagent.vlm.client import NotReady, check_ready, pin_model, pinned_name
from tests.support import DEVICE, SETTINGS, IsolatedHome, ReadyClient, models


class ReadinessTests(IsolatedHome):
    def ready(self, client, roles, host="http://localhost:11434", settings=SETTINGS):
        conn = self.connect()
        with conn:
            set_model_settings(conn, settings)
            set_device_settings(conn, DEVICE, "vd")
        with mock.patch.object(config, "OLLAMA_HOST", host), mock.patch.object(vlm_client, "make_client", return_value=client), \
                mock.patch.object(vlm_client, "server_version", return_value=(0, 9, 0)):
            return check_ready(conn, roles)

    def test_a_changed_embedding_model_blocks_with_the_way_out(self):
        client = ReadyClient(models(**{"v:latest": "vd", "e:latest": "NEW"}))
        with self.assertRaises(NotReady) as raised:
            self.ready(client, ("embedding",))
        message = str(raised.exception)
        self.assertIn("not the same version", message)
        self.assertIn("locked and cannot be changed", message)

    def blocked(self, role, name):
        client = ReadyClient(models(**{"v:latest": "vd", "j:latest": "jd", "e:latest": "ed", name: "NEW"}))
        with self.assertRaises(NotReady) as raised:
            self.ready(client, (role,))
        message = str(raised.exception)
        self.assertIn(f"The {role} model '{name}' is not the same version", message)
        self.assertIn("locked and cannot be changed", message)

    def test_a_changed_vision_model_blocks_too(self):
        self.blocked("vlm", "v:latest")

    def test_a_changed_text_model_blocks_too(self):
        self.blocked("judge", "j:latest")

    def test_models_that_are_exactly_the_ones_set_up_are_ready(self):
        client = ReadyClient(models(**{"v:latest": "vd", "e:latest": "ed", "j:latest": "jd"}))
        _, settings = self.ready(client, ("vlm", "embedding", "judge"))
        self.assertEqual(settings["vlm_model"], "v:latest")

    def test_a_model_that_cannot_see_images_is_refused(self):
        client = ReadyClient(models(**{"v:latest": "vd"}), capabilities=["completion"])
        with self.assertRaisesRegex(NotReady, "cannot look at images"):
            self.ready(client, ("vlm",))

    def test_an_ollama_that_cannot_say_what_a_model_can_do_is_trusted(self):
        client = ReadyClient(models(**{"v:latest": "vd"}))
        client.show = mock.Mock(side_effect=RuntimeError("too old"))
        self.ready(client, ("vlm",))  # setup already checked it; no error

    def test_an_ollama_on_another_computer_is_refused(self):
        client = ReadyClient(models(**{"v:latest": "vd"}))
        with self.assertRaisesRegex(NotReady, "on this computer"):
            self.ready(client, ("vlm",), host="http://192.168.1.20:11434")


class PinTests(unittest.TestCase):
    def test_the_pinned_name_follows_the_weights(self):
        self.assertEqual(pinned_name("bge-m3:latest", "sha256:3a1f9c2d7e11aabbcc"), "imagent-pin/bge-m3:3a1f9c2d7e11")
        self.assertEqual(pinned_name("hf.co/org/model:Q4", "abcdef0123456789"), "imagent-pin/model:abcdef012345")
        self.assertTrue(vlm_client.is_pinned("imagent-pin/x:1"))
        self.assertFalse(vlm_client.is_pinned("x:1"))

    def test_a_model_is_copied_once(self):
        client = ReadyClient(models(**{"e:latest": "sha256:abc123abc123xx"}))
        name, digest = pin_model(client, "e:latest")
        self.assertEqual((name, digest), ("imagent-pin/e:abc123abc123", "sha256:abc123abc123xx"))
        self.assertEqual(client.copied, [("e:latest", name)])
        pin_model(client, "e:latest")
        self.assertEqual(len(client.copied), 1)  # the copy exists already


class SetupPinTests(IsolatedHome):
    LISTING = {"v:latest": "vd", "e:latest": "sha256:abc123abc123xx", "j:latest": "jd"}

    def download(self, vlm="v", embedding="e", text_model="j", client=None, problems=None):
        """Run the download step with a stand-in Ollama; returns (settings, rejected, what verify_models was given, client, say)."""
        client = client or ReadyClient(models(**self.LISTING))
        captured = {}

        def verify(c, vlm_used, embedding_used, text_used):
            captured.update(vlm=vlm_used, embedding=embedding_used, text=text_used)
            if problems:
                return None, problems
            return {"vlm_model": vlm_used, "embedding_model": embedding_used, "judge_model": text_used}, {}

        @contextlib.contextmanager
        def fake_ollama(host):
            yield client
        with mock.patch.object(setup, "ollama_client", fake_ollama), mock.patch.object(setup, "_pull_models"), \
                mock.patch.object(setup, "verify_models", verify), mock.patch.object(setup, "say") as say, \
                mock.patch.object(setup, "measure_computer", return_value=(DEVICE, None)):
            settings, rejected = setup.download_and_verify("http://localhost:11434", vlm, embedding, text_model)
        return settings, rejected, captured, client, say

    def test_setup_locks_every_model_as_a_copy_of_its_own_and_records_them(self):
        settings, rejected, captured, client, _ = self.download()
        pinned = {"v:latest": "imagent-pin/v:vd", "e:latest": "imagent-pin/e:abc123abc123", "j:latest": "imagent-pin/j:jd"}
        self.assertEqual(captured, {"vlm": pinned["v:latest"], "embedding": pinned["e:latest"], "text": pinned["j:latest"]})
        self.assertEqual((settings["vlm_model"], settings["embedding_model"], settings["judge_model"]), tuple(pinned.values()))
        record = config.load_install_record()
        self.assertEqual(set(record["ollama_models"]) & set(pinned.values()), set(pinned.values()))  # the uninstaller removes them
        self.assertEqual(record["pinned_models"], {used: original for original, used in pinned.items()})

    def test_a_model_used_for_two_roles_is_copied_once(self):
        settings, _, captured, client, _ = self.download(text_model="v")
        self.assertEqual(captured["vlm"], captured["text"])
        self.assertEqual(sorted(original for original, _ in client.copied), ["e:latest", "v:latest"])
        self.assertEqual(settings["vlm_model"], settings["judge_model"])

    def test_a_model_that_cannot_be_copied_is_used_as_it_is_with_a_warning(self):
        client = ReadyClient(models(**self.LISTING))
        copy = client.copy

        def copy_unless_vision(source, destination):
            if source == "v:latest":
                raise RuntimeError("no space")
            copy(source, destination)
        client.copy = copy_unless_vision
        settings, _, captured, _, say = self.download(client=client)
        self.assertEqual(captured["vlm"], "v:latest")
        self.assertEqual(captured["embedding"], "imagent-pin/e:abc123abc123")
        self.assertTrue(any("Could not save a fixed copy of v:latest" in call.args[0] for call in say.call_args_list))

    def test_a_problem_with_a_copied_model_is_reported_under_the_name_the_user_typed(self):
        settings, rejected, _, _, _ = self.download(problems={"imagent-pin/v:vd": "is not a vision model."})
        self.assertIsNone(settings)
        self.assertEqual(rejected, {"v"})

    def test_a_pinned_copy_that_was_removed_is_made_again_from_the_same_version(self):
        config.save_install_record({"pinned_models": {"imagent-pin/e:abc123abc123": "e:latest"}})
        client = ReadyClient(models(**{"e:latest": "sha256:abc123abc123xx"}))
        with mock.patch.object(setup, "say") as say:
            setup._reinstall(client, "imagent-pin/e:abc123abc123")
        self.assertEqual(client.copied, [("e:latest", "imagent-pin/e:abc123abc123")])

    def test_a_pinned_copy_whose_original_changed_cannot_be_restored(self):
        config.save_install_record({"pinned_models": {"imagent-pin/e:abc123abc123": "e:latest"}})
        client = ReadyClient(models(**{"e:latest": "sha256:ffffffffffff"}))
        with mock.patch.object(setup, "say") as say:
            setup._reinstall(client, "imagent-pin/e:abc123abc123")
        self.assertEqual(client.copied, [])
        self.assertIn("cannot be restored", say.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
