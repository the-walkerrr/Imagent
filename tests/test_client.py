import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import config
from db.operations import set_model_settings
from tests.support import IsolatedHome
from vlm import client as vlm_client
from vlm.client import NotReady, _base_url, check_ready, load_model, other_models_running, server_version, start_server


def serve_version(version: str) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"version": version}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class AddressTests(unittest.TestCase):
    def test_addresses_match_what_the_ollama_client_uses(self):
        self.assertEqual(_base_url("localhost"), "http://localhost:11434")
        self.assertEqual(_base_url("http://localhost:11434"), "http://localhost:11434")
        self.assertEqual(_base_url("http://box"), "http://box:80")
        self.assertEqual(_base_url("https://box"), "https://box:443")
        self.assertEqual(_base_url("[::1]:9"), "http://[::1]:9")

    def test_server_version_is_read_from_the_server(self):
        for reported, expected in (("0.6.4", (0, 6, 4)), ("0.9.1-rc2", (0, 9, 1)), ("0.0.0", None)):
            server = serve_version(reported)
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            self.assertEqual(server_version(f"127.0.0.1:{server.server_port}"), expected, reported)

    def test_an_unreachable_server_has_no_version(self):
        self.assertIsNone(server_version("127.0.0.1:1"))


class LoadModelTests(unittest.TestCase):
    def test_a_chat_model_is_loaded_with_the_options_it_will_be_used_with(self):
        client = mock.Mock()
        load_model(client, "judge")
        client.generate.assert_called_once_with(model="judge", prompt="", options=config.LLM_OPTIONS)

    def test_an_embedding_model_is_loaded_by_embedding_something(self):
        client = mock.Mock()
        load_model(client, "embed", embedding=True)
        self.assertEqual(client.embed.call_args.kwargs["model"], "embed")

    def test_a_failure_is_left_for_the_real_request_to_report(self):
        load_model(mock.Mock(**{"generate.side_effect": RuntimeError("down")}), "judge")


class OtherModelsRunningTests(unittest.TestCase):
    @staticmethod
    def running(*names):
        return mock.Mock(**{"ps.return_value": mock.Mock(models=[mock.Mock(model=name) for name in names])})

    def test_models_the_search_uses_are_not_other_models(self):
        self.assertEqual(other_models_running(self.running("judge", "embed"), ["judge", "embed"]), [])

    def test_any_other_loaded_model_is_reported_by_name(self):
        self.assertEqual(other_models_running(self.running("judge", "vision", "chat"), ["judge", "embed"]), ["vision", "chat"])

    def test_nothing_loaded_means_no_other_models(self):
        self.assertEqual(other_models_running(self.running(), ["judge"]), [])

    def test_an_ollama_that_cannot_say_is_not_treated_as_busy(self):
        self.assertEqual(other_models_running(mock.Mock(**{"ps.side_effect": RuntimeError("down")}), ["judge"]), [])


class StartServerTests(unittest.TestCase):
    def test_the_server_listens_where_imagent_looks(self):
        with mock.patch.object(vlm_client.subprocess, "Popen") as popen, \
                mock.patch.object(vlm_client, "is_reachable", return_value=True), \
                mock.patch.object(vlm_client.signal, "signal") as handler:
            self.assertIsNotNone(start_server(object(), "http://localhost:5555"))
        self.assertEqual(popen.call_args.kwargs["env"]["OLLAMA_HOST"], "http://localhost:5555")
        installed = {call.args[0] for call in handler.call_args_list}
        self.assertIn(vlm_client.signal.SIGTERM, installed)  # a plain `kill` must still stop the server

    def test_a_server_that_never_answers_is_stopped_and_reported(self):
        with mock.patch.object(vlm_client.subprocess, "Popen") as popen, \
                mock.patch.object(vlm_client, "is_reachable", return_value=False), \
                mock.patch.object(vlm_client.time, "sleep"), \
                mock.patch.object(vlm_client.signal, "signal"):
            self.assertIsNone(start_server(object(), "localhost", wait_seconds=2))
        popen.return_value.terminate.assert_called_once()


class ReadinessTests(IsolatedHome):
    SETTINGS = {"vlm_model": "v:latest", "embedding_model": "e:latest", "judge_model": "j:latest",
                "vlm_digest": None, "embedding_digest": None, "judge_digest": None, "embedding_dim": 3}

    def test_not_set_up(self):
        with self.assertRaisesRegex(NotReady, "not set up"):
            check_ready(self.connect(), ("vlm",))

    def test_models_known_but_config_json_lost(self):
        conn = self.connect()
        with conn:
            set_model_settings(conn, self.SETTINGS)
        with mock.patch.object(config, "OLLAMA_HOST", ""):
            with self.assertRaisesRegex(NotReady, "config.json"):
                check_ready(conn, ("vlm",))

    def test_an_old_ollama_is_refused_with_the_fix(self):
        conn = self.connect()
        with conn:
            set_model_settings(conn, self.SETTINGS)
        listing = mock.Mock(models=[mock.Mock(model="v:latest", digest="d")])
        fake = mock.Mock(list=mock.Mock(return_value=listing))
        with mock.patch.object(config, "OLLAMA_HOST", "http://localhost:11434"), \
                mock.patch.object(vlm_client, "make_client", return_value=fake), \
                mock.patch.object(vlm_client, "server_version", return_value=(0, 3, 0)):
            with self.assertRaisesRegex(NotReady, "too old"):
                check_ready(conn, ("vlm",))


if __name__ == "__main__":
    unittest.main()
