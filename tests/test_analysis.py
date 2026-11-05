"""The vision model's reply: checking it, putting right what is wrong with it, and building the stored text from it."""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
import ollama

import config
from db.operations import add_failure_history, get_or_add_root, park_backlog, upsert_backlog
from pipeline.failure import RunStopped
from scan_and_index import scan_and_index
from tests.support import IsolatedHome
from text_processing.formatters import build_description, build_search_text
from vlm import analyzer
from vlm.reply_validation import check_reply

GOOD_REPLY = {
    "short_description": "A cat.", "scene": "A cat sits on a sofa.", "objects_and_layout": "A sofa.",
    "people_details": "No people are visible.", "colors_light_mood": "Warm.", "tags": ["cat", "sofa"],
}


class FakeClient:
    """Stands in for ollama.Client: returns the queued replies and remembers every request.

    A reply is the text of the model's answer, or ``(text, done_reason)`` for one that ended at the cap.
    """

    def __init__(self, *replies, prompt_tokens=1000):
        self.replies, self.requests, self.prompt_tokens = list(replies), [], prompt_tokens

    def chat(self, **request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        text, done_reason = reply if isinstance(reply, tuple) else (reply, "stop")
        return SimpleNamespace(message=SimpleNamespace(content=text), done_reason=done_reason,
                               eval_count=request["options"]["num_predict"] if done_reason == "length" else 100,
                               prompt_eval_count=self.prompt_tokens)


CUT = ('{"short_description": "A ca', "length")


class CheckReplyTests(unittest.TestCase):
    schema = config.ANALYSIS_SCHEMA

    def test_a_reply_that_fits_is_returned_as_it_is(self):
        reply, problems = check_reply(json.dumps(GOOD_REPLY), self.schema)
        self.assertEqual((reply, problems), (GOOD_REPLY, []))

    def test_what_is_wrong_is_named(self):
        cases = {
            "": "not valid JSON",
            '{"short_description": "A ca': "not valid JSON",
            "[1, 2]": "expected a JSON object",
            json.dumps({k: v for k, v in GOOD_REPLY.items() if k != "scene"}): "reply.scene: missing",
            json.dumps(dict(GOOD_REPLY, tags="cat")): "reply.tags: expected array",
            json.dumps(dict(GOOD_REPLY, tags=["cat", 3])): "reply.tags[1]: expected string",
            json.dumps({k: ("" if isinstance(v, str) else []) for k, v in GOOD_REPLY.items()}): "every field is empty",
        }
        for text, expected in cases.items():
            reply, problems = check_reply(text, self.schema)
            self.assertIsNone(reply, text)
            self.assertTrue(any(expected in problem for problem in problems), (text, problems))

    def test_long_and_extra_content_is_accepted_untouched(self):
        long_reply = dict(GOOD_REPLY, scene="x" * 5000, tags=[str(n) for n in range(40)], extra_field="kept")
        self.assertEqual(check_reply(json.dumps(long_reply), self.schema), (long_reply, []))

    def test_it_follows_whatever_schema_it_is_given(self):
        schema = {"type": "object", "properties": {"count": {"type": "integer"}, "kind": {"enum": ["a", "b"]}},
                  "required": ["count"]}
        self.assertEqual(check_reply('{"count": 2, "kind": "a"}', schema)[1], [])
        self.assertTrue(check_reply('{"count": 2.5}', schema)[1])
        self.assertTrue(check_reply('{"count": 2, "kind": "c"}', schema)[1])
        self.assertTrue(check_reply('{"count": true}', schema)[1])  # a boolean is not a number


class SchemaAndPromptTests(unittest.TestCase):
    def test_the_summary_field_is_a_required_text_field_of_the_schema(self):
        schema = config.ANALYSIS_SCHEMA
        self.assertIn(config.SUMMARY_FIELD, schema["required"])
        self.assertEqual(schema["properties"][config.SUMMARY_FIELD]["type"], "string")

    def test_the_prompt_describes_every_field_of_the_schema(self):
        for field in config.ANALYSIS_SCHEMA["properties"]:
            self.assertIn(field, config.ANALYSIS_PROMPT)


class AnalysisLadderTests(unittest.TestCase):
    def ask(self, client):
        return analyzer._request_analysis(client, "vlm", b"jpeg")

    def test_a_reply_that_works_is_one_request(self):
        client = FakeClient(json.dumps(GOOD_REPLY))
        reply, attempts = self.ask(client)
        self.assertEqual(reply, GOOD_REPLY)
        self.assertEqual(len(client.requests), 1)
        self.assertIsNone(attempts[0]["kind"])
        self.assertEqual(client.requests[0]["format"], config.ANALYSIS_SCHEMA)

    def test_a_cut_off_reply_is_first_asked_again_with_different_sampling_at_the_same_cap(self):
        client = FakeClient(CUT, json.dumps(GOOD_REPLY))
        reply, attempts = self.ask(client)
        first, second = (request["options"] for request in client.requests)
        self.assertEqual(second["num_predict"], first["num_predict"])
        self.assertEqual(first["temperature"], 0)
        self.assertEqual(second["temperature"], config.ANALYSIS_RETRY_TEMPERATURE)
        self.assertEqual(second["repeat_penalty"], config.ANALYSIS_RETRY_REPEAT_PENALTY)
        self.assertEqual(second["num_ctx"], first["num_ctx"])  # a different num_ctx would reload the model
        self.assertEqual([a["kind"] for a in attempts], ["cut_off", None])
        self.assertEqual(attempts[1]["step"], "changed sampling")
        self.assertEqual(len(client.requests[1]["messages"]), 2)  # the broken reply is not sent back

    def test_a_reply_cut_off_again_is_asked_for_with_a_higher_cap_and_the_normal_sampling(self):
        client = FakeClient(CUT, CUT, json.dumps(GOOD_REPLY))
        reply, attempts = self.ask(client)
        first, _, third = (request["options"] for request in client.requests)
        self.assertEqual(third["num_predict"], int(first["num_predict"] * config.ANALYSIS_CAP_STEP))
        self.assertEqual(third["temperature"], 0)
        self.assertNotIn("repeat_penalty", third)
        self.assertEqual(attempts[2]["step"], "raised cap")

    def test_the_cap_never_goes_beyond_what_the_context_window_has_room_for(self):
        client = FakeClient(*[CUT] * 10, prompt_tokens=config.OLLAMA_NUM_CTX - config.OLLAMA_MAX_OUTPUT_TOKENS - 100)
        with self.assertRaises(analyzer.AnalysisFailed) as raised:
            self.ask(client)
        caps = [request["options"]["num_predict"] for request in client.requests]
        self.assertEqual(max(caps), config.OLLAMA_NUM_CTX - client.prompt_tokens)
        self.assertEqual(raised.exception.kind, analyzer.CUT_OFF)

    def test_a_reply_that_stays_cut_off_fails_with_the_whole_record(self):
        client = FakeClient(*[CUT] * 10)
        with self.assertRaises(analyzer.AnalysisFailed) as raised:
            self.ask(client)
        self.assertEqual(len(client.requests), 2 + config.ANALYSIS_CAP_RAISES)  # first try, changed sampling, then the raises
        attempts = raised.exception.attempts
        self.assertEqual(len(attempts), len(client.requests))
        self.assertTrue(all(a["kind"] == "cut_off" and a["raw_reply"] for a in attempts))

    def test_a_complete_reply_that_does_not_fit_is_sent_back_with_the_problems(self):
        wrong = json.dumps({k: v for k, v in GOOD_REPLY.items() if k != "scene"})
        client = FakeClient(wrong, json.dumps(GOOD_REPLY))
        reply, attempts = self.ask(client)
        self.assertEqual(reply, GOOD_REPLY)
        feedback = client.requests[1]["messages"][-1]["content"]
        self.assertIn("reply.scene: missing", feedback)
        self.assertNotIn("short", feedback.replace("short_description", ""))  # nothing asks the model to write less
        self.assertEqual(client.requests[1]["messages"][-2], {"role": "assistant", "content": wrong})
        self.assertEqual([a["kind"] for a in attempts], ["invalid_reply", None])

    def test_it_gives_up_on_an_invalid_reply_after_the_configured_tries(self):
        client = FakeClient(*["not json"] * (config.ANALYSIS_RETRIES + 1))
        with self.assertRaises(analyzer.AnalysisFailed) as raised:
            self.ask(client)
        self.assertEqual(len(client.requests), config.ANALYSIS_RETRIES + 1)
        self.assertEqual(raised.exception.kind, analyzer.INVALID_REPLY)

    def test_trouble_of_one_kind_does_not_use_up_the_steps_of_another(self):
        wrong = json.dumps({"short_description": "x"})
        client = FakeClient(wrong, CUT, CUT, json.dumps(GOOD_REPLY))  # invalid, then cut off twice, then fine
        reply, attempts = self.ask(client)
        self.assertEqual(reply, GOOD_REPLY)
        self.assertEqual([a["step"] for a in attempts], ["first try", "problems sent back", "changed sampling", "raised cap"])

    def test_it_reports_each_retry(self):
        told = []
        analyzer._request_analysis(FakeClient(CUT, json.dumps(GOOD_REPLY)), "vlm", b"jpeg", on_retry=told.append)
        self.assertEqual(told, ["cut_off"])


REQUEST = httpx.Request("POST", "http://localhost")


class FlakyClient:
    """A client whose first requests fail with the given errors, and then answer."""

    def __init__(self, *outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def chat(self, **request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(message=SimpleNamespace(content=outcome), done_reason="stop", eval_count=10, prompt_eval_count=10)


GOOD = json.dumps(GOOD_REPLY)


class TroubleReachingTheModelTests(unittest.TestCase):
    def ask(self, client, **patches):
        defaults = {"restart_server": mock.Mock(return_value=True), "make_client": mock.Mock(), "unload_model": mock.Mock(),
                    "other_models_running": mock.Mock(return_value=[]), "sleep": mock.Mock()}
        defaults.update(patches)
        with mock.patch.object(analyzer, "restart_server", defaults["restart_server"]), \
                mock.patch.object(analyzer, "make_client", defaults["make_client"]), \
                mock.patch.object(analyzer, "unload_model", defaults["unload_model"]), \
                mock.patch.object(analyzer, "other_models_running", defaults["other_models_running"]), \
                mock.patch.object(analyzer.time, "sleep", defaults["sleep"]):
            told = []
            try:
                result = analyzer._request_analysis(client, "vlm", b"jpeg", on_retry=told.append)
            except Exception as exc:
                result = exc
            return result, told, defaults

    def test_no_reply_in_time_is_asked_again_with_a_longer_wait(self):
        longer = FlakyClient(GOOD)
        make_client = mock.Mock(return_value=longer)
        (reply, attempts), told, _ = self.ask(FlakyClient(httpx.ReadTimeout("slow", request=REQUEST)), make_client=make_client)
        self.assertEqual(reply, GOOD_REPLY)
        make_client.assert_called_once_with(timeout=config.OLLAMA_TIMEOUT * config.OLLAMA_TIMEOUT_STEP)
        self.assertEqual(told, ["timeout"])
        self.assertEqual([a["kind"] for a in attempts], ["timeout", None])
        self.assertEqual(len(longer.requests), 1)  # the same request, to the client that waits longer

    def test_the_wait_is_lengthened_a_set_number_of_times_and_then_the_file_is_set_aside(self):
        slow = httpx.ReadTimeout("slow", request=REQUEST)
        clients = [FlakyClient(slow) for _ in range(config.OLLAMA_TIMEOUT_RAISES)]
        make_client = mock.Mock(side_effect=clients)
        failure, _, _ = self.ask(FlakyClient(slow), make_client=make_client)
        self.assertIsInstance(failure, analyzer.AnalysisFailed)
        self.assertEqual((failure.kind, failure.park, failure.service), ("timeout", True, True))
        self.assertEqual(make_client.call_count, config.OLLAMA_TIMEOUT_RAISES)
        self.assertEqual(len(failure.attempts), config.OLLAMA_TIMEOUT_RAISES + 1)

    def test_a_broken_connection_is_asked_again_once_after_making_sure_ollama_runs(self):
        client = FlakyClient(httpx.RemoteProtocolError("server disconnected", request=REQUEST), GOOD)
        (reply, _), told, patched = self.ask(client)
        self.assertEqual(reply, GOOD_REPLY)
        patched["restart_server"].assert_called_once()
        self.assertEqual(told, ["server_crash"])
        self.assertEqual(client.requests[0], client.requests[1])  # the same request

    def test_a_runner_that_stopped_is_treated_the_same_way(self):
        client = FlakyClient(ollama.ResponseError("model runner has unexpectedly stopped", 500), GOOD)
        (reply, _), _, patched = self.ask(client)
        self.assertEqual(reply, GOOD_REPLY)
        patched["restart_server"].assert_called_once()

    def test_failing_again_after_the_restart_sets_the_file_aside_and_counts_as_trouble_with_the_machine(self):
        broken = httpx.RemoteProtocolError("server disconnected", request=REQUEST)
        failure, _, patched = self.ask(FlakyClient(broken, broken))
        self.assertEqual((failure.kind, failure.park, failure.service), ("server_crash", True, True))
        patched["restart_server"].assert_called_once()  # only one restart per image

    def test_ollama_that_cannot_be_started_is_left_to_the_usual_handling(self):
        failure, _, _ = self.ask(FlakyClient(ConnectionError("Failed to connect to Ollama")), restart_server=mock.Mock(return_value=False))
        self.assertIsInstance(failure, ConnectionError)

    def test_not_enough_memory_frees_the_other_models_and_asks_again(self):
        client = FlakyClient(ollama.ResponseError("model requires more system memory (4 GiB) than is available (2 GiB)", 500), GOOD)
        others = mock.Mock(return_value=["embed", "judge"])
        (reply, _), told, patched = self.ask(client, other_models_running=others)
        self.assertEqual(reply, GOOD_REPLY)
        others.assert_called_once_with(client, ["vlm"])
        self.assertEqual([call.args[1] for call in patched["unload_model"].call_args_list], ["embed", "judge"])
        patched["sleep"].assert_called_once_with(config.OLLAMA_RECOVERY_WAIT)
        patched["restart_server"].assert_not_called()
        self.assertEqual(told, ["out_of_memory"])

    def test_memory_that_stays_short_is_not_held_against_the_file(self):
        short = ollama.ResponseError("model requires more system memory", 500)
        failure, _, _ = self.ask(FlakyClient(short, short))
        self.assertEqual((failure.kind, failure.park, failure.service), ("out_of_memory", False, True))

    def test_a_missing_model_is_not_mistaken_for_a_crash(self):
        failure, _, patched = self.ask(FlakyClient(ollama.ResponseError("model 'x' not found", 404)))
        self.assertIsInstance(failure, ollama.ResponseError)
        patched["restart_server"].assert_not_called()

    def test_the_longer_wait_is_kept_for_the_steps_that_follow(self):
        longer = FlakyClient(CUT[0], GOOD)  # answers, but is cut off, and then a second request works
        longer.outcomes[0] = CUT[0]
        make_client = mock.Mock(return_value=longer)
        (reply, _), _, _ = self.ask(FlakyClient(httpx.ReadTimeout("slow", request=REQUEST)), make_client=make_client)
        self.assertEqual(make_client.call_count, 1)


class EmbeddingTests(unittest.TestCase):
    class Embedder:
        """Refuses an input of more than ``limit`` characters, as Ollama does when truncation is off."""

        def __init__(self, limit):
            self.limit, self.requests = limit, []

        def embed(self, **request):
            self.requests.append(request)
            text = request["input"][0]
            if len(text) > self.limit:
                raise ollama.ResponseError("the input length exceeds the context length", 400)
            return SimpleNamespace(embeddings=[[float(len(text)), 1.0]])

    def test_a_text_that_fits_is_embedded_in_one_request_without_truncation(self):
        client = self.Embedder(100)
        self.assertEqual(analyzer.embed_search_text(client, "e", "short text"), [10.0, 1.0])
        self.assertEqual(len(client.requests), 1)
        self.assertIs(client.requests[0]["truncate"], False)

    def test_a_text_too_long_is_split_and_nothing_is_cut_off(self):
        text = "\n".join(f"line number {n} has some words in it" for n in range(20))
        client = self.Embedder(150)
        vector = analyzer.embed_search_text(client, "e", text)
        leaves = [r["input"][0] for r in client.requests if len(r["input"][0]) <= 150]
        self.assertTrue(all(len(leaf) <= 150 for leaf in leaves))
        words_embedded = {word for leaf in leaves for word in leaf.split()}
        self.assertEqual(words_embedded, set(text.split()))  # every word was embedded
        self.assertEqual(vector[1], 1.0)
        self.assertGreater(vector[0], 0)

    def test_a_single_long_line_is_split_at_spaces(self):
        text = " ".join(["word"] * 100)
        vector = analyzer.embed_search_text(self.Embedder(120), "e", text)
        self.assertEqual(len(vector), 2)

    def test_other_errors_are_not_mistaken_for_a_text_that_is_too_long(self):
        class Broken:
            def embed(self, **request):
                raise ollama.ResponseError("model 'e' not found", 404)
        with self.assertRaises(ollama.ResponseError):
            analyzer.embed_search_text(Broken(), "e", "text")

    def test_the_average_weighs_each_part_by_its_length(self):
        client = self.Embedder(13)
        vector = analyzer.embed_search_text(client, "e", "aaaaaaaa bbbbbbbbbbbb")  # cut at the space into 8 and 13 characters
        self.assertAlmostEqual(vector[0], (8 * 8 + 13 * 13) / 21)


class SearchTextTests(unittest.TestCase):
    def test_the_description_is_the_text_fields_without_the_summary(self):
        self.assertEqual(build_description(GOOD_REPLY, "short_description"),
                         "A cat sits on a sofa.\n\nA sofa.\n\nNo people are visible.\n\nWarm.")

    def test_the_search_text_holds_the_description_summary_lists_and_text_in_the_image(self):
        text = build_search_text(GOOD_REPLY, ["EXIT", "exit", "Cafe"], "short_description")
        self.assertIn("A cat sits on a sofa. A sofa. No people are visible. Warm.", text)
        self.assertIn("A cat.", text)
        self.assertIn("Tags:\n- cat\n- sofa", text)
        self.assertIn("Visible text:\nEXIT\nCafe", text)  # "exit" is the same text again

    def test_it_works_for_any_fields_and_names_none(self):
        reply = {"headline": "A harbour at dawn.", "story": "Boats rest in calm water.", "boats": ["red boat", "Red boat"],
                 "boat_count": 7, "details": {"flag": "blue"}}
        text = build_search_text(reply, [], "headline")
        self.assertIn("Boats rest in calm water.", text)
        self.assertIn("Boats:\n- red boat", text)
        self.assertEqual(text.count("red boat") + text.count("Red boat"), 1)
        self.assertIn("Boat count: 7", text)
        self.assertIn('Details: {"flag": "blue"}', text)


class IndexImageTests(IsolatedHome):
    def test_the_reply_is_stored_exactly_as_the_model_wrote_it_and_the_retry_is_recorded(self):
        conn = self.connect()
        with conn:
            root_id = get_or_add_root(conn, str(self.work))
        image = self.make_image(self.work / "a.png")
        reply = dict(GOOD_REPLY, scene="x" * 3000, tags=["cat"] * 20, extra_field={"kept": True})
        client = FakeClient(CUT, json.dumps(reply))
        client.embed = lambda **request: SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])
        settings = {"vlm_model": "v", "embedding_model": "e", "embedding_dim": 3}
        with mock.patch.object(analyzer, "extract_text_blocks", return_value=["EXIT"]):
            result = scan_and_index.index_image(conn, client, settings, root_id, image)
        self.assertEqual(result, ("indexed", "A cat."))
        row = conn.execute("SELECT short_description, description, vlm_metadata_json FROM image_data").fetchone()
        stored = json.loads(row["vlm_metadata_json"])
        self.assertEqual(stored["vlm"], reply)  # nothing trimmed, dropped or added
        self.assertEqual(stored["ocr_text"], ["EXIT"])
        self.assertEqual(stored["features"]["orientation"], "square")
        self.assertEqual(row["short_description"], "A cat.")
        self.assertIn("x" * 3000, row["description"])
        self.assertEqual(conn.execute("SELECT length(embedding) FROM text_embeddings").fetchone()[0], 12)
        steps = conn.execute("SELECT kind, step FROM failure_history ORDER BY id").fetchall()
        self.assertEqual([tuple(r) for r in steps], [("cut_off", "first try"), ("resolved", "changed sampling")])


    def test_an_embedding_of_another_size_than_measured_at_setup_stops_the_run(self):
        conn = self.connect()
        with conn:
            root_id = get_or_add_root(conn, str(self.work))
        image = self.make_image(self.work / "a.png")
        client = FakeClient(json.dumps(GOOD_REPLY))
        client.embed = lambda **request: SimpleNamespace(embeddings=[[0.1, 0.2]])
        settings = {"vlm_model": "v", "embedding_model": "e", "embedding_dim": 3}
        with mock.patch.object(analyzer, "extract_text_blocks", return_value=[]), self.assertRaises(RunStopped):
            scan_and_index.index_image(conn, client, settings, root_id, image)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM image_data").fetchone()[0], 0)


class FailureRecordTests(IsolatedHome):
    def attempt(self, kind, step):
        return {"step": step, "kind": kind, "options": {"num_predict": 5}, "detail": "d" if kind else "",
                "raw_reply": "{" if kind else None, "done_reason": "length", "eval_count": 5, "prompt_eval_count": 9}

    def test_a_reply_that_needed_no_retry_leaves_no_history(self):
        conn = self.connect()
        with conn:
            add_failure_history(conn, "/a.png", [self.attempt(None, "first try")])
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM failure_history").fetchone()[0], 0)

    def test_a_retry_that_worked_is_recorded_as_resolved(self):
        conn = self.connect()
        with conn:
            add_failure_history(conn, "/a.png", [self.attempt("cut_off", "first try"), self.attempt(None, "changed sampling")])
        rows = conn.execute("SELECT kind, step FROM failure_history ORDER BY id").fetchall()
        self.assertEqual([tuple(r) for r in rows], [("cut_off", "first try"), ("resolved", "changed sampling")])

    def test_a_parked_file_keeps_its_reason_until_it_changes(self):
        conn = self.connect()
        with conn:
            root_id = get_or_add_root(conn, str(self.work))
            upsert_backlog(conn, "/a.png", root_id, "t1")
            park_backlog(conn, "/a.png", "cut_off", "cut off", "model_changed")
            upsert_backlog(conn, "/a.png", root_id, "t1")  # scanned again, unchanged
        row = conn.execute("SELECT status, park_kind, last_error FROM backlog").fetchone()
        self.assertEqual(tuple(row), ("parked", "cut_off", "cut off"))
        with conn:
            upsert_backlog(conn, "/a.png", root_id, "t2")  # the file changed
        row = conn.execute("SELECT status, park_kind, last_error, retry_when FROM backlog").fetchone()
        self.assertEqual(tuple(row), ("queued", None, None, None))


if __name__ == "__main__":
    unittest.main()
