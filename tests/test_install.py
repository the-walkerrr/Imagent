"""Reading one pasted model set from the AI's answer: tolerant of how it is written, strict about what it holds."""

import unittest
from unittest import mock

import questionary

from install import install

EXPECTED = ("qwen2.5vl:7b", "bge-m3:567m", "qwen3:8b")  # (vlm, embedding, text), the order ask_models uses
PASTED = "qwen2.5vl:7b,qwen3:8b,bge-m3:567m"  # the order the AI is asked for: vision, text, embedding


class ParseModelSetTests(unittest.TestCase):
    def test_one_line_comma_separated(self):
        self.assertEqual(install.parse_model_set(PASTED), EXPECTED)

    def test_decoration_spaces_and_other_separators_are_ignored(self):
        for raw in (
            f"`{PASTED}`",
            f"```{PASTED}```",
            "  qwen2.5vl:7b , qwen3:8b , bge-m3:567m  ",
            '"qwen2.5vl:7b","qwen3:8b","bge-m3:567m"',
            "qwen2.5vl:7b;qwen3:8b;bge-m3:567m",
            "qwen2.5vl:7b\nqwen3:8b\nbge-m3:567m",
            "[qwen2.5vl:7b, qwen3:8b, bge-m3:567m]",
        ):
            self.assertEqual(install.parse_model_set(raw), EXPECTED, repr(raw))

    def test_a_model_from_another_registry_or_a_user(self):
        raw = "hf.co/org/model-GGUF:Q4_K_M,user/chat:7b,nomic-embed-text"
        self.assertEqual(install.parse_model_set(raw), ("hf.co/org/model-GGUF:Q4_K_M", "nomic-embed-text", "user/chat:7b"))

    def test_the_wrong_number_of_names_is_not_readable(self):
        self.assertIsNone(install.parse_model_set("qwen2.5vl:7b,qwen3:8b"))
        self.assertIsNone(install.parse_model_set(PASTED + ",extra:1b"))
        self.assertIsNone(install.parse_model_set("qwen2.5vl:7b"))
        self.assertIsNone(install.parse_model_set(""))

    def test_none_or_a_bad_name_is_not_readable(self):
        self.assertIsNone(install.parse_model_set("NONE,qwen3:8b,bge-m3:567m"))
        self.assertIsNone(install.parse_model_set("<vision model name>:<tag>,qwen3:8b,bge-m3:567m"))
        self.assertIsNone(install.parse_model_set("qwen2.5vl:7b,,bge-m3:567m"))
        self.assertIsNone(install.parse_model_set("Set 1: qwen2.5vl:7b,qwen3:8b,bge-m3:567m"))  # text around it


class ModelMenuTests(unittest.TestCase):
    """The paste row appears on top only after the prompt was copied, and goes away once a set was read."""

    def run_menu(self, answers, paste_results=()):
        """Run ask_models with the menu answers given; return (result, [(first row, default, all rows) of every menu shown])."""
        shown, scripted, pastes = [], iter(answers), iter(paste_results)

        def fake_select(message, choices, default=None, **kwargs):
            shown.append((choices[0].value, default, [c.value for c in choices if hasattr(c, "value")]))
            return mock.Mock(ask=lambda: next(scripted))

        with mock.patch.object(install, "questionary", questionary), \
                mock.patch.object(questionary, "select", side_effect=fake_select), \
                mock.patch.object(install, "say"), mock.patch.object(install, "_erase_last_line"), \
                mock.patch.object(install, "offer_prompt_copy", return_value=True), \
                mock.patch.object(install, "_paste_set", side_effect=lambda: next(pastes)):
            return install.ask_models(), shown

    def test_no_paste_row_before_the_prompt_is_copied(self):
        _, shown = self.run_menu(["continue"])
        self.assertNotIn("paste", shown[0][2])

    def test_paste_row_is_first_after_the_copy_and_gone_after_a_good_paste(self):
        names = ("a:1", "b:2", "c:3")
        result, shown = self.run_menu(["ask", "paste", "continue"], paste_results=[names])
        self.assertEqual(result, names)
        self.assertEqual(shown[1][0:2], ("paste", "paste"))  # first row, and where the pointer starts
        self.assertNotIn("paste", shown[2][2])
        self.assertEqual(shown[2][1], "continue")  # everything is set

    def test_paste_row_stays_after_an_unreadable_paste(self):
        names = ("a:1", "b:2", "c:3")
        _, shown = self.run_menu(["ask", "paste", "paste", "continue"], paste_results=[None, names])
        self.assertEqual(shown[2][0], "paste")  # still there after the failed attempt
        self.assertNotIn("paste", shown[3][2])


if __name__ == "__main__":
    unittest.main()
