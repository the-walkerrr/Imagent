"""Reading images: how much memory they may use, what is wrong with one that cannot be opened, and the steps of reading its text."""

import functools
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image, UnidentifiedImageError

import config
from image_processing import memory, ocr
from image_processing.features import extract_image_features
from image_processing.preprocess import preprocess_image
from image_processing.problems import BUSY, OCR_FAILED, TOO_LARGE, UNREADABLE, UNSUPPORTED, ImageProblem, explain
from pipeline.failure import IndexingFailure
from tests.support import IsolatedHome
from vlm import analyzer

MEMINFO = "MemTotal:       16000000 kB\nMemFree:          500000 kB\nMemAvailable:    8000000 kB\nBuffers: 1 kB\n"
VM_STAT = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:                               100.\n"
           "Pages active:                            9999.\nPages inactive:                           50.\n"
           "Pages speculative:                        10.\nPages purgeable:                           7.\n")


class MemoryTests(unittest.TestCase):
    def test_free_memory_is_read_from_what_each_system_says(self):
        self.assertEqual(memory._parse_meminfo(MEMINFO), 8000000 * 1024)
        self.assertEqual(memory._parse_vm_stat(VM_STAT), (100 + 50 + 10) * 16384)
        self.assertIsNone(memory._parse_meminfo("MemTotal: 1 kB\n"))
        self.assertIsNone(memory._parse_vm_stat("nonsense"))

    def test_the_route_follows_the_memory_free_and_scales_with_it(self):
        pixels = 10_000_000  # a 10 megapixel image
        with mock.patch.object(config, "IMAGE_MEMORY_SHARE", 1.0), mock.patch.object(config, "IMAGE_MAX_MEGAPIXELS", 0.0):
            self.assertEqual(memory.choose_route(5000, 2000, pixels * memory.NORMAL_BYTES_PER_PIXEL), memory.NORMAL)
            self.assertEqual(memory.choose_route(5000, 2000, pixels * memory.NORMAL_BYTES_PER_PIXEL - 1), memory.GENTLE)
            self.assertEqual(memory.choose_route(5000, 2000, pixels * memory.GENTLE_BYTES_PER_PIXEL - 1), TOO_LARGE)
            self.assertEqual(memory.choose_route(5000, 2000, pixels * 100), memory.NORMAL)  # more memory, bigger images allowed

    def test_the_share_of_free_memory_an_image_may_use_is_a_setting(self):
        with mock.patch.object(config, "IMAGE_MAX_MEGAPIXELS", 0.0):
            free = 10_000_000 * memory.NORMAL_BYTES_PER_PIXEL
            with mock.patch.object(config, "IMAGE_MEMORY_SHARE", 1.0):
                self.assertEqual(memory.choose_route(5000, 2000, free), memory.NORMAL)
            with mock.patch.object(config, "IMAGE_MEMORY_SHARE", 0.5):
                self.assertEqual(memory.choose_route(5000, 2000, free), memory.GENTLE)

    def test_when_free_memory_is_not_known_the_gentle_way_is_used(self):
        with mock.patch.object(config, "IMAGE_MAX_MEGAPIXELS", 0.0):
            self.assertEqual(memory.choose_route(100, 100, None), memory.GENTLE)

    def test_a_limit_in_megapixels_set_by_the_user_always_applies(self):
        with mock.patch.object(config, "IMAGE_MAX_MEGAPIXELS", 5.0):
            self.assertEqual(memory.choose_route(3000, 2000, 10**12), TOO_LARGE)
            self.assertEqual(memory.choose_route(2000, 2000, 10**12), memory.NORMAL)

    def test_pillows_own_pixel_limit_is_off(self):
        self.assertIsNone(Image.MAX_IMAGE_PIXELS)


class PlanImageTests(IsolatedHome):
    def test_an_image_that_fits_is_read_normally(self):
        image = self.make_image(self.work / "a.png")
        with mock.patch.object(memory, "available_memory", return_value=10**12):
            self.assertEqual(memory.plan_image(str(image)), memory.NORMAL)

    def test_an_image_too_large_is_refused_with_the_numbers(self):
        image = self.make_image(self.work / "a.png")
        with mock.patch.object(memory, "available_memory", return_value=100), self.assertRaises(ImageProblem) as raised:
            memory.plan_image(str(image))
        self.assertEqual((raised.exception.kind, raised.exception.park, raised.exception.retry_when), (TOO_LARGE, False, "on_request"))  # not parked: tried again by every run
        self.assertIn("megapixels", str(raised.exception))
        self.assertIn("Close other programs", str(raised.exception))
        self.assertIsInstance(raised.exception, IndexingFailure)

    def test_a_file_that_is_not_an_image_is_unreadable(self):
        bad = self.work / "bad.png"
        bad.write_bytes(b"this is not a picture")
        with mock.patch.object(config, "FILE_SETTLE_SECONDS", 0), self.assertRaises(ImageProblem) as raised:
            memory.plan_image(str(bad))
        self.assertEqual((raised.exception.kind, raised.exception.retry_when), (UNREADABLE, "file_changed"))

    def test_a_format_without_a_decoder_is_unsupported_not_damaged(self):
        odd = self.work / "picture.zzz"
        odd.write_bytes(b"whatever")
        with self.assertRaises(ImageProblem) as raised:
            memory.plan_image(str(odd))
        self.assertEqual((raised.exception.kind, raised.exception.retry_when), (UNSUPPORTED, "on_request"))
        self.assertIn(".zzz", str(raised.exception))


class ExplainTests(unittest.TestCase):
    def test_errors_that_are_not_about_the_image_are_left_alone(self):
        self.assertIsNone(explain(PermissionError(13, "denied"), "/a.png"))  # a real I/O error carries an error number
        self.assertIsNone(explain(ValueError("x"), "/a.png"))

    def test_a_file_still_being_written_is_not_held_against_it(self):
        with mock.patch("image_processing.problems.file_is_changing", return_value=True):
            found = explain(OSError("image file is truncated"), "/a.png")
        self.assertEqual((found.kind, found.park), (BUSY, False))

    def test_pillows_errors_are_told_from_memory_trouble(self):
        with mock.patch("image_processing.problems.file_is_changing", return_value=False):
            self.assertEqual(explain(OSError("image file is truncated"), "/a.png").kind, UNREADABLE)
            self.assertEqual(explain(SyntaxError("broken PNG file"), "/a.png").kind, UNREADABLE)
        self.assertEqual(explain(MemoryError(), "/a.png").kind, TOO_LARGE)


class GentleReadingTests(IsolatedHome):
    def make_jpeg(self, size=(3000, 2000), orientation=None):
        path = self.work / "big.jpg"
        exif = Image.Exif()
        if orientation:
            exif[0x0112] = orientation
        Image.effect_noise(size, 30).convert("RGB").save(path, quality=85, exif=exif)
        return str(path)

    def test_a_shrunk_jpeg_gives_nearly_the_same_picture_for_the_model(self):
        path = self.make_jpeg()
        normal = Image.open(io.BytesIO(preprocess_image(path)))
        gentle = Image.open(io.BytesIO(preprocess_image(path, gentle=True)))
        self.assertEqual(normal.size, gentle.size)
        self.assertLessEqual(max(gentle.size), config.IMAGE_MAX_LONG_EDGE)
        difference = sum(abs(a - b) for a, b in zip(normal.tobytes(), gentle.tobytes())) / len(normal.tobytes())
        self.assertLess(difference, 10)  # of 255 per channel, on a picture of pure noise, the hardest case

    def test_other_formats_are_not_changed_by_the_gentle_way(self):
        path = self.work / "big.png"
        Image.new("RGB", (3000, 2000), (10, 20, 30)).save(path)
        self.assertEqual(preprocess_image(str(path)), preprocess_image(str(path), gentle=True))

    def test_the_features_do_not_depend_on_the_way_of_reading(self):
        path = self.make_jpeg((3000, 2000), orientation=6)  # stored sideways: displayed as portrait
        normal, gentle = extract_image_features(path), extract_image_features(path, gentle=True)
        self.assertEqual(normal["orientation"], "portrait")
        self.assertEqual(gentle["orientation"], "portrait")


class FakeEngine:
    """Stands in for the OCR engine; ``fails(array)`` says when it raises ``error``."""

    def __init__(self, max_side_len=2000, fails=lambda pixels: False, error=RuntimeError("engine failed")):
        self.max_side_len, self.fails, self.error, self.shapes = max_side_len, fails, error, []

    def __call__(self, pixels):
        self.shapes.append(pixels.shape)
        if self.fails(pixels):
            raise self.error
        return SimpleNamespace(txts=("TEXT",), scores=(0.9,))


class OcrTests(IsolatedHome):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(config, "OCR_TILE_SHARE", 1.0)  # a tile as big as the engine reads without shrinking, to keep the numbers simple
        patcher.start()
        self.addCleanup(patcher.stop)

    def read(self, engine, size=(50, 30), tiled=False, path=None):
        path = path or self.work / "a.png"
        Image.new("RGB", size, (200, 100, 50)).save(path)
        with mock.patch.object(ocr, "_engine", functools.lru_cache(maxsize=1)(lambda: engine)):
            return ocr.extract_text_blocks(str(path), tiled=tiled)

    def test_tiles_start_where_the_overlap_says_and_the_last_ends_at_the_edge(self):
        self.assertEqual(ocr._starts(50, 20, 18), [0, 18, 30])
        self.assertEqual(ocr._starts(20, 20, 18), [0])
        self.assertEqual(ocr._starts(10, 20, 18), [0])

    def test_a_small_image_is_read_whole(self):
        engine = FakeEngine()
        self.assertEqual(self.read(engine), ["TEXT"])
        self.assertEqual(engine.shapes, [(30, 50, 3)])

    def test_an_image_larger_than_the_engine_reads_without_shrinking_is_tiled_at_once(self):
        engine = FakeEngine(max_side_len=40)
        self.assertEqual(self.read(engine, size=(50, 30)), ["TEXT"] * 2)  # 2 across, 1 down
        self.assertNotIn((30, 50, 3), engine.shapes)  # never read whole

    def test_an_image_the_engine_cannot_take_whole_is_then_read_in_tiles(self):
        engine = FakeEngine(max_side_len=60, fails=lambda pixels: max(pixels.shape[:2]) > 20)
        with mock.patch.object(config, "OCR_TILE_SHARE", 0.25):
            self.assertEqual(self.read(engine), ["TEXT"] * 12)  # tiles of 15: 4 across, 3 down
        self.assertEqual(engine.shapes[0], (30, 50, 3))  # the whole image was tried first
        self.assertTrue(all(max(shape[:2]) <= 15 for shape in engine.shapes[1:]))

    def test_a_tile_is_a_share_of_what_the_engine_reads_without_shrinking(self):
        engine = FakeEngine(max_side_len=40)
        with mock.patch.object(config, "OCR_TILE_SHARE", 0.5):
            self.read(engine, size=(50, 30), tiled=True)
        self.assertTrue(all(max(shape[:2]) <= 20 for shape in engine.shapes))

    def test_an_image_big_for_the_memory_goes_straight_to_tiles(self):
        engine = FakeEngine(max_side_len=20)
        self.read(engine, tiled=True)
        self.assertNotIn((30, 50, 3), engine.shapes)

    def test_not_enough_memory_for_the_whole_image_leads_to_tiles(self):
        engine = FakeEngine(max_side_len=60, fails=lambda pixels: max(pixels.shape[:2]) > 20, error=MemoryError())
        with mock.patch.object(config, "OCR_TILE_SHARE", 0.25):
            self.assertEqual(len(self.read(engine)), 12)

    def test_not_enough_memory_for_a_tile_means_the_image_is_too_large(self):
        engine = FakeEngine(max_side_len=20, fails=lambda pixels: True, error=MemoryError())
        with self.assertRaises(ImageProblem) as raised:
            self.read(engine)
        self.assertEqual(raised.exception.kind, TOO_LARGE)

    def test_when_every_way_fails_the_image_is_set_aside_not_stored_without_its_text(self):
        engine = FakeEngine(fails=lambda pixels: True)
        with self.assertRaises(ImageProblem) as raised:
            self.read(engine)
        failure = raised.exception
        self.assertEqual((failure.kind, failure.park, failure.service), (OCR_FAILED, True, True))
        self.assertEqual([a["step"] for a in failure.attempts], ["whole image", "tiles", "tiles, engine loaded afresh"])
        self.assertIn("engine failed", str(failure))

    def test_the_engine_is_loaded_afresh_for_the_last_step(self):
        loads = []
        engine = FakeEngine(fails=lambda pixels: True)

        def load():
            loads.append(1)
            return engine
        path = self.work / "a.png"
        Image.new("RGB", (50, 30)).save(path)
        with mock.patch.object(ocr, "_engine", functools.lru_cache(maxsize=1)(load)), self.assertRaises(ImageProblem):
            ocr.extract_text_blocks(str(path))
        self.assertEqual(len(loads), 2)  # once, then again after the cache was cleared

    def test_an_image_with_no_text_is_not_a_failure(self):
        engine = FakeEngine()
        engine.__class__ = type("NoText", (FakeEngine,), {"__call__": lambda self, pixels: SimpleNamespace(txts=None, scores=None)})
        self.assertEqual(self.read(engine), [])

    def test_a_damaged_image_is_reported_as_such_and_the_engine_is_never_asked(self):
        engine = FakeEngine()
        bad = self.work / "bad.png"
        bad.write_bytes(b"not a picture")
        with mock.patch.object(config, "FILE_SETTLE_SECONDS", 0), mock.patch.object(ocr, "_engine", functools.lru_cache(maxsize=1)(lambda: engine)):
            with self.assertRaises(ImageProblem) as raised:
                ocr.extract_text_blocks(str(bad))
        self.assertEqual(raised.exception.kind, UNREADABLE)
        self.assertEqual(engine.shapes, [])

    def test_tiles_of_a_picture_stored_sideways_are_turned_upright(self):
        path = self.work / "turned.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6
        Image.new("RGB", (40, 20)).save(path, exif=exif)
        engine = FakeEngine()
        with mock.patch.object(ocr, "_engine", functools.lru_cache(maxsize=1)(lambda: engine)):
            ocr.extract_text_blocks(str(path), tiled=True)
        self.assertEqual(engine.shapes, [(40, 20, 3)])


class AnalyseImageOrderTests(unittest.TestCase):
    def run_analysis(self, **patches):
        calls = []

        def record(name, result=None):
            def call(*args, **kwargs):
                calls.append(name)
                return result
            return call
        defaults = {
            "plan_image": record("plan", memory.NORMAL), "preprocess_image": record("preprocess", b"jpeg"),
            "extract_image_features": record("features", {"orientation": "square"}),
            "extract_text_blocks": record("ocr", ["EXIT"]), "_request_analysis": record("model", ({"a": "b"}, [])),
        }
        defaults.update(patches)
        with mock.patch.multiple(analyzer, **defaults):
            try:
                return analyzer.analyse_image(object(), "vlm", "/a.png"), calls
            except Exception as exc:
                return exc, calls

    def test_the_quick_steps_that_need_no_model_come_before_the_model(self):
        result, calls = self.run_analysis()
        self.assertEqual(calls, ["plan", "preprocess", "features", "ocr", "model"])
        self.assertEqual(result.ocr_text, ["EXIT"])

    def test_when_the_text_cannot_be_read_the_model_is_never_asked(self):
        def fail(*args, **kwargs):
            raise ImageProblem(OCR_FAILED, "no text for you")
        result, calls = self.run_analysis(extract_text_blocks=fail)
        self.assertIsInstance(result, ImageProblem)
        self.assertNotIn("model", calls)

    def test_an_error_from_pillow_while_preparing_becomes_a_problem_with_the_image(self):
        def fail(*args, **kwargs):
            raise UnidentifiedImageError("cannot identify image file")
        with mock.patch.object(config, "FILE_SETTLE_SECONDS", 0):
            result, calls = self.run_analysis(preprocess_image=fail)
        self.assertEqual((type(result), result.kind), (ImageProblem, UNREADABLE))
        self.assertNotIn("model", calls)

    def test_other_errors_are_passed_on_unchanged(self):
        def fail(*args, **kwargs):
            raise PermissionError(13, "denied")
        result, _ = self.run_analysis(extract_image_features=fail)
        self.assertIsInstance(result, PermissionError)

    def test_a_big_image_is_read_the_gentle_way(self):
        seen = {}

        def preprocess(path, gentle):
            seen["gentle"] = gentle
            return b"jpeg"
        self.run_analysis(plan_image=lambda path: memory.GENTLE, preprocess_image=preprocess)
        self.assertTrue(seen["gentle"])


if __name__ == "__main__":
    unittest.main()
