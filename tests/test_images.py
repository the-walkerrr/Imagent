"""Reading images: how much memory they may use, what is wrong with one that cannot be opened, and the steps of reading its text."""

import functools
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image, UnidentifiedImageError
from PIL.PngImagePlugin import PngImageFile

import config
from image_processing import memory, ocr
from image_processing.features import extract_image_features
from image_processing.preprocess import preprocess_image
from os_specific import memory as os_memory
from image_processing.problems import OCR_FAILED, TOO_LARGE, UNREADABLE, UNSUPPORTED, ImageProblem, explain, problem
from pipeline.failure import IndexingFailure
from tests.support import IsolatedHome
from vlm import analyzer

MEMINFO = "MemTotal:       16000000 kB\nMemFree:          500000 kB\nMemAvailable:    8000000 kB\nBuffers: 1 kB\n"
VM_STAT = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:                               100.\n"
           "Pages active:                            9999.\nPages inactive:                           50.\n"
           "Pages speculative:                        10.\nPages purgeable:                           7.\n")


class MemoryTests(unittest.TestCase):
    def test_free_memory_is_read_from_what_each_system_says(self):
        self.assertEqual(os_memory._parse_meminfo(MEMINFO), 8000000 * 1024)
        self.assertEqual(os_memory._parse_vm_stat(VM_STAT), (100 + 50 + 10) * 16384)
        self.assertIsNone(os_memory._parse_meminfo("MemTotal: 1 kB\n"))
        self.assertIsNone(os_memory._parse_vm_stat("nonsense"))

    def test_an_image_fits_when_the_memory_it_needs_is_within_the_share_and_this_scales_with_the_memory(self):
        pixels, per_pixel = 10_000_000, memory.peak_bytes_per_pixel("RGB")  # a 10 megapixel image
        with mock.patch.object(config, "IMAGE_MEMORY_SHARE", 1.0):
            self.assertTrue(memory.fits_in_memory(5000, 2000, per_pixel, pixels * per_pixel))
            self.assertFalse(memory.fits_in_memory(5000, 2000, per_pixel, pixels * per_pixel - 1))
            self.assertTrue(memory.fits_in_memory(5000, 2000, per_pixel, pixels * 100))  # more memory, bigger images allowed

    def test_the_share_of_free_memory_an_image_may_use_is_a_setting(self):
        per_pixel = memory.peak_bytes_per_pixel("RGB")
        free = 10_000_000 * per_pixel
        with mock.patch.object(config, "IMAGE_MEMORY_SHARE", 1.0):
            self.assertTrue(memory.fits_in_memory(5000, 2000, per_pixel, free))
        with mock.patch.object(config, "IMAGE_MEMORY_SHARE", 0.5):
            self.assertFalse(memory.fits_in_memory(5000, 2000, per_pixel, free))

    def test_a_share_given_for_one_check_replaces_the_setting(self):
        per_pixel = memory.peak_bytes_per_pixel("RGB")
        with mock.patch.object(config, "IMAGE_MEMORY_SHARE", 0.5):
            free = 10_000_000 * per_pixel * 2 - 1  # 10 megapixels need more than half of this
            self.assertFalse(memory.fits_in_memory(5000, 2000, per_pixel, free))
            self.assertTrue(memory.fits_in_memory(5000, 2000, per_pixel, free, 0.75))

    def test_what_a_pixel_needs_depends_on_the_mode_of_the_picture(self):
        # Pillow's storage (1, 2 or 4 bytes) plus the larger of the RGB copy and the turned copy, plus what the decoder keeps.
        needs = {mode: memory.peak_bytes_per_pixel(mode) for mode in ("RGB", "RGBA", "CMYK", "LA", "I", "F", "L", "P", "1", "I;16")}
        self.assertEqual(needs, {"RGB": 5, "RGBA": 9, "CMYK": 9, "LA": 9, "I": 9, "F": 9, "L": 6, "P": 6, "1": 6, "I;16": 7})
        self.assertEqual(memory.peak_bytes_per_pixel("an unknown mode"), 9)  # taken to be one of 4 bytes

    def test_a_picture_that_is_turned_needs_a_copy_of_its_own_size_only_where_that_is_more_than_the_rgb_copy(self):
        self.assertEqual(memory.peak_bytes_per_pixel("RGB", turned=True), 9)  # no RGB copy, but the turned one
        self.assertEqual(memory.peak_bytes_per_pixel("RGBA", turned=True), memory.peak_bytes_per_pixel("RGBA"))  # the copies are not held together
        self.assertEqual(memory.peak_bytes_per_pixel("L", turned=True), memory.peak_bytes_per_pixel("L"))

    def test_when_free_memory_is_not_known_pillows_own_limit_decides(self):
        limit = memory.PIXEL_LIMIT_WHEN_MEMORY_UNKNOWN
        self.assertIsNotNone(limit)  # read from Pillow when the package is imported
        self.assertTrue(memory.fits_in_memory(limit, 1, 9, None))
        self.assertFalse(memory.fits_in_memory(limit + 1, 1, 9, None))
        with mock.patch.object(memory, "PIXEL_LIMIT_WHEN_MEMORY_UNKNOWN", None):
            self.assertTrue(memory.fits_in_memory(100_000, 100_000, 9, None))  # a Pillow with no limit of its own: nothing to check against

    def test_pillows_own_pixel_limit_is_off(self):
        self.assertIsNone(Image.MAX_IMAGE_PIXELS)


class CheckImageSizeTests(IsolatedHome):
    def test_an_image_that_fits_is_let_through(self):
        image = self.make_image(self.work / "a.png")
        with mock.patch.object(memory, "available_memory", return_value=10**12):
            self.assertIsNone(memory.check_image_size(str(image)))

    def test_when_free_memory_is_not_known_a_small_image_is_let_through(self):
        image = self.make_image(self.work / "a.png")
        with mock.patch.object(memory, "available_memory", return_value=None):
            self.assertIsNone(memory.check_image_size(str(image)))

    def test_when_free_memory_is_not_known_pillows_own_limit_stands_in_for_it(self):
        image = self.make_image(self.work / "a.png")  # 64 pixels
        with mock.patch.object(memory, "available_memory", return_value=None), \
                mock.patch.object(memory, "PIXEL_LIMIT_WHEN_MEMORY_UNKNOWN", 63), self.assertRaises(ImageProblem) as raised:
            memory.check_image_size(str(image))
        self.assertEqual((raised.exception.kind, raised.exception.park), (TOO_LARGE, False))
        self.assertIn("does not say how much memory is free", str(raised.exception))
        with mock.patch.object(memory, "available_memory", return_value=None), mock.patch.object(memory, "PIXEL_LIMIT_WHEN_MEMORY_UNKNOWN", 64):
            self.assertIsNone(memory.check_image_size(str(image)))  # exactly the limit is still let through

    def test_the_limit_used_when_memory_is_not_known_is_twice_pillows_own_default_read_from_pillow(self):
        import importlib
        import image_processing
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 1000):
            importlib.reload(image_processing)
            self.assertEqual(image_processing.PIXEL_LIMIT_WHEN_MEMORY_UNKNOWN, 2000)
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", None):
            importlib.reload(image_processing)
            self.assertIsNone(image_processing.PIXEL_LIMIT_WHEN_MEMORY_UNKNOWN)
        importlib.reload(image_processing)  # Pillow's limit is off again, as in production

    def test_exif_of_a_png_without_an_exif_chunk_is_not_asked_for_when_that_would_decode_a_picture_too_large(self):
        path = self.make_image(self.work / "a.png")  # 64 pixels of RGB: 320 bytes
        with Image.open(path) as img:
            with mock.patch.object(memory, "available_memory", return_value=100):
                self.assertFalse(memory.exif_can_be_read(img))
            with mock.patch.object(memory, "available_memory", return_value=10**12):
                self.assertTrue(memory.exif_can_be_read(img))

    def test_exif_is_asked_for_in_every_other_case(self):
        jpeg = self.make_image(self.work / "a.jpg")
        with Image.open(jpeg) as img, mock.patch.object(memory, "available_memory", return_value=1):
            self.assertTrue(memory.exif_can_be_read(img))  # a JPEG keeps its EXIF in the header
        with_exif = self.work / "exif.png"
        exif = Image.Exif()
        exif[0x010F] = "Maker"
        Image.new("RGB", (8, 8)).save(with_exif, exif=exif)
        with Image.open(with_exif) as img, mock.patch.object(memory, "available_memory", return_value=1):
            self.assertTrue(memory.exif_can_be_read(img))  # a PNG with an EXIF chunk in its header too

    def test_a_larger_share_lets_an_image_through_that_the_usual_share_refuses(self):
        image = self.make_image(self.work / "a.png")  # 64 pixels of RGB: 320 bytes
        with mock.patch.object(memory, "available_memory", return_value=500):
            with self.assertRaises(ImageProblem):
                memory.check_image_size(str(image))  # 50% of 500 is 250 bytes
            self.assertIsNone(memory.check_image_size(str(image), 0.75))  # 75% is 375
            with self.assertRaises(ImageProblem):
                memory.check_image_size(str(image), 0.6)  # 300 bytes: still not enough

    def test_the_mode_of_the_picture_decides_how_much_memory_it_needs(self):
        rgb, rgba = self.work / "rgb.png", self.work / "rgba.png"
        Image.new("RGB", (8, 8)).save(rgb)
        Image.new("RGBA", (8, 8)).save(rgba)
        with mock.patch.object(memory, "available_memory", return_value=800):  # 50% is 400 bytes: RGB needs 320, RGBA 576
            self.assertIsNone(memory.check_image_size(str(rgb)))
            with self.assertRaises(ImageProblem):
                memory.check_image_size(str(rgba))

    def test_a_jpeg_stored_sideways_needs_the_memory_of_a_turned_copy(self):
        sideways = self.work / "sideways.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6
        Image.new("RGB", (8, 8)).save(sideways, exif=exif)
        upright = self.make_image(self.work / "upright.jpg")
        with mock.patch.object(memory, "available_memory", return_value=800):  # 50% is 400 bytes: 8 x 8 needs 320 upright, 576 turned
            self.assertIsNone(memory.check_image_size(str(upright)))
            with self.assertRaises(ImageProblem):
                memory.check_image_size(str(sideways))

    def test_the_orientation_of_a_png_without_exif_is_not_asked_for_because_that_decodes_the_picture(self):
        path = self.make_image(self.work / "a.png")
        with Image.open(path) as img, mock.patch.object(PngImageFile, "getexif", side_effect=AssertionError("decoded the picture")):
            self.assertFalse(memory._is_turned(img))

    def test_an_image_too_large_is_refused_with_the_numbers(self):
        image = self.make_image(self.work / "a.png")
        with mock.patch.object(memory, "available_memory", return_value=100), self.assertRaises(ImageProblem) as raised:
            memory.check_image_size(str(image))
        self.assertEqual((raised.exception.kind, raised.exception.park, raised.exception.retry_when), (TOO_LARGE, False, "on_request"))  # not parked: tried again by every run
        self.assertIn("megapixels", str(raised.exception))
        self.assertIn("Close other programs", str(raised.exception))
        self.assertIsInstance(raised.exception, IndexingFailure)

    def test_a_file_that_is_not_an_image_is_unreadable(self):
        bad = self.work / "bad.png"
        bad.write_bytes(b"this is not a picture")
        with self.assertRaises(ImageProblem) as raised:
            memory.check_image_size(str(bad))
        self.assertEqual((raised.exception.kind, raised.exception.retry_when), (UNREADABLE, "file_changed"))

    def test_a_format_without_a_decoder_is_unsupported_not_damaged(self):
        odd = self.work / "picture.zzz"
        odd.write_bytes(b"whatever")
        with self.assertRaises(ImageProblem) as raised:
            memory.check_image_size(str(odd))
        self.assertEqual((raised.exception.kind, raised.exception.retry_when), (UNSUPPORTED, "on_request"))
        self.assertIn(".zzz", str(raised.exception))


class ExplainTests(unittest.TestCase):
    def test_errors_that_are_not_about_the_image_are_left_alone(self):
        self.assertIsNone(explain(PermissionError(13, "denied"), "/a.png"))  # a real I/O error carries an error number
        self.assertIsNone(explain(ValueError("x"), "/a.png"))

    def test_pillows_errors_are_told_from_memory_trouble(self):
        self.assertEqual(explain(OSError("image file is truncated"), "/a.png").kind, UNREADABLE)
        self.assertEqual(explain(SyntaxError("broken PNG file"), "/a.png").kind, UNREADABLE)
        self.assertEqual(explain(MemoryError(), "/a.png").kind, TOO_LARGE)

    def test_explaining_a_damaged_file_never_waits(self):
        with mock.patch("time.sleep", side_effect=AssertionError("it waited")):
            self.assertEqual(explain(OSError("image file is truncated"), "/a.png").kind, UNREADABLE)


class PreparingTests(IsolatedHome):
    def make_jpeg(self, size=(3000, 2000), orientation=None):
        path = self.work / "big.jpg"
        exif = Image.Exif()
        if orientation:
            exif[0x0112] = orientation
        Image.effect_noise(size, 30).convert("RGB").save(path, quality=85, exif=exif)
        return str(path)

    def test_the_picture_for_the_model_is_no_larger_than_the_long_edge_allows(self):
        picture = Image.open(io.BytesIO(preprocess_image(self.make_jpeg())))
        self.assertEqual(max(picture.size), config.IMAGE_MAX_LONG_EDGE)

    def test_a_picture_stored_sideways_is_sent_upright(self):
        picture = Image.open(io.BytesIO(preprocess_image(self.make_jpeg((3000, 2000), orientation=6))))
        self.assertLess(picture.size[0], picture.size[1])  # stored as landscape, displayed as portrait

    def test_a_picture_that_is_not_rgb_is_sent_as_a_jpeg_in_rgb(self):
        for mode in ("RGBA", "P", "L"):
            path = self.work / f"{mode}.png"
            Image.new(mode, (40, 30)).save(path)
            self.assertEqual(Image.open(io.BytesIO(preprocess_image(str(path)))).mode, "RGB", mode)

    def test_a_small_rgb_picture_is_sent_whole(self):
        path = self.work / "small.png"
        Image.new("RGB", (40, 30), (10, 20, 30)).save(path)
        self.assertEqual(Image.open(io.BytesIO(preprocess_image(str(path)))).size, (40, 30))

    def test_the_features_judge_a_picture_as_displayed(self):
        path = self.make_jpeg((3000, 2000), orientation=6)  # stored sideways: displayed as portrait
        self.assertEqual(extract_image_features(path)["orientation"], "portrait")


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

    def read(self, engine, size=(50, 30), path=None):
        path = path or self.work / "a.png"
        Image.new("RGB", size, (200, 100, 50)).save(path)
        with mock.patch.object(ocr, "_engine", functools.lru_cache(maxsize=1)(lambda: engine)):
            return ocr.extract_text_blocks(str(path))

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
            self.read(engine, size=(50, 30))
        self.assertTrue(all(max(shape[:2]) <= 20 for shape in engine.shapes))

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
        with mock.patch.object(ocr, "_engine", functools.lru_cache(maxsize=1)(lambda: engine)):
            with self.assertRaises(ImageProblem) as raised:
                ocr.extract_text_blocks(str(bad))
        self.assertEqual(raised.exception.kind, UNREADABLE)
        self.assertEqual(engine.shapes, [])

    def test_tiles_of_a_picture_stored_sideways_are_turned_upright(self):
        path = self.work / "turned.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6
        Image.new("RGB", (40, 20)).save(path, exif=exif)
        engine = FakeEngine(max_side_len=30)  # smaller than the picture: read in tiles, here one as big as the whole of it
        with mock.patch.object(config, "OCR_TILE_SHARE", 2.0), mock.patch.object(ocr, "_engine", functools.lru_cache(maxsize=1)(lambda: engine)):
            ocr.extract_text_blocks(str(path))
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
            "check_image_size": record("check"), "preprocess_image": record("preprocess", b"jpeg"),
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
        self.assertEqual(calls, ["check", "preprocess", "features", "ocr", "model"])
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
        result, calls = self.run_analysis(preprocess_image=fail)
        self.assertEqual((type(result), result.kind), (ImageProblem, UNREADABLE))
        self.assertNotIn("model", calls)

    def test_other_errors_are_passed_on_unchanged(self):
        def fail(*args, **kwargs):
            raise PermissionError(13, "denied")
        result, _ = self.run_analysis(extract_image_features=fail)
        self.assertIsInstance(result, PermissionError)


class LargeImageFreesTheModelsTests(unittest.TestCase):
    """An image too large for the memory free is checked again after every loaded model is unloaded, whatever setup found,
    and once more with a larger share of the memory free."""

    def plan(self, outcomes, loaded=("vision", "embed")):
        """Run ``_check_image_size`` with ``check_image_size`` answering from ``outcomes`` (an exception is raised, ``None`` is a pass); returns what happened."""
        answers = list(outcomes)
        seen = {"stages": [], "unloaded": [], "asked": 0, "shares": []}

        def check_image_size(path, share=None):
            seen["asked"] += 1
            seen["shares"].append(share)
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
        client = object()
        with mock.patch.object(analyzer, "check_image_size", check_image_size), \
                mock.patch.object(analyzer, "other_models_running", mock.Mock(return_value=list(loaded))) as running, \
                mock.patch.object(analyzer, "unload_model", lambda c, name: seen["unloaded"].append(name)), \
                mock.patch.object(analyzer.time, "sleep") as sleep:
            try:
                analyzer._check_image_size(client, "/a.png", seen["stages"].append)
            except ImageProblem as exc:
                seen["problem"] = exc
        seen["running"], seen["sleep"] = running, sleep
        return seen

    def too_large(self):
        return problem(TOO_LARGE, "The picture has 900 megapixels, more than the memory free right now allows.", park=False)

    def test_every_loaded_model_is_unloaded_and_the_size_is_checked_again(self):
        seen = self.plan([self.too_large(), None])
        self.assertNotIn("problem", seen)
        self.assertEqual(seen["unloaded"], ["vision", "embed"])
        self.assertEqual(seen["asked"], 2)
        seen["running"].assert_called_once_with(mock.ANY, [])  # nothing is kept: every model goes
        seen["sleep"].assert_called_once_with(config.OLLAMA_RECOVERY_WAIT)
        self.assertEqual(seen["stages"], ["freeing memory for a large image"])

    def test_one_model_at_a_time_makes_no_difference(self):
        with mock.patch.object(config, "ONE_MODEL_AT_A_TIME", False):
            self.assertEqual(self.plan([self.too_large(), None])["unloaded"], ["vision", "embed"])
        with mock.patch.object(config, "ONE_MODEL_AT_A_TIME", True):
            self.assertEqual(self.plan([self.too_large(), None])["unloaded"], ["vision", "embed"])

    def test_an_image_that_fits_only_with_the_larger_share_is_let_through(self):
        seen = self.plan([self.too_large(), self.too_large(), None])
        self.assertNotIn("problem", seen)
        self.assertEqual(seen["unloaded"], ["vision", "embed"])  # the models are freed first, then the share is raised
        self.assertEqual(seen["shares"], [None, None, config.IMAGE_MEMORY_SHARE_LAST_TRY])

    def test_an_image_still_too_large_with_the_larger_share_is_refused(self):
        seen = self.plan([self.too_large(), self.too_large(), self.too_large()])
        self.assertEqual(seen["problem"].kind, TOO_LARGE)
        self.assertEqual((seen["unloaded"], seen["asked"]), (["vision", "embed"], 3))  # three checks, not a loop

    def test_when_no_model_is_loaded_there_is_nothing_to_free_and_no_wait_but_the_larger_share_is_tried(self):
        seen = self.plan([self.too_large(), None], loaded=())
        self.assertEqual((seen.get("problem"), seen["unloaded"], seen["stages"]), (None, [], []))
        self.assertEqual(seen["shares"], [None, config.IMAGE_MEMORY_SHARE_LAST_TRY])
        seen["sleep"].assert_not_called()
        seen = self.plan([self.too_large(), self.too_large()], loaded=())
        self.assertEqual((seen["problem"].kind, seen["asked"]), (TOO_LARGE, 2))

    def test_another_problem_after_the_models_are_gone_stops_the_steps(self):
        seen = self.plan([self.too_large(), problem(UNREADABLE, "damaged", retry_when="file_changed")])
        self.assertEqual((seen["problem"].kind, seen["asked"]), (UNREADABLE, 2))

    def test_an_image_that_fits_leaves_the_models_alone(self):
        seen = self.plan([None])
        self.assertEqual((seen.get("problem"), seen["unloaded"], seen["asked"]), (None, [], 1))
        seen["running"].assert_not_called()
        seen["sleep"].assert_not_called()

    def test_another_problem_with_the_image_does_not_unload_anything(self):
        seen = self.plan([problem(UNREADABLE, "damaged", retry_when="file_changed")])
        self.assertEqual((seen["problem"].kind, seen["unloaded"], seen["asked"]), (UNREADABLE, [], 1))
        seen["running"].assert_not_called()


if __name__ == "__main__":
    unittest.main()


class MetadataGuardTests(IsolatedHome):
    """Reading a picture's metadata is the first thing indexing does, before the size check: it must not decode a picture that does not fit."""

    def test_a_png_too_large_to_decode_is_not_decoded_to_read_its_metadata(self):
        from image_processing.metadata import extract_file_metadata
        path = self.make_image(self.work / "a.png")
        with mock.patch.object(memory, "available_memory", return_value=100), \
                mock.patch.object(PngImageFile, "load", side_effect=AssertionError("decoded the picture")):
            meta = extract_file_metadata(str(path))
        self.assertEqual((meta["width"], meta["height"]), (8, 8))  # the size comes from the header
        self.assertIsNone(meta["camera_make"])

    def test_a_png_that_fits_still_has_its_exif_read(self):
        from image_processing.metadata import extract_file_metadata
        path = self.work / "a.png"
        exif = Image.Exif()
        exif[0x010F] = "Maker"
        Image.new("RGB", (8, 8)).save(path, exif=exif)
        with mock.patch.object(memory, "available_memory", return_value=10**12):
            self.assertEqual(extract_file_metadata(str(path))["camera_make"], "Maker")


class MimeTypeTests(IsolatedHome):
    def test_the_formats_imagent_reads_have_a_type_whatever_the_computers_own_list_says(self):
        from image_processing.metadata import extract_file_metadata
        expected = {".heic": "image/heic", ".HEIF": "image/heif", ".avif": "image/avif", ".webp": "image/webp", ".jpg": "image/jpeg", ".tif": "image/tiff"}
        with mock.patch("mimetypes.guess_type", return_value=(None, None)):  # a computer whose list has none of them
            for suffix, mime in expected.items():
                path = self.work / f"a{suffix}"
                path.write_bytes(b"x")  # not a picture: the type comes from the name
                self.assertEqual(extract_file_metadata(str(path))["mime_type"], mime, suffix)

    def test_a_name_that_is_not_one_of_them_still_asks_the_computer(self):
        from image_processing.metadata import extract_file_metadata
        path = self.work / "a.xyz"
        path.write_bytes(b"x")
        with mock.patch("mimetypes.guess_type", return_value=("image/x-odd", None)):
            self.assertEqual(extract_file_metadata(str(path))["mime_type"], "image/x-odd")
