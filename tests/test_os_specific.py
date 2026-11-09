"""The code that differs by operating system, one platform at a time: each is tested by saying which system this is, so every system is covered on every computer."""

import os
import tempfile
import unicodedata
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from os_specific import convert, files, locking, memory, process, system_name

MEMINFO = "MemTotal:       16000000 kB\nMemAvailable:    8000000 kB\n"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class Cgroup(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "cgroup"
        self.root.mkdir()
        self.proc = Path(tmp.name) / "proc_cgroup"

    def left(self):
        return memory._cgroup_available(self.root, self.proc)


class CgroupV2Tests(Cgroup):
    def setUp(self):
        super().setUp()
        write(self.root / "cgroup.controllers", "cpu memory io\n")  # what tells v2 from v1

    def limit(self, folder, maximum, current, inactive=0):
        write(self.root / folder / "memory.max", f"{maximum}\n")
        write(self.root / folder / "memory.current", f"{current}\n")
        write(self.root / folder / "memory.stat", f"anon 1\ninactive_file {inactive}\nactive_file 5\n")

    def test_a_container_that_sees_its_own_cgroup_as_the_top_of_the_tree(self):
        write(self.proc, "0::/\n")
        self.limit(".", 2_000, 500)
        self.assertEqual(self.left(), 1_500)

    def test_cache_the_kernel_can_take_back_is_not_counted_as_used(self):
        write(self.proc, "0::/\n")
        self.limit(".", 2_000, 1_000, inactive=400)
        self.assertEqual(self.left(), 1_400)  # 2000 - (1000 - 400)

    def test_the_tightest_limit_on_the_way_up_counts(self):
        write(self.proc, "12:cpu:/x\n0::/user.slice/app\n")
        self.limit("user.slice/app", "max", 10)  # no limit of its own
        self.limit("user.slice", 1_000, 400)  # 600 left
        self.limit(".", 5_000, 100)  # 4900 left
        self.assertEqual(self.left(), 600)
        self.limit("user.slice/app", 500, 450)  # now the process's own cgroup is the tightest: 50 left
        self.assertEqual(self.left(), 50)

    def test_no_limit_means_nothing_to_report(self):
        write(self.proc, "0::/\n")
        self.limit(".", "max", 500)
        self.assertIsNone(self.left())

    def test_a_cgroup_that_is_over_its_limit_has_nothing_left_not_a_negative_number(self):
        write(self.proc, "0::/\n")
        self.limit(".", 100, 300)
        self.assertEqual(self.left(), 0)

    def test_a_cgroup_path_that_is_not_there_falls_back_to_the_top(self):
        write(self.proc, "0::/not/mounted/here\n")
        self.limit(".", 2_000, 500)
        self.assertEqual(self.left(), 1_500)

    def test_unreadable_files_are_no_limit_not_an_error(self):
        self.assertIsNone(self.left())  # no /proc file at all
        write(self.proc, "0::/\n")
        write(self.root / "memory.max", "garbage")
        self.assertIsNone(self.left())


class CgroupV1Tests(Cgroup):
    def limit(self, folder, maximum, usage, inactive=0):
        write(self.root / "memory" / folder / "memory.limit_in_bytes", f"{maximum}\n")
        write(self.root / "memory" / folder / "memory.usage_in_bytes", f"{usage}\n")
        write(self.root / "memory" / folder / "memory.stat", f"cache 9\ntotal_inactive_file {inactive}\n")

    def test_a_limit_is_read_from_the_memory_controller(self):
        write(self.proc, "4:memory:/docker/abc\n3:cpu,cpuacct:/docker/abc\n")
        self.limit("docker/abc", 1_000, 300, inactive=50)
        self.assertEqual(self.left(), 750)

    def test_a_container_that_has_its_cgroup_mounted_as_the_top(self):
        write(self.proc, "4:memory:/docker/abc\n")  # the path is not under the mount: its top is the container's own cgroup
        self.limit(".", 1_000, 300)
        self.assertEqual(self.left(), 700)

    def test_the_number_v1_writes_for_no_limit_is_no_limit(self):
        write(self.proc, "4:memory:/\n")
        self.limit(".", 9223372036854771712, 300)
        self.assertIsNone(self.left())

    def test_a_malformed_line_is_ignored(self):
        write(self.proc, "nonsense\n4:memory:/g\n")
        self.limit("g", 100, 10)
        self.assertEqual(self.left(), 90)


class AvailableMemoryTests(unittest.TestCase):
    def test_on_linux_the_smaller_of_the_computer_and_the_cgroup_counts(self):
        with mock.patch.object(memory.platform, "system", return_value="Linux"), mock.patch.object(memory.Path, "read_text", return_value=MEMINFO):
            with mock.patch.object(memory, "_cgroup_available", return_value=1_000):
                self.assertEqual(memory.available_memory(), 1_000)  # a 1 KB limit inside an 8 GB computer
            with mock.patch.object(memory, "_cgroup_available", return_value=10 ** 15):
                self.assertEqual(memory.available_memory(), 8_000_000 * 1024)
            with mock.patch.object(memory, "_cgroup_available", return_value=None):
                self.assertEqual(memory.available_memory(), 8_000_000 * 1024)

    def test_on_linux_a_cgroup_limit_counts_even_when_meminfo_cannot_be_read(self):
        with mock.patch.object(memory.platform, "system", return_value="Linux"), \
                mock.patch.object(memory.Path, "read_text", return_value="nonsense"), mock.patch.object(memory, "_cgroup_available", return_value=77):
            self.assertEqual(memory.available_memory(), 77)

    def test_a_system_that_will_not_say_gives_none(self):
        with mock.patch.object(memory.platform, "system", return_value="Plan9"):
            self.assertIsNone(memory.available_memory())
        with mock.patch.object(memory.platform, "system", return_value="Darwin"), mock.patch.object(memory.subprocess, "run", side_effect=OSError("no vm_stat")):
            self.assertIsNone(memory.available_memory())


def stat_result(**fields):
    return SimpleNamespace(**fields)


class NotDownloadedTests(unittest.TestCase):
    def test_a_macos_file_whose_content_is_in_icloud_is_not_downloaded(self):
        with mock.patch.object(files.platform, "system", return_value="Darwin"):
            self.assertTrue(files.is_not_downloaded(stat_result(st_flags=files._SF_DATALESS)))
            self.assertTrue(files.is_not_downloaded(stat_result(st_flags=files._SF_DATALESS | 0x20)))
            self.assertFalse(files.is_not_downloaded(stat_result(st_flags=0)))
            self.assertFalse(files.is_not_downloaded(stat_result()))  # a system whose stat has no flags

    def test_a_windows_file_that_the_cloud_would_have_to_send_is_not_downloaded(self):
        with mock.patch.object(files.platform, "system", return_value="Windows"):
            for attribute in (files._FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS, files._FILE_ATTRIBUTE_RECALL_ON_OPEN, files.stat.FILE_ATTRIBUTE_OFFLINE):
                self.assertTrue(files.is_not_downloaded(stat_result(st_file_attributes=attribute | files.stat.FILE_ATTRIBUTE_ARCHIVE)), hex(attribute))
            self.assertFalse(files.is_not_downloaded(stat_result(st_file_attributes=files.stat.FILE_ATTRIBUTE_ARCHIVE)))
            self.assertFalse(files.is_not_downloaded(stat_result()))

    def test_the_other_systems_have_no_such_files(self):
        with mock.patch.object(files.platform, "system", return_value="Linux"):
            self.assertFalse(files.is_not_downloaded(stat_result(st_flags=files._SF_DATALESS, st_file_attributes=files._FILE_ATTRIBUTE_RECALL_ON_OPEN)))

    def test_the_constants_are_the_ones_the_systems_define(self):
        self.assertEqual(files._SF_DATALESS, 0x40000000)  # sys/stat.h on macOS
        self.assertEqual(files._FILE_ATTRIBUTE_RECALL_ON_OPEN, 0x40000)  # winnt.h
        self.assertEqual(files._FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS, 0x400000)  # winnt.h
        self.assertEqual(files._UF_HIDDEN, 0x8000)  # sys/stat.h on macOS


class HiddenFolderTests(unittest.TestCase):
    def test_a_windows_folder_with_the_hidden_attribute_is_hidden(self):
        with mock.patch.object(files.platform, "system", return_value="Windows"):
            with mock.patch.object(files.os, "stat", return_value=stat_result(st_file_attributes=files.stat.FILE_ATTRIBUTE_HIDDEN | 16)):
                self.assertTrue(files.is_hidden_folder("C:\\Users\\me\\AppData"))
            with mock.patch.object(files.os, "stat", return_value=stat_result(st_file_attributes=16)):
                self.assertFalse(files.is_hidden_folder("C:\\Users\\me\\Pictures"))

    def test_a_macos_folder_with_the_hidden_flag_is_hidden(self):
        with mock.patch.object(files.platform, "system", return_value="Darwin"):
            with mock.patch.object(files.os, "stat", return_value=stat_result(st_flags=files._UF_HIDDEN)):
                self.assertTrue(files.is_hidden_folder("/Users/me/Library"))
            with mock.patch.object(files.os, "stat", return_value=stat_result(st_flags=0)):
                self.assertFalse(files.is_hidden_folder("/Users/me/Pictures"))

    def test_linux_has_no_such_attribute_and_the_folder_is_not_even_examined(self):
        with mock.patch.object(files.platform, "system", return_value="Linux"), mock.patch.object(files.os, "stat", side_effect=AssertionError("examined")):
            self.assertFalse(files.is_hidden_folder("/home/me/Pictures"))

    def test_a_folder_that_cannot_be_examined_is_not_hidden(self):
        with mock.patch.object(files.platform, "system", return_value="Windows"), mock.patch.object(files.os, "stat", side_effect=OSError("gone")):
            self.assertFalse(files.is_hidden_folder("C:\\x"))

    def test_a_real_folder_is_not_hidden_on_this_computer(self):
        self.assertFalse(files.is_hidden_folder(tempfile.gettempdir()))


class ComparablePathTests(unittest.TestCase):
    NFC = unicodedata.normalize("NFC", "/Photos/caf\u00e9")
    NFD = unicodedata.normalize("NFD", "/Photos/caf\u00e9")

    def test_macos_does_not_tell_case_or_the_way_an_accent_is_written(self):
        with mock.patch.object(files.platform, "system", return_value="Darwin"):
            self.assertNotEqual(self.NFC, self.NFD)
            self.assertEqual(files.comparable_path(self.NFC), files.comparable_path(self.NFD))
            self.assertEqual(files.comparable_path("/Users/Me/Pics/"), files.comparable_path("/users/me/pics"))

    def test_windows_does_not_tell_case_but_does_tell_an_accent_written_two_ways(self):
        with mock.patch.object(files.platform, "system", return_value="Windows"), mock.patch.object(files.os.path, "normcase", side_effect=lambda p: p.replace("/", "\\").lower()):
            self.assertEqual(files.comparable_path("C:\\Users\\Me"), files.comparable_path("c:\\users\\me"))
            self.assertNotEqual(files.comparable_path(self.NFC), files.comparable_path(self.NFD))

    def test_linux_tells_everything_apart(self):
        with mock.patch.object(files.platform, "system", return_value="Linux"):
            self.assertNotEqual(files.comparable_path("/Photos"), files.comparable_path("/photos"))
            self.assertNotEqual(files.comparable_path(self.NFC), files.comparable_path(self.NFD))
            self.assertEqual(files.comparable_path("/photos/"), "/photos")

    def test_it_is_for_comparing_only_it_never_changes_what_is_stored(self):
        with mock.patch.object(files.platform, "system", return_value="Darwin"):
            original = self.NFD
            files.comparable_path(original)
        self.assertEqual(original, unicodedata.normalize("NFD", "/Photos/caf\u00e9"))


class ProcessTests(unittest.TestCase):
    def test_windows_starts_it_detached_and_the_others_in_a_session_of_their_own(self):
        with mock.patch.object(process.platform, "system", return_value="Windows"):
            self.assertEqual(process.background_process_options(), {"creationflags": 0x00000008})
        for system in ("Darwin", "Linux"):
            with mock.patch.object(process.platform, "system", return_value=system):
                self.assertEqual(process.background_process_options(), {"start_new_session": True})


class LockingTests(unittest.TestCase):
    def test_a_second_lock_on_the_same_file_is_refused_until_the_first_is_released(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "a.lock")
            with open(path, "a+b") as first, open(path, "a+b") as second:
                locking.lock(first)
                with self.assertRaises(OSError):
                    locking.lock(second)
                locking.unlock(first)
                locking.lock(second)
                locking.unlock(second)


class ConvertCommandTests(unittest.TestCase):
    def test_each_system_has_its_own_tool_and_linux_has_none(self):
        with mock.patch.object(convert.platform, "system", return_value="Darwin"), mock.patch.object(convert.shutil, "which", return_value="/usr/bin/sips"):
            command, environment = convert.command("/a.heic", "/b.png")
            self.assertEqual((command[0], command[-1], environment), ("sips", "/b.png", {}))
        with mock.patch.object(convert.platform, "system", return_value="Windows"), mock.patch.object(convert.shutil, "which", return_value="powershell"):
            command, environment = convert.command("C:\\a.heic", "C:\\b.png")
            self.assertEqual((command[0], environment), ("powershell", {"IMAGENT_IN": "C:\\a.heic", "IMAGENT_OUT": "C:\\b.png"}))
        with mock.patch.object(convert.platform, "system", return_value="Linux"):
            self.assertIsNone(convert.command("/a.heic", "/b.png"))
        with mock.patch.object(convert.platform, "system", return_value="Darwin"), mock.patch.object(convert.shutil, "which", return_value=None):
            self.assertIsNone(convert.command("/a.heic", "/b.png"))  # a Mac without the tool

    def test_the_system_is_named_for_messages(self):
        self.assertTrue(system_name())


if __name__ == "__main__":
    unittest.main()
