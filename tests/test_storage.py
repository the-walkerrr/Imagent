"""Telling a folder on this computer's own disk from one on a network or an external drive."""

import os
import unittest
from unittest import mock

from os_specific import storage as os_storage
from pipeline import storage, validation

MOUNT_OUTPUT = """/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)
/dev/disk3s5 on /System/Volumes/Data (apfs, local, journaled, nobrowse)
//user@nas._smb._tcp.local/Photos on /Volumes/My Photos (smbfs, nodev, nosuid, mounted by user)
/dev/disk5s1 on /Volumes/USB Drive (exfat, local, nodev, nosuid, noowners)
"""
MOUNTINFO = r"""36 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw
40 36 8:17 / /media/usb rw,nosuid shared:7 - vfat /dev/sdb1 rw
41 36 0:45 / /mnt/nas\040share rw,relatime shared:9 - nfs4 server:/export rw,vers=4.2
"""


class ParsingTests(unittest.TestCase):
    def test_macos_mount_output(self):
        mounts = os_storage._parse_mount_output(MOUNT_OUTPUT)
        self.assertIn(("/Volumes/My Photos", "smbfs", "//user@nas._smb._tcp.local/Photos"), mounts)
        self.assertIn(("/", "apfs", "/dev/disk3s1s1"), mounts)
        self.assertEqual(len(mounts), 4)

    def test_linux_mountinfo_with_an_escaped_space(self):
        mounts = os_storage._parse_mountinfo(MOUNTINFO)
        self.assertIn(("/mnt/nas share", "nfs4", "server:/export"), mounts)
        self.assertIn(("/media/usb", "vfat", "/dev/sdb1"), mounts)

    def test_a_folder_belongs_to_the_mount_that_holds_it_most_closely(self):
        mounts = os_storage._parse_mount_output(MOUNT_OUTPUT)
        self.assertEqual(os_storage._owner("/Volumes/My Photos/2020/a", mounts)[1], "smbfs")
        self.assertEqual(os_storage._owner("/Users/me/Pictures", mounts)[0], "/")
        self.assertEqual(os_storage._owner("/Volumes/My PhotosOther", mounts)[0], "/")  # not inside "My Photos"


class ClassifyTests(unittest.TestCase):
    def classify(self, system, path, mounts, **patches):
        parse = "_parse_mount_output" if system == "Darwin" else "_parse_mountinfo"
        with mock.patch.object(os_storage.platform, "system", return_value=system), \
                mock.patch.object(os_storage.subprocess, "run", return_value=mock.Mock(stdout="")), \
                mock.patch.object(os_storage.Path, "read_text", return_value=""), \
                mock.patch.object(os_storage, parse, return_value=mounts), \
                mock.patch.object(os_storage, "_macos_is_external", patches.get("mac", mock.Mock(return_value=False))), \
                mock.patch.object(os_storage, "_linux_is_external", patches.get("linux", mock.Mock(return_value=False))):
            return os_storage.classify(path)

    def test_a_network_file_system_is_refused_on_both_systems(self):
        self.assertEqual(self.classify("Darwin", "/Volumes/Photos/a", [("/Volumes/Photos", "smbfs", "//x/y")]), storage.NETWORK)
        self.assertEqual(self.classify("Linux", "/mnt/nas/a", [("/mnt/nas", "nfs4", "server:/export")]), storage.NETWORK)

    def test_an_external_disk_is_refused_when_the_system_says_it_is_external(self):
        self.assertEqual(self.classify("Darwin", "/Volumes/U/a", [("/Volumes/U", "exfat", "/dev/disk5s1")], mac=mock.Mock(return_value=True)), storage.EXTERNAL)
        self.assertEqual(self.classify("Linux", "/media/usb/a", [("/media/usb", "vfat", "/dev/sdb1")], linux=mock.Mock(return_value=True)), storage.EXTERNAL)

    def test_the_computers_own_disk_is_allowed(self):
        self.assertIsNone(self.classify("Darwin", "/Users/me", [("/", "apfs", "/dev/disk3s1s1")]))
        self.assertIsNone(self.classify("Linux", "/home/me", [("/", "ext4", "/dev/sda1")]))

    def test_a_device_that_is_not_a_disk_is_not_asked_about_usb(self):
        linux = mock.Mock(return_value=True)
        self.assertIsNone(self.classify("Linux", "/mnt/x", [("/mnt/x", "tmpfs", "tmpfs")], linux=linux))
        linux.assert_not_called()

    def test_an_unknown_system_or_an_unknown_folder_is_allowed(self):
        self.assertIsNone(self.classify("Plan9", "/x", []))
        self.assertIsNone(self.classify("Linux", "/nowhere", []))


class StorageProblemTests(unittest.TestCase):
    def test_a_folder_that_cannot_be_checked_is_allowed(self):
        with mock.patch.object(storage, "classify", side_effect=OSError("no diskutil")):
            self.assertIsNone(storage.storage_problem("/x"))

    def test_the_message_says_what_is_wrong_and_what_to_do(self):
        with mock.patch.object(storage, "classify", return_value=storage.NETWORK):
            self.assertIn("network drive", storage.storage_problem("/x"))
        with mock.patch.object(storage, "classify", return_value=storage.EXTERNAL):
            message = storage.storage_problem("/x")
        self.assertIn("external or removable drive", message)
        self.assertIn("Copy the pictures to this computer", message)

    def test_a_folder_on_such_a_drive_is_refused_before_scanning(self):
        with mock.patch.object(validation, "storage_problem", return_value="it is on a network drive"):
            with self.assertRaises(SystemExit):
                validation.validate_folder(os.path.expanduser("~"))

    def test_an_ordinary_folder_is_accepted(self):
        with mock.patch.object(validation, "storage_problem", return_value=None):
            self.assertEqual(validation.validate_folder(os.path.expanduser("~")), validation.Path(os.path.expanduser("~")).resolve())



class RootOverlapTests(unittest.TestCase):
    """Whether two folders are the same one, as the file system of each operating system sees it."""

    def overlap(self, system, root, known):
        from pathlib import Path
        from os_specific import files
        with mock.patch.object(files.platform, "system", return_value=system):
            return validation.find_overlapping_root(Path(root), known)

    def test_a_folder_inside_a_known_one_overlaps_whatever_the_system(self):
        for system in ("Darwin", "Windows", "Linux"):
            self.assertEqual(self.overlap(system, "/pics/2020", ["/pics"]), "/pics", system)

    def test_the_same_folder_is_a_rescan_not_an_overlap(self):
        self.assertIsNone(self.overlap("Linux", "/pics", ["/pics"]))

    def test_a_folder_that_differs_only_by_case_overlaps_on_macos_and_not_on_linux(self):
        self.assertEqual(self.overlap("Darwin", "/Pics/2020", ["/pics"]), "/pics")
        self.assertIsNone(self.overlap("Linux", "/Pics/2020", ["/pics"]))

    def test_a_folder_whose_accent_is_written_the_other_way_overlaps_on_macos_and_not_on_linux(self):
        import unicodedata
        typed = unicodedata.normalize("NFD", "/caf\u00e9/2020")
        stored = unicodedata.normalize("NFC", "/caf\u00e9")
        self.assertEqual(self.overlap("Darwin", typed, [stored]), stored)
        self.assertIsNone(self.overlap("Linux", typed, [stored]))  # on Linux they are two folders


if __name__ == "__main__":
    unittest.main()
