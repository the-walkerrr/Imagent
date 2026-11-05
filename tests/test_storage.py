"""Telling a folder on this computer's own disk from one on a network or an external drive."""

import os
import unittest
from unittest import mock

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
        mounts = storage._parse_mount_output(MOUNT_OUTPUT)
        self.assertIn(("/Volumes/My Photos", "smbfs", "//user@nas._smb._tcp.local/Photos"), mounts)
        self.assertIn(("/", "apfs", "/dev/disk3s1s1"), mounts)
        self.assertEqual(len(mounts), 4)

    def test_linux_mountinfo_with_an_escaped_space(self):
        mounts = storage._parse_mountinfo(MOUNTINFO)
        self.assertIn(("/mnt/nas share", "nfs4", "server:/export"), mounts)
        self.assertIn(("/media/usb", "vfat", "/dev/sdb1"), mounts)

    def test_a_folder_belongs_to_the_mount_that_holds_it_most_closely(self):
        mounts = storage._parse_mount_output(MOUNT_OUTPUT)
        self.assertEqual(storage._owner("/Volumes/My Photos/2020/a", mounts)[1], "smbfs")
        self.assertEqual(storage._owner("/Users/me/Pictures", mounts)[0], "/")
        self.assertEqual(storage._owner("/Volumes/My PhotosOther", mounts)[0], "/")  # not inside "My Photos"


class ClassifyTests(unittest.TestCase):
    def classify(self, system, path, mounts, **patches):
        parse = "_parse_mount_output" if system == "Darwin" else "_parse_mountinfo"
        with mock.patch.object(storage.platform, "system", return_value=system), \
                mock.patch.object(storage.subprocess, "run", return_value=mock.Mock(stdout="")), \
                mock.patch.object(storage.Path, "read_text", return_value=""), \
                mock.patch.object(storage, parse, return_value=mounts), \
                mock.patch.object(storage, "_macos_is_external", patches.get("mac", mock.Mock(return_value=False))), \
                mock.patch.object(storage, "_linux_is_external", patches.get("linux", mock.Mock(return_value=False))):
            return storage._classify(path)

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
        with mock.patch.object(storage, "_classify", side_effect=OSError("no diskutil")):
            self.assertIsNone(storage.storage_problem("/x"))

    def test_the_message_says_what_is_wrong_and_what_to_do(self):
        with mock.patch.object(storage, "_classify", return_value=storage.NETWORK):
            self.assertIn("network drive", storage.storage_problem("/x"))
        with mock.patch.object(storage, "_classify", return_value=storage.EXTERNAL):
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


if __name__ == "__main__":
    unittest.main()
