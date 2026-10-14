"""Offline verification of Colab Drive snapshot and restore guarantees."""
import json
import tempfile
import unittest
from pathlib import Path
from colab_backup_runner import BackupStore, sha256


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.drive = root / 'Drive'
        self.results = root / 'local' / 'results'
        self.data = root / 'local' / 'data'
        self.results.mkdir(parents=True)
        self.data.mkdir(parents=True)
        self.store = BackupStore(self.drive, {'results': self.results, 'data': self.data})

    def test_snapshot_roundtrip(self):
        (self.results / 'small.csv').write_text('Q,F\n.9,.1\n')
        (self.data / 'image.bin').write_bytes(b'source' * 200)
        self.store.snapshot()
        (self.results / 'small.csv').unlink()
        (self.data / 'image.bin').unlink()
        self.assertTrue(self.store.restore())
        self.assertEqual((self.results / 'small.csv').read_text(), 'Q,F\n.9,.1\n')
        self.assertEqual((self.data / 'image.bin').read_bytes(), b'source' * 200)

    def test_restore_while_data_cache_exists(self):
        (self.results / 'result.json').write_text('{"ok":true}')
        self.store.snapshot()
        (self.results / 'result.json').unlink()
        (self.data / 'cache').write_bytes(b'existing')
        self.assertTrue(self.store.restore())
        self.assertTrue((self.results / 'result.json').exists())
        self.assertEqual((self.data / 'cache').read_bytes(), b'existing')

    def test_content_addressed_copy(self):
        target = self.results / 'mutable.json'
        target.write_text('one')
        self.store.snapshot()
        n1 = len(list((self.drive / 'blobs').rglob('*')))
        self.store.snapshot()
        n2 = len(list((self.drive / 'blobs').rglob('*')))
        self.assertEqual(n1, n2)
        target.write_text('two')
        self.store.snapshot()
        n3 = len(list((self.drive / 'blobs').rglob('*')))
        self.assertGreater(n3, n2)

    def test_incomplete_newest_snapshot_falls_back(self):
        f = self.results / 'test.txt'
        f.write_text('old')
        self.store.snapshot()
        f.write_text('new')
        self.store.snapshot()
        self.store.blob_path(sha256(f)).unlink()  # simulate failed/vanished Drive file
        f.unlink()
        self.assertTrue(self.store.restore())
        self.assertEqual(f.read_text(), 'old')

    def test_newer_local_run_not_overwritten(self):
        f = self.results / 'test.txt'
        f.write_text('old')
        self.store.snapshot()
        f.write_text('new local result')
        self.assertFalse(self.store.restore())
        self.assertEqual(f.read_text(), 'new local result')


if __name__ == '__main__':
    unittest.main()
