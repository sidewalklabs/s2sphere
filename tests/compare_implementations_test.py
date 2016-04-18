from __future__ import print_function, unicode_literals, division

import unittest

import s2  # SWIG wrapped original C++ implementation
import s2sphere.sphere as sphere


class TestCellId(unittest.TestCase):

    def test_cellid(self):
        lat, lon = (33, -122)
        pyc = sphere.CellId.from_lat_lon(sphere.LatLon.from_degrees(lat, lon))
        cc = s2.S2CellId.FromLatLng(s2.S2LatLng.FromDegrees(lat, lon))
        print(pyc.id(), cc.id())
        self.assertEqual(pyc.id(), cc.id())

    def test_cellid_level(self, level=12):
        lat, lon = (33, -122)
        pyc = (sphere.CellId
               .from_lat_lon(sphere.LatLon.from_degrees(lat, lon))
               .parent(level))
        cc = (s2.S2CellId
              .FromLatLng(s2.S2LatLng.FromDegrees(lat, lon))
              .parent(level))
        print(pyc.id(), cc.id())
        self.assertEqual(pyc.id(), cc.id())


if __name__ == '__main__':
    unittest.main()
