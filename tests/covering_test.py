import s2sphere
import unittest


class CoveringTest(unittest.TestCase):
    def test_covering(self):
        r = s2sphere.RegionCoverer()
        p1 = s2sphere.LatLon.from_degrees(33, -122)
        p2 = s2sphere.LatLon.from_degrees(33.1, -122.1)
        cell_ids = r.get_covering(s2sphere.LatLonRect.from_point_pair(p1, p2))
        ids = sorted([c.id() for c in cell_ids])
        target = [9291041754864156672,
                  9291043953887412224,
                  9291044503643226112,
                  9291045878032760832,
                  9291047252422295552,
                  9291047802178109440,
                  9291051650468806656,
                  9291052200224620544]
        self.assertEquals(ids, target)


if __name__ == '__main__':
    unittest.main()
