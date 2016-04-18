import s2sphere
import unittest


class CoveringTest(unittest.TestCase):
    def test_covering(self):
        r = s2sphere.RegionCoverer()
        p1 = s2sphere.LatLon.from_degrees(33, -122)
        p2 = s2sphere.LatLon.from_degrees(33, -122.1)
        cell_ids = r.get_covering(s2sphere.LatLonRect.from_point_pair(p1, p2))
        ids = sorted([long(c.id()) for c in cell_ids])
        target = [9291046973249421312,
                  9291046994724257792,
                  9291047097803472896,
                  9291051444310376448,
                  9291051856627236864,
                  9291051994066190336,
                  9291052195929653248,
                  9291052268944097280]
        print(ids)
        print(target)
        assert ids == target


if __name__ == '__main__':
    unittest.main()
