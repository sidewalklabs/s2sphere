import s2sphere
import unittest


class CellType(unittest.TestCase):
    def test_constructor(self):
        c = s2sphere.Cell(s2sphere.CellId(0x12345)).id()
        print(c)


if __name__ == '__main__':
    unittest.main()
