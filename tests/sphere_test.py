from __future__ import print_function, unicode_literals, division

import unittest
import math
import random

import numpy as np

from collections import defaultdict

try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip  # Python 3

try:
    xrange
except NameError:
    xrange = range

from pstats import Stats
import cProfile

import s2sphere
from s2sphere import Angle, CellId, LatLon, Point, Cell
from s2sphere import LineInterval, SphereInterval, LatLonRect
from s2sphere import RegionCoverer, CellUnion, Cap


# Some tests are based on spot checking random cell ids. The following list
# of 'ITERATIONS' defines how many spot checks are done per test.
'''
INVERSE_ITERATIONS = 200000
TOKEN_ITERATIONS = 10000
COVERAGE_ITERATIONS = 1000000
NEIGHBORS_ITERATIONS = 1000
NORMALIZE_ITERATIONS = 2000
REGION_COVERER_ITERATIONS = 1000
RANDOM_CAPS_ITERATIONS = 1000
SIMPLE_COVERINGS_ITERATIONS = 1000
'''
INVERSE_ITERATIONS = 20
TOKEN_ITERATIONS = 10
COVERAGE_ITERATIONS = 10
NEIGHBORS_ITERATIONS = 10
NORMALIZE_ITERATIONS = 20
REGION_COVERER_ITERATIONS = 10
RANDOM_CAPS_ITERATIONS = 10
SIMPLE_COVERINGS_ITERATIONS = 10

PROFILE = False


class TestAngle(unittest.TestCase):

    def testDefaultConstructor(self):
        angle = Angle()
        self.assertEqual(angle.radians, 0)

    def testPiRadiansExactly180Degrees(self):
        self.assertEqual(Angle.from_radians(math.pi).radians, math.pi)
        self.assertEqual(Angle.from_radians(math.pi).degrees, 180.0)
        self.assertEqual(Angle.from_degrees(180).radians, math.pi)
        self.assertEqual(Angle.from_degrees(180).degrees, 180.0)

        self.assertEqual(Angle.from_radians((-math.pi / 2)).degrees, -90.0)
        self.assertEqual(Angle.from_degrees((-45)).radians, -math.pi / 4)


class TestLatLon(unittest.TestCase):

    def testBasics(self):
        ll_rad = LatLon.from_radians(math.pi / 4, math.pi / 2)
        self.assertEqual(ll_rad.lat().radians, math.pi / 4)
        self.assertEqual(ll_rad.lon().radians, math.pi / 2)
        self.assertTrue(ll_rad.is_valid())

        ll_deg = LatLon.from_degrees(45, 90)
        self.assertEqual(ll_rad, ll_deg)
        self.assertFalse(LatLon.from_degrees(-91, 0).is_valid())
        self.assertFalse(LatLon.from_degrees(0, 181).is_valid())

        bad = LatLon.from_degrees(120, 200)
        self.assertFalse(bad.is_valid())
        better = bad.normalized()
        self.assertTrue(better.is_valid())
        self.assertEqual(Angle.from_degrees(90), better.lat())
        self.assertEqual(Angle.from_degrees(-160).radians,
                         better.lon().radians)

        self.assertTrue(
            (LatLon.from_degrees(10, 20) + LatLon.from_degrees(20, 30))
            .approx_equals(LatLon.from_degrees(30, 50)))
        self.assertTrue(
            (LatLon.from_degrees(10, 20) - LatLon.from_degrees(20, 30))
            .approx_equals(LatLon.from_degrees(-10, -10)))
        # self.assertTrue((0.5 * LatLon.from_degrees(10, 20)).approx_equals(
        #            LatLon.from_degrees(5, 10)))

        invalid = LatLon.invalid()
        self.assertFalse(invalid.is_valid())

        default_ll = LatLon.default()
        self.assertTrue(default_ll.is_valid())
        self.assertEqual(0, default_ll.lat().radians)
        self.assertEqual(0, default_ll.lon().radians)

    def testConversion(self):
        self.assertEqual(LatLon.from_point(LatLon.from_degrees(
                        90.0, 65.0).to_point()).lat().degrees, 90.0)

        self.assertEqual(LatLon.from_point(LatLon.from_radians(
                    -math.pi / 2, 1).to_point()).lat().radians, -math.pi / 2)

        self.assertEqual(abs(LatLon.from_point(LatLon.from_degrees(
                        12.2, 180.0).to_point()).lon().degrees), 180.0)

        self.assertEqual(abs(LatLon.from_point(LatLon.from_radians(
                        0.1, -math.pi).to_point()).lon().radians), math.pi)

    def testDistance(self):
        self.assertEqual(
            0.0,
            LatLon.from_degrees(90, 0).get_distance(
                LatLon.from_degrees(90, 0)
            ).radians)
        self.assertAlmostEqual(
            77.0,
            LatLon.from_degrees(-37, 25).get_distance(
                LatLon.from_degrees(-66, -155)
            ).degrees, delta=1e-13)
        self.assertAlmostEqual(
            115.0,
            LatLon.from_degrees(0, 165).get_distance(
                LatLon.from_degrees(0, -80)
            ).degrees, delta=1e-13)
        self.assertAlmostEqual(
            180.0,
            LatLon.from_degrees(47, -127).get_distance(
                LatLon.from_degrees(-47, 53)
            ).degrees, delta=2e-6)


class TestCellId(unittest.TestCase):

    def setUp(self):
        random.seed(20)

        if PROFILE:
            self.profile = cProfile.Profile()
            self.profile.enable()

    def tearDown(self):
        if hasattr(self, 'pr'):
            p = Stats(self.profile)
            p.strip_dirs()
            p.sort_stats('cumtime')
            p.print_stats()

    @staticmethod
    def get_random_cell_id(*args):
        if len(args) == 0:
            level = random.randrange(CellId.MAX_LEVEL + 1)
        else:
            level = args[0]
        face = random.randrange(CellId.NUM_FACES)
        pos = (random.randrange(0xffffffffffffffff) &
               ((1 << (2 * CellId.MAX_LEVEL)) - 1))
        return CellId.from_face_pos_level(face, pos, level)

    def get_random_point(self):
        x = 2 * random.random() - 1
        y = 2 * random.random() - 1
        z = 2 * random.random() - 1
        return Point(x, y, z).normalize()

    @staticmethod
    def get_cell_id(lat, lon):
        return CellId.from_lat_lon(LatLon.from_degrees(lat, lon))

    def testDefaultConstructor(self):
        cell_id = CellId()
        self.assertEqual(cell_id.id(), 0)
        self.assertFalse(cell_id.is_valid())

    def testFaceDefinitions(self):
        self.assertEqual(TestCellId.get_cell_id(0, 0).face(), 0)
        self.assertEqual(TestCellId.get_cell_id(0, 90).face(), 1)
        self.assertEqual(TestCellId.get_cell_id(90, 0).face(), 2)
        self.assertEqual(TestCellId.get_cell_id(0, 180).face(), 3)
        self.assertEqual(TestCellId.get_cell_id(0, -90).face(), 4)
        self.assertEqual(TestCellId.get_cell_id(-90, 0).face(), 5)

    def testParentChildRelationships(self):
        cell_id = CellId.from_face_pos_level(3, 0x12345678,
                                             CellId.MAX_LEVEL - 4)

        self.assertTrue(cell_id.is_valid())
        self.assertEqual(cell_id.face(), 3)
        self.assertEqual(cell_id.pos(), 0x12345700)
        self.assertEqual(cell_id.level(), CellId.MAX_LEVEL - 4)
        self.assertFalse(cell_id.is_leaf())

        self.assertEqual(cell_id.child_begin(cell_id.level() + 2).pos(),
                         0x12345610)
        self.assertEqual(cell_id.child_begin().pos(), 0x12345640)
        self.assertEqual(cell_id.parent().pos(), 0x12345400)
        self.assertEqual(cell_id.parent(cell_id.level() - 2).pos(), 0x12345000)

        # Check ordering of children relative to parents.
        self.assertLess(cell_id.child_begin(), cell_id)
        self.assertGreater(cell_id.child_end(), cell_id)
        self.assertEqual(cell_id.child_begin().next().next().next().next(),
                         cell_id.child_end())
        self.assertEqual(cell_id.child_begin(CellId.MAX_LEVEL),
                         cell_id.range_min())
        self.assertEqual(cell_id.child_end(CellId.MAX_LEVEL),
                         cell_id.range_max().next())

        # Check that cells are represented by the position of their center
        # along the Hilbert curve.
        self.assertEqual(cell_id.range_min().id() + cell_id.range_max().id(),
                         2 * cell_id.id())

    def testWrapping(self):
        self.assertEqual(CellId.begin(0).prev_wrap(), CellId.end(0).prev())
        self.assertEqual(
            CellId.begin(CellId.MAX_LEVEL).prev_wrap(),
            CellId.from_face_pos_level(
                5,
                0xffffffffffffffff >> CellId.FACE_BITS,
                CellId.MAX_LEVEL
            ))

        self.assertEqual(
            CellId.begin(CellId.MAX_LEVEL).advance_wrap(-1),
            CellId.from_face_pos_level(
                5,
                0xffffffffffffffff >> CellId.FACE_BITS,
                CellId.MAX_LEVEL
            ))

        self.assertEqual(CellId.end(4).advance(-1).advance_wrap(1),
                         CellId.begin(4))

        self.assertEqual(CellId.end(
                    CellId.MAX_LEVEL).advance(-1).advance_wrap(1),
                CellId.from_face_pos_level(0, 0, CellId.MAX_LEVEL))

        self.assertEqual(CellId.end(4).prev().next_wrap(), CellId.begin(4))

        self.assertEqual(CellId.end(CellId.MAX_LEVEL).prev().next_wrap(),
                         CellId.from_face_pos_level(0, 0, CellId.MAX_LEVEL))

    def testAdvance(self):

        cell_id = CellId.from_face_pos_level(3, 0x12345678,
                                             CellId.MAX_LEVEL - 4)

        self.assertEqual(CellId.begin(0).advance(7), CellId.end(0))
        self.assertEqual(CellId.begin(0).advance(12), CellId.end(0))
        self.assertEqual(CellId.end(0).advance(-7), CellId.begin(0))
        self.assertEqual(CellId.end(0).advance(-12000000), CellId.begin(0))

        num_level_5_cells = 6 << (2 * 5)
        self.assertEqual(CellId.begin(5).advance(500),
                         CellId.end(5).advance(500 - num_level_5_cells))
        self.assertEqual(cell_id.child_begin(CellId.MAX_LEVEL).advance(256),
                         cell_id.next().child_begin(CellId.MAX_LEVEL))
        self.assertEqual(CellId.from_face_pos_level(1, 0, CellId.MAX_LEVEL)
                         .advance(4 << (2 * CellId.MAX_LEVEL)),
                         CellId.from_face_pos_level(5, 0, CellId.MAX_LEVEL))

        # Check basic properties of advance_wrap().
        self.assertEqual(CellId.begin(0).advance_wrap(7),
                         CellId.from_face_pos_level(1, 0, 0))
        self.assertEqual(CellId.begin(0).advance_wrap(12), CellId.begin(0))

        self.assertEqual(CellId.from_face_pos_level(5, 0, 0).advance_wrap(-7),
                         CellId.from_face_pos_level(4, 0, 0))
        self.assertEqual(CellId.begin(0).advance_wrap(-12000000),
                         CellId.begin(0))
        self.assertEqual(CellId.begin(5).advance_wrap(6644),
                         CellId.begin(5).advance_wrap(-11788))
        self.assertEqual(
            cell_id.child_begin(CellId.MAX_LEVEL).advance_wrap(256),
            cell_id.next().child_begin(CellId.MAX_LEVEL))
        self.assertEqual(
            CellId.from_face_pos_level(5, 0, CellId.MAX_LEVEL)
            .advance_wrap(2 << (2 * CellId.MAX_LEVEL)),
            CellId.from_face_pos_level(1, 0, CellId.MAX_LEVEL))

    def testInverse(self):
        for i in xrange(INVERSE_ITERATIONS):
            cell_id = TestCellId.get_random_cell_id(CellId.MAX_LEVEL)
            self.assertTrue(cell_id.is_leaf())
            self.assertEqual(cell_id.level(), CellId.MAX_LEVEL)
            center = cell_id.to_lat_lon()
            self.assertEqual(CellId.from_lat_lon(center).id(), cell_id.id())

    def testTokens(self):
        for i in xrange(TOKEN_ITERATIONS):
            cell_id = TestCellId.get_random_cell_id()
            token = cell_id.to_token()
            self.assertLessEqual(len(token), 16)
            self.assertEqual(CellId.from_token(token), cell_id)

    def expand_cells(self, parent, cells, parent_map):
        cells.append(parent)
        if parent.level() == 3:  # max level for expand
            return

        face, i, j, orientation = parent.to_face_ij_orientation()
        self.assertEqual(face, parent.face())

        child = parent.child_begin()
        child_end = parent.child_end()
        pos = 0
        while child != child_end:
            self.assertEqual(parent.child(pos), child)
            self.assertEqual(child.level(), parent.level() + 1)
            self.assertFalse(child.is_leaf())
            cface, ci, cj, corientation = child.to_face_ij_orientation()
            self.assertEqual(cface, face)
            self.assertEqual(corientation,
                             orientation ^ s2sphere.POS_TO_ORIENTATION[pos])

            parent_map[child] = parent
            self.expand_cells(child, cells, parent_map)
            child = child.next()
            pos = pos + 1

    def testContainment(self):
        parent_map = {}
        cells = []
        for face in range(6):
            self.expand_cells(CellId.from_face_pos_level(face, 0, 0),
                              cells, parent_map)

        for i, cell_id_i in enumerate(cells):
            for j, cell_id_j in enumerate(cells):
                contained = True
                cell_id = cell_id_j
                while cell_id != cell_id_i:
                    next_cell_id = parent_map.get(cell_id)
                    if next_cell_id is None:
                        contained = False
                        break
                    cell_id = next_cell_id

                self.assertEqual(cells[i].contains(cells[j]), contained)
                self.assertEqual(cells[j] >= cells[i].range_min() and
                                 cells[j] <= cells[i].range_max(), contained)
                self.assertEqual(cells[i].intersects(cells[j]),
                                 cells[i].contains(cells[j]) or
                                 cells[j].contains(cells[i]))

    def testContinuity(self):
        # Make sure that sequentially increasing cell ids form a continuous
        # path over the surface of the sphere, i.e. there are no
        # discontinuous jumps from one region to another.

        max_walk_level = 8
        cell_size = 1 / (1 << max_walk_level)
        max_dist = CellId.max_edge().get_value(max_walk_level)

        end = CellId.end(max_walk_level)
        cell_id = CellId.begin(max_walk_level)
        while cell_id != end:
            self.assertLessEqual(
                cell_id.to_point_raw().angle(
                    cell_id.next_wrap().to_point_raw()), max_dist)

            self.assertEqual(cell_id.advance_wrap(1), cell_id.next_wrap())
            self.assertEqual(cell_id.next_wrap().advance_wrap(-1), cell_id)

            # Check that the ToPointRaw() returns the center of each cell
            # in (s,t) coordinates.
            face, u, v = s2sphere.xyz_to_face_uv(cell_id.to_point_raw())
            self.assertAlmostEqual(
                    CellId.uv_to_st(u) % (0.5 * cell_size), 0, delta=1-15)
            self.assertAlmostEqual(
                    CellId.uv_to_st(v) % (0.5 * cell_size), 0, delta=1-15)

            cell_id = cell_id.next()

    def testCoverage(self):
        max_dist = 0.5 * CellId.max_diag().get_value(CellId.MAX_LEVEL)
        for i in xrange(COVERAGE_ITERATIONS):
            p = self.get_random_point()
            q = CellId.from_point(p).to_point_raw()
            self.assertLessEqual(p.angle(q), max_dist)

    def testNeighbors(self):
        # Check the edge neighbors of face 1.
        out_faces = (5, 3, 2, 0)
        face_nbrs = CellId.from_face_pos_level(1, 0, 0).get_edge_neighbors()
        for i, face_nbr in enumerate(face_nbrs):
            self.assertTrue(face_nbr.is_face())
            self.assertEqual(face_nbr.face(), out_faces[i])

        # Check the vertex neighbors of the center of face 2 at level 5.
        neighbors = CellId.from_point(Point(0, 0, 1)).get_vertex_neighbors(5)
        neighbors.sort()
        for i, neighbor in enumerate(neighbors):
            self.assertEqual(
                neighbor,
                CellId.from_face_ij(
                    2,
                    (1 << 29) - (i < 2), (1 << 29) - (i == 0 or i == 3)
                ).parent(5)
            )

        # Check the vertex neighbors of the corner of faces 0, 4, and 5.
        cell_id = CellId.from_face_pos_level(0, 0, CellId.MAX_LEVEL)
        neighbors = cell_id.get_vertex_neighbors(0)
        neighbors.sort()
        self.assertEqual(len(neighbors), 3)
        self.assertEqual(neighbors[0], CellId.from_face_pos_level(0, 0, 0))
        self.assertEqual(neighbors[1], CellId.from_face_pos_level(4, 0, 0))
        self.assertEqual(neighbors[2], CellId.from_face_pos_level(5, 0, 0))

        for i in xrange(NEIGHBORS_ITERATIONS):
            cell_id = TestCellId.get_random_cell_id()
            if cell_id.is_leaf():
                cell_id = cell_id.parent()

            max_diff = min(6, CellId.MAX_LEVEL - cell_id.level() - 1)
            if max_diff == 0:
                level = cell_id.level()
            else:
                level = cell_id.level() + random.randrange(max_diff)

            self.check_all_neighbors(cell_id, level)

    def check_all_neighbors(self, cell_id, level):
        self.assertGreaterEqual(level, cell_id.level())
        self.assertLess(level, CellId.MAX_LEVEL)

        all, expected = set(), set()

        neighbors = cell_id.get_all_neighbors(level)
        all.update(neighbors)
        for c in cell_id.children(level + 1):
            all.add(c.parent())
            expected.update(c.get_vertex_neighbors(level))

        self.assertEqual(expected, all)


class LevelStats(object):
    def __init__(self):
        self.count = 0
        self.min_area = 100
        self.max_area = 0
        self.avg_area = 0
        self.min_width = 100
        self.max_width = 0
        self.avg_width = 0
        self.min_edge = 100
        self.max_edge = 0
        self.avg_edge = 0
        self.max_edge_aspect = 0
        self.min_diag = 100
        self.max_diag = 0
        self.avg_diag = 0
        self.max_diag_aspect = 0
        self.min_angle_span = 100
        self.max_angle_span = 0
        self.avg_angle_span = 0
        self.min_approx_ratio = 100
        self.max_approx_ratio = 0


class TestCell(unittest.TestCase):

    def setUp(self):
        random.seed(20)
        self.level_stats = [LevelStats()] * (CellId.MAX_LEVEL + 1)

        if PROFILE:
            self.profile = cProfile.Profile()
            self.profile.enable()

    def tearDown(self):
        if hasattr(self, 'pr'):
            p = Stats(self.profile)
            p.strip_dirs()
            p.sort_stats('cumtime')
            p.print_stats()

        del self.level_stats

    def testFaces(self):
        edge_counts, vertex_counts = defaultdict(int), defaultdict(int)

        for face in range(6):
            cell_id = CellId.from_face_pos_level(face, 0, 0)
            cell = Cell(cell_id)
            self.assertEqual(cell_id, cell.id())
            self.assertEqual(face, cell.face())
            self.assertEqual(0, cell.level())

            # Top-level faces have alternating orientations to get RHS
            # coordinates.
            self.assertEqual(face & s2sphere.SWAP_MASK, cell.orientation())
            self.assertFalse(cell.is_leaf())

            for k in range(4):
                edge_counts[cell.get_edge_raw(k)] += 1
                vertex_counts[cell.get_vertex_raw(k)] += 1
                self.assertEqual(
                    cell.get_vertex_raw(k).dot_prod(cell.get_edge_raw(k)), 0.0)
                self.assertEqual(
                    cell
                    .get_vertex_raw((k + 1) & 3)
                    .dot_prod(cell.get_edge_raw(k)),
                    0.0,
                )
                # this is assertEqual in C++ code
                self.assertAlmostEqual(
                    cell
                    .get_vertex_raw(k)
                    .cross_prod(cell.get_vertex_raw((k + 1) & 3))
                    .normalize()
                    .dot_prod(cell.get_edge(k)),
                    1.0,
                )

        # Check that edges have multiplicity 2 and
        # vertices have multiplicity 3.
        for count in edge_counts.values():
            self.assertEqual(count, 2)

        for count in vertex_counts.values():
            self.assertEqual(count, 3)

    def gather_stats(self, cell):
        s = self.level_stats[cell.level()]
        exact_area = cell.exact_area()
        approx_area = cell.approx_area()
        min_edge = 100
        max_edge = 0
        avg_edge = 0
        min_diag = 100
        max_diag = 0
        min_width = 100
        max_width = 0
        min_angle_span = 100
        max_angle_span = 0

        for i in range(4):
            edge = cell.get_vertex_raw(i).angle(
                    cell.get_vertex_raw((i + 1) & 3))
            min_edge = min(edge, min_edge)
            max_edge = max(edge, max_edge)
            # this could be wrong
            avg_edge += 0.25 * edge
            mid = cell.get_vertex_raw(i) + cell.get_vertex_raw((i + 1) & 3)
            width = math.pi / 2.0 - mid.angle(cell.get_edge_raw(i ^ 2))
            min_width = min(width, min_width)
            max_width = max(width, max_width)
            if i < 2:
                diag = cell.get_vertex_raw(i).angle(cell.get_vertex_raw(i ^ 2))
                min_diag = min(diag, min_diag)
                max_diag = max(diag, max_diag)
                angle_span = cell.get_edge_raw(i).angle(
                        -cell.get_edge_raw(i ^ 2))
                min_angle_span = min(angle_span, min_angle_span)
                max_angle_span = max(angle_span, max_angle_span)

        s.count += 1
        s.min_area = min(exact_area, s.min_area)
        s.max_area = max(exact_area, s.max_area)
        s.avg_area += exact_area
        s.min_width = min(min_width, s.min_width)
        s.max_width = max(max_width, s.max_width)
        s.avg_width += 0.5 * (min_width + max_width)
        s.min_edge = min(min_edge, s.min_edge)
        s.max_edge = max(max_edge, s.max_edge)
        s.avg_edge += avg_edge
        s.max_edge_aspect = max(max_edge / min_edge, s.max_edge_aspect)
        s.min_diag = min(min_diag, s.min_diag)
        s.max_diag = max(max_diag, s.max_diag)
        s.avg_diag += 0.5 * (min_diag + max_diag)
        s.max_diag_aspect = max(max_diag / min_diag, s.max_diag_aspect)
        s.min_angle_span = min(min_angle_span, s.min_angle_span)
        s.max_angle_span = max(max_angle_span, s.max_angle_span)
        s.avg_angle_span += 0.5 * (min_angle_span + max_angle_span)
        approx_ratio = approx_area / exact_area
        s.min_approx_ratio = min(approx_ratio, s.min_approx_ratio)
        s.max_approx_ratio = max(approx_ratio, s.max_approx_ratio)

    def check_subdivide(self, cell):
        self.gather_stats(cell)
        if cell.is_leaf():
            return

        children = tuple(cell.subdivide())

        exact_area = 0
        approx_area = 0
        average_area = 0

        for i, (child, child_id) \
                in enumerate(izip(children, cell.id().children())):

            exact_area += child.exact_area()
            approx_area += child.approx_area()
            average_area += child.average_area()
            self.assertEqual(child.id(), child_id)
            self.assertLess(
                    child.get_center().angle(child_id.to_point()), 1e-15)

            direct = Cell(child_id)
            self.assertEqual(direct.face(), child.face())
            self.assertEqual(direct.level(), child.level())
            self.assertEqual(direct.orientation(), child.orientation())
            self.assertEqual(direct.get_center_raw(), child.get_center_raw())

            for k in range(4):
                self.assertEqual(direct.get_vertex_raw(k),
                                 child.get_vertex_raw(k))
                self.assertEqual(direct.get_edge_raw(k),
                                 child.get_edge_raw(k))

            # Test contains() and may_intersect().
            self.assertTrue(cell.contains(child))
            self.assertTrue(cell.may_intersect(child))
            self.assertFalse(child.contains(cell))
            self.assertTrue(cell.contains(child.get_center_raw()))

            for j in range(4):
                self.assertTrue(cell.contains(child.get_vertex_raw(j)))
                if i != j:
                    # cannot get to pass test
                    self.assertFalse(
                           child.contains(children[j].get_center_raw()))
                    self.assertFalse(child.may_intersect(children[j]))

            # Test get_cap_bound and get_rect_bound
            parent_cap = cell.get_cap_bound()
            parent_rect = cell.get_rect_bound()
            if cell.contains(Point(0, 0, 1)) \
                    or cell.contains(Point(0, 0, -1)):
                self.assertTrue(parent_rect.lon().is_full())
            child_cap = children[i].get_cap_bound()
            child_rect = children[i].get_rect_bound()

            self.assertTrue(child_cap.contains(children[i].get_center()))
            self.assertTrue(child_rect.contains(children[i].get_center_raw()))
            self.assertTrue(parent_cap.contains(children[i].get_center()))
            self.assertTrue(parent_rect.contains(children[i].get_center_raw()))
            for j in range(4):
                self.assertTrue(
                    child_cap.contains(children[i].get_vertex(j)))
                self.assertTrue(
                    child_rect.contains(children[i].get_vertex(j)))
                self.assertTrue(
                    child_rect.contains(children[i].get_vertex_raw(j)))
                self.assertTrue(
                    parent_cap.contains(children[i].get_vertex(j)))
                self.assertTrue(
                    parent_rect.contains(children[i].get_vertex(j)))
                self.assertTrue(
                    parent_rect.contains(children[i].get_vertex_raw(j)))
                if j != i:
                    # The bounding caps and rectangles should be tight
                    # enough so that they exclude at least two vertices of
                    # each adjacent cell.
                    cap_count = 0
                    rect_count = 0
                    for k in range(4):
                        if child_cap.contains(children[j].get_vertex(k)):
                            ++cap_count
                        if child_rect.contains(children[j].get_vertex_raw(k)):
                            ++rect_count

                    self.assertLessEqual(cap_count, 2)
                    if child_rect.lat_lo().radians > -math.pi / 2.0 and \
                       child_rect.lat_hi().radians < math.pi / 2.0:
                        # Bounding rectangles may be too large at the poles
                        # because the pole itself has an arb fixed longitude.
                        self.assertLessEqual(rect_count, 2)

            force_subdivide = False
            center = s2sphere.get_norm(children[i].face())
            edge = center + s2sphere.get_u_axis(children[i].face())
            corner = edge + s2sphere.get_v_axis(children[i].face())
            for j in range(4):
                p = children[i].get_vertex_raw(j)
                if p == center or p == edge or p == corner:
                    force_subdivide = True
            if force_subdivide or cell.level() < 5 or \
               random.randrange(50) == 0:
                self.check_subdivide(children[i])

        self.assertLessEqual(
                math.fabs(math.log(exact_area / cell.exact_area())),
                math.fabs(math.log(1 + 1e-6)))
        self.assertLessEqual(
                math.fabs(math.log(approx_area / cell.approx_area())),
                math.fabs(math.log(1.03)))
        self.assertLessEqual(
                math.fabs(math.log(average_area / cell.average_area())),
                math.fabs(math.log(1 + 1e-15)))

    def testSubdivide(self):
        for face in range(6):
            self.check_subdivide(Cell.from_face_pos_level(face, 0, 0))


class TestLineInterval(unittest.TestCase):

    def check_interval_ops(self, x, y, expected):

        self.assertEqual(expected[0] == 'T', x.contains(y))
        self.assertEqual(expected[1] == 'T', x.interior_contains(y))
        self.assertEqual(expected[2] == 'T', x.intersects(y))
        self.assertEqual(expected[3] == 'T', x.interior_intersects(y))

    def testBasic(self):
        unit = LineInterval(0, 1)
        negunit = LineInterval(-1, 0)
        self.assertEqual(0, unit.lo())
        self.assertEqual(1, unit.hi())
        self.assertEqual(-1, negunit.bound(0))
        self.assertEqual(0, negunit.bound(1))

        # Keep immutable for now
        # ten = LineInterval(0, 0)
        # ten.set_hi(10)
        # self.assertEqual(10, ten.hi())

        half = LineInterval(0.5, 0.5)
        self.assertFalse(unit.is_empty())
        self.assertFalse(half.is_empty())
        empty = LineInterval.empty()
        self.assertTrue(empty.is_empty())

        default_empty = LineInterval()
        self.assertTrue(default_empty.is_empty())
        self.assertEqual(empty.lo(), default_empty.lo())
        self.assertEqual(empty.hi(), default_empty.hi())

        self.assertEqual(unit.get_center(), 0.5)
        self.assertEqual(half.get_center(), 0.5)
        self.assertEqual(negunit.get_length(), 1.0)
        self.assertLess(empty.get_length(), 0)

        # Contains(double), InteriorContains(double)
        self.assertTrue(unit.contains(0.5))
        self.assertTrue(unit.interior_contains(0.5))
        self.assertTrue(unit.contains(0))
        self.assertFalse(unit.interior_contains(0))
        self.assertTrue(unit.contains(1))
        self.assertFalse(unit.interior_contains(1))

        self.check_interval_ops(empty, empty, 'TTFF')
        self.check_interval_ops(empty, unit, 'FFFF')
        self.check_interval_ops(unit, half, 'TTTT')
        self.check_interval_ops(unit, unit, 'TFTT')
        self.check_interval_ops(unit, empty, 'TTFF')
        self.check_interval_ops(unit, negunit, 'FFTF')
        self.check_interval_ops(unit, LineInterval(0, 0.5), 'TFTT')
        self.check_interval_ops(half, LineInterval(0, 0.5), 'FFTF')

        # AddPont() should go here but trying to keep class immutable

        # from_point_pair
        self.assertEqual(LineInterval(4, 4),
                         LineInterval.from_point_pair(4, 4))
        self.assertEqual(LineInterval(-2, -1),
                         LineInterval.from_point_pair(-1, -2))
        self.assertEqual(LineInterval(-5, 3),
                         LineInterval.from_point_pair(-5, 3))

        # expanded
        self.assertEqual(empty, empty.expanded(0.45))
        self.assertEqual(LineInterval(-0.5, 1.5), unit.expanded(0.5))

        # union, intersection
        self.assertEqual(LineInterval(99, 100),
                         LineInterval(99, 100).union(empty))
        self.assertEqual(LineInterval(99, 100),
                         empty.union(LineInterval(99, 100)))
        self.assertTrue(
                LineInterval(5, 3).union(LineInterval(0, -2).is_empty()))
        self.assertTrue(
                LineInterval(0, -2).union(LineInterval(5, 3)).is_empty())
        self.assertEqual(unit, unit.union(unit))
        self.assertEqual(LineInterval(-1, 1), unit.union(negunit))
        self.assertEqual(LineInterval(-1, 1), negunit.union(unit))
        self.assertEqual(unit, half.union(unit))
        self.assertEqual(half, unit.intersection(half))
        self.assertTrue(negunit.intersection(half).is_empty())
        self.assertTrue(unit.intersection(empty).is_empty())
        self.assertTrue(empty.intersection(unit).is_empty())


class TestSphereInterval(unittest.TestCase):

    def setUp(self):
        self.empty = SphereInterval.empty()
        self.full = SphereInterval.full()

        self.zero = SphereInterval(0, 0)
        self.pi2 = SphereInterval(math.pi / 2.0, math.pi / 2.0)
        self.pi = SphereInterval(math.pi, math.pi)
        self.mipi = SphereInterval(-math.pi, -math.pi)
        self.mipi2 = SphereInterval(-math.pi / 2.0, -math.pi / 2.0)
        # Single quadrants:
        self.quad1 = SphereInterval(0, math.pi / 2.0)
        self.quad2 = SphereInterval(math.pi / 2.0, -math.pi)
        self.quad3 = SphereInterval(math.pi, -math.pi / 2.0)
        self.quad4 = SphereInterval(-math.pi / 2.0, 0)
        # Quadrant pairs:
        self.quad12 = SphereInterval(0, -math.pi)
        self.quad23 = SphereInterval(math.pi / 2.0, -math.pi / 2.0)
        self.quad34 = SphereInterval(-math.pi, 0)
        self.quad41 = SphereInterval(-math.pi / 2.0, math.pi / 2.0)
        # Quadrant triples:
        self.quad123 = SphereInterval(0, -math.pi / 2.0)
        self.quad234 = SphereInterval(math.pi / 2.0, 0)
        self.quad341 = SphereInterval(math.pi, math.pi / 2.0)
        self.quad412 = SphereInterval(-math.pi / 2.0, -math.pi)
        # Small intervals around the midpoints between quadrants, such that
        # the center of each interval is offset slightly CCW from the midpoint.
        self.mid12 = SphereInterval(math.pi / 2 - 0.01, math.pi / 2 + 0.02)
        self.mid23 = SphereInterval(math.pi - 0.01, -math.pi + 0.02)
        self.mid34 = SphereInterval(-math.pi / 2.0 - 0.01,
                                    -math.pi / 2.0 + 0.02)
        self.mid41 = SphereInterval(-0.01, 0.02)

    def testConstructorsAndAccessors(self):
        self.assertEqual(self.quad12.lo(), 0)
        self.assertEqual(self.quad12.hi(), math.pi)
        self.assertEqual(self.quad34.bound(0), math.pi)
        self.assertEqual(self.quad34.bound(1), 0)
        self.assertEqual(self.pi.lo(), math.pi)
        self.assertEqual(self.pi.hi(), math.pi)

        # Check that [-Pi, -Pi] is normalized to [Pi, Pi].
        self.assertEqual(self.mipi.lo(), math.pi)
        self.assertEqual(self.mipi.hi(), math.pi)
        self.assertEqual(self.quad23.lo(), math.pi / 2.0)
        self.assertEqual(self.quad23.hi(), -math.pi / 2.0)

        default_empty = SphereInterval()
        self.assertTrue(default_empty.is_valid())
        self.assertTrue(default_empty.is_empty())
        self.assertEqual(self.empty.lo(), default_empty.lo())
        self.assertEqual(self.empty.hi(), default_empty.hi())
        # Should check intervals can be modified here

    def testSimplePredicates(self):
        # is_valid(), is_empty(), is_full(), is_inverted()
        self.assertTrue(self.zero.is_valid() and
                        not self.zero.is_empty() and
                        not self.zero.is_full())
        self.assertTrue(self.empty.is_valid() and
                        self.empty.is_empty() and
                        not self.empty.is_full())
        self.assertTrue(self.empty.is_inverted())
        self.assertTrue(self.full.is_valid() and
                        not self.full.is_empty() and
                        self.full.is_full())
        self.assertTrue(not self.quad12.is_empty() and
                        not self.quad12.is_full() and
                        not self.quad12.is_inverted())
        self.assertTrue(not self.quad23.is_empty() and
                        not self.quad23.is_full() and
                        self.quad23.is_inverted())
        self.assertTrue(self.pi.is_valid() and
                        not self.pi.is_empty() and
                        not self.pi.is_inverted())
        self.assertTrue(self.mipi.is_valid() and
                        not self.mipi.is_empty() and
                        not self.mipi.is_inverted())

    def testGetCenter(self):
        self.assertEqual(self.quad12.get_center(), math.pi / 2.0)
        self.assertEqual(SphereInterval(3.1, 2.9).get_center(),
                         3.0 - math.pi)
        self.assertEqual(SphereInterval(-2.9, -3.1).get_center(),
                         math.pi - 3.0)
        self.assertEqual(SphereInterval(2.1, -2.1).get_center(), math.pi)
        self.assertEqual(self.pi.get_center(), math.pi)
        self.assertEqual(self.mipi.get_center(), math.pi)
        self.assertEqual(self.quad123.get_center(), 0.75 * math.pi)

    def testGetLength(self):
        self.assertEqual(self.quad12.get_length(), math.pi)
        self.assertEqual(self.pi.get_length(), 0)
        self.assertEqual(self.mipi.get_length(), 0)
        self.assertEqual(self.quad123.get_length(), 1.5 * math.pi)
        self.assertEqual(math.fabs(self.quad23.get_length()), math.pi)
        self.assertEqual(self.full.get_length(), 2 * math.pi)
        self.assertLess(self.empty.get_length(), 0)

    def testComplement(self):
        self.assertTrue(self.empty.complement().is_full())
        self.assertTrue(self.full.complement().is_empty())
        self.assertTrue(self.pi.complement().is_full())
        self.assertTrue(self.mipi.complement().is_full())
        self.assertTrue(self.zero.complement().is_full())
        self.assertTrue(self.quad12.complement().approx_equals(self.quad34))
        self.assertTrue(self.quad34.complement().approx_equals(self.quad12))
        self.assertTrue(self.quad123.complement().approx_equals(self.quad4))

    def testContains(self):
        self.assertTrue(not self.empty.contains(0) and
                        not self.empty.contains(math.pi) and
                        not self.empty.contains(-math.pi))
        self.assertTrue(not self.empty.interior_contains(math.pi) and
                        not self.empty.interior_contains(-math.pi))
        self.assertTrue(self.full.contains(0) and
                        self.full.contains(math.pi) and
                        self.full.contains(-math.pi))
        self.assertTrue(self.full.interior_contains(math.pi) and
                        self.full.interior_contains(-math.pi))
        self.assertTrue(self.quad12.contains(0) and
                        self.quad12.contains(math.pi) and
                        self.quad12.contains(-math.pi))
        self.assertTrue(self.quad12.interior_contains(math.pi / 2.0) and
                        not self.quad12.interior_contains(0))
        self.assertTrue(not self.quad12.interior_contains(math.pi) and
                        not self.quad12.interior_contains(-math.pi))
        self.assertTrue(self.quad23.contains(math.pi / 2.0) and
                        self.quad23.contains(-math.pi / 2.0))
        self.assertTrue(self.quad23.contains(math.pi) and
                        self.quad23.contains(-math.pi))
        self.assertTrue(not self.quad23.contains(0))
        self.assertTrue(not self.quad23.interior_contains(math.pi / 2.0) and
                        not self.quad23.interior_contains(-math.pi / 2.0))
        self.assertTrue(self.quad23.interior_contains(math.pi) and
                        self.quad23.interior_contains(-math.pi))
        self.assertTrue(not self.quad23.interior_contains(0))
        self.assertTrue(self.pi.contains(math.pi) and
                        self.pi.contains(-math.pi) and not self.pi.contains(0))
        self.assertTrue(not self.pi.interior_contains(math.pi) and
                        not self.pi.interior_contains(-math.pi))
        self.assertTrue(self.mipi.contains(math.pi) and
                        self.mipi.contains(-math.pi) and
                        not self.mipi.contains(0))
        self.assertTrue(not self.mipi.interior_contains(math.pi) and
                        not self.mipi.interior_contains(-math.pi))
        self.assertTrue(self.zero.contains(0) and
                        not self.zero.interior_contains(0))

    def check_interval_ops(self, x, y,
                           expected_relation,
                           expected_union,
                           expected_intersection):
        self.assertEqual(x.contains(y), expected_relation[0] == 'T')
        self.assertEqual(x.interior_contains(y), expected_relation[1] == 'T')
        self.assertEqual(x.intersects(y), expected_relation[2] == 'T')
        self.assertEqual(x.interior_intersects(y), expected_relation[3] == 'T')

        self.assertEqual(x.union(y).bounds(), expected_union.bounds())
        self.assertEqual(x.intersection(y).bounds(),
                         expected_intersection.bounds())

        self.assertEqual(x.contains(y), x.union(y) == x)
        self.assertEqual(x.intersects(y), not x.intersection(y).is_empty())

        # if y.lo() == y.hi():
        #     S1Interval r = x
        #     r.AddPoint(y.lo())
        #     self.assertEqual(r.bounds(), expected_union.bounds())

    def testIntervalOps(self):

        self.check_interval_ops(self.empty, self.empty,
                                "TTFF", self.empty, self.empty)
        self.check_interval_ops(self.empty, self.full,
                                "FFFF", self.full, self.empty)
        self.check_interval_ops(self.empty, self.zero,
                                "FFFF", self.zero, self.empty)
        self.check_interval_ops(self.empty, self.pi,
                                "FFFF", self.pi, self.empty)
        self.check_interval_ops(self.empty, self.mipi,
                                "FFFF", self.mipi, self.empty)
        self.check_interval_ops(self.full, self.empty,
                                "TTFF", self.full, self.empty)
        self.check_interval_ops(self.full, self.full,
                                "TTTT", self.full, self.full)
        self.check_interval_ops(self.full, self.zero,
                                "TTTT", self.full, self.zero)
        self.check_interval_ops(self.full, self.pi,
                                "TTTT", self.full, self.pi)
        self.check_interval_ops(self.full, self.mipi,
                                "TTTT", self.full, self.mipi)
        self.check_interval_ops(self.full, self.quad12,
                                "TTTT", self.full, self.quad12)
        self.check_interval_ops(self.full, self.quad23,
                                "TTTT", self.full, self.quad23)

        self.check_interval_ops(self.zero, self.empty,
                                "TTFF", self.zero, self.empty)
        self.check_interval_ops(self.zero, self.full,
                                "FFTF", self.full, self.zero)
        self.check_interval_ops(self.zero, self.zero,
                                "TFTF", self.zero, self.zero)
        self.check_interval_ops(self.zero, self.pi,
                                "FFFF", SphereInterval(0, math.pi), self.empty)
        self.check_interval_ops(self.zero, self.pi2,
                                "FFFF", self.quad1, self.empty)
        self.check_interval_ops(self.zero, self.mipi,
                                "FFFF", self.quad12, self.empty)
        self.check_interval_ops(self.zero, self.mipi2,
                                "FFFF", self.quad4, self.empty)
        self.check_interval_ops(self.zero, self.quad12,
                                "FFTF", self.quad12, self.zero)
        self.check_interval_ops(self.zero, self.quad23,
                                "FFFF", self.quad123, self.empty)

        self.check_interval_ops(self.pi2, self.empty,
                                "TTFF", self.pi2, self.empty)
        self.check_interval_ops(self.pi2, self.full,
                                "FFTF", self.full, self.pi2)
        self.check_interval_ops(self.pi2, self.zero,
                                "FFFF", self.quad1, self.empty)
        self.check_interval_ops(self.pi2, self.pi,
                                "FFFF", SphereInterval(math.pi / 2.0, math.pi),
                                self.empty)
        self.check_interval_ops(self.pi2, self.pi2,
                                "TFTF", self.pi2, self.pi2)
        self.check_interval_ops(self.pi2, self.mipi,
                                "FFFF", self.quad2, self.empty)
        self.check_interval_ops(self.pi2, self.mipi2,
                                "FFFF", self.quad23, self.empty)
        self.check_interval_ops(self.pi2, self.quad12,
                                "FFTF", self.quad12, self.pi2)
        self.check_interval_ops(self.pi2, self.quad23,
                                "FFTF", self.quad23, self.pi2)

        self.check_interval_ops(self.pi, self.empty,
                                "TTFF", self.pi, self.empty)
        self.check_interval_ops(self.pi, self.full,
                                "FFTF", self.full, self.pi)
        self.check_interval_ops(self.pi, self.zero,
                                "FFFF", SphereInterval(math.pi, 0), self.empty)
        self.check_interval_ops(self.pi, self.pi,
                                "TFTF", self.pi, self.pi)
        self.check_interval_ops(self.pi, self.pi2,
                                "FFFF", SphereInterval(math.pi / 2.0, math.pi),
                                self.empty)
        self.check_interval_ops(self.pi, self.mipi,
                                "TFTF", self.pi, self.pi)
        self.check_interval_ops(self.pi, self.mipi2,
                                "FFFF", self.quad3, self.empty)
        self.check_interval_ops(self.pi, self.quad12,
                                "FFTF", SphereInterval(0, math.pi), self.pi)
        self.check_interval_ops(self.pi, self.quad23,
                                "FFTF", self.quad23, self.pi)

        self.check_interval_ops(self.mipi, self.empty,
                                "TTFF", self.mipi, self.empty)
        self.check_interval_ops(self.mipi, self.full,
                                "FFTF", self.full, self.mipi)
        self.check_interval_ops(self.mipi, self.zero,
                                "FFFF", self.quad34, self.empty)
        self.check_interval_ops(self.mipi, self.pi,
                                "TFTF", self.mipi, self.mipi)
        self.check_interval_ops(self.mipi, self.pi2,
                                "FFFF", self.quad2, self.empty)
        self.check_interval_ops(self.mipi, self.mipi,
                                "TFTF", self.mipi, self.mipi)
        self.check_interval_ops(self.mipi, self.mipi2,
                                "FFFF", SphereInterval(-math.pi,
                                                       -math.pi / 2.0),
                                self.empty)
        self.check_interval_ops(self.mipi, self.quad12,
                                "FFTF", self.quad12, self.mipi)
        self.check_interval_ops(self.mipi, self.quad23,
                                "FFTF", self.quad23, self.mipi)

        self.check_interval_ops(self.quad12, self.empty,
                                "TTFF", self.quad12, self.empty)
        self.check_interval_ops(self.quad12, self.full,
                                "FFTT", self.full, self.quad12)
        self.check_interval_ops(self.quad12, self.zero,
                                "TFTF", self.quad12, self.zero)
        self.check_interval_ops(self.quad12, self.pi,
                                "TFTF", self.quad12, self.pi)
        self.check_interval_ops(self.quad12, self.mipi,
                                "TFTF", self.quad12, self.mipi)
        self.check_interval_ops(self.quad12, self.quad12,
                                "TFTT", self.quad12, self.quad12)
        self.check_interval_ops(self.quad12, self.quad23,
                                "FFTT", self.quad123, self.quad2)
        self.check_interval_ops(self.quad12, self.quad34,
                                "FFTF", self.full, self.quad12)

        self.check_interval_ops(self.quad23, self.empty,
                                "TTFF", self.quad23, self.empty)
        self.check_interval_ops(self.quad23, self.full,
                                "FFTT", self.full, self.quad23)
        self.check_interval_ops(self.quad23, self.zero,
                                "FFFF", self.quad234, self.empty)
        self.check_interval_ops(self.quad23, self.pi,
                                "TTTT", self.quad23, self.pi)
        self.check_interval_ops(self.quad23, self.mipi,
                                "TTTT", self.quad23, self.mipi)
        self.check_interval_ops(self.quad23, self.quad12,
                                "FFTT", self.quad123, self.quad2)
        self.check_interval_ops(self.quad23, self.quad23,
                                "TFTT", self.quad23, self.quad23)
        self.check_interval_ops(self.quad23, self.quad34,
                                "FFTT", self.quad234,
                                SphereInterval(-math.pi, -math.pi / 2.0))

        self.check_interval_ops(self.quad1, self.quad23,
                                "FFTF", self.quad123,
                                SphereInterval(math.pi / 2.0, math.pi / 2.0))
        self.check_interval_ops(self.quad2, self.quad3,
                                "FFTF", self.quad23, self.mipi)
        self.check_interval_ops(self.quad3, self.quad2,
                                "FFTF", self.quad23, self.pi)
        self.check_interval_ops(self.quad2, self.pi,
                                "TFTF", self.quad2, self.pi)
        self.check_interval_ops(self.quad2, self.mipi,
                                "TFTF", self.quad2, self.mipi)
        self.check_interval_ops(self.quad3, self.pi,
                                "TFTF", self.quad3, self.pi)
        self.check_interval_ops(self.quad3, self.mipi,
                                "TFTF", self.quad3, self.mipi)

        self.check_interval_ops(self.quad12, self.mid12,
                                "TTTT", self.quad12, self.mid12)
        self.check_interval_ops(self.mid12, self.quad12,
                                "FFTT", self.quad12, self.mid12)

        quad12eps = SphereInterval(self.quad12.lo(), self.mid23.hi())
        quad2hi = SphereInterval(self.mid23.lo(), self.quad12.hi())
        self.check_interval_ops(self.quad12, self.mid23,
                                "FFTT", quad12eps, quad2hi)
        self.check_interval_ops(self.mid23, self.quad12,
                                "FFTT", quad12eps, quad2hi)

        quad412eps = SphereInterval(self.mid34.lo(), self.quad12.hi())
        self.check_interval_ops(self.quad12, self.mid34,
                                "FFFF", quad412eps, self.empty)
        self.check_interval_ops(self.mid34, self.quad12,
                                "FFFF", quad412eps, self.empty)

        quadeps12 = SphereInterval(self.mid41.lo(), self.quad12.hi())
        quad1lo = SphereInterval(self.quad12.lo(), self.mid41.hi())
        self.check_interval_ops(self.quad12, self.mid41,
                                "FFTT", quadeps12, quad1lo)
        self.check_interval_ops(self.mid41, self.quad12,
                                "FFTT", quadeps12, quad1lo)

        quad2lo = SphereInterval(self.quad23.lo(), self.mid12.hi())
        quad3hi = SphereInterval(self.mid34.lo(), self.quad23.hi())
        quadeps23 = SphereInterval(self.mid12.lo(), self.quad23.hi())
        quad23eps = SphereInterval(self.quad23.lo(), self.mid34.hi())
        quadeps123 = SphereInterval(self.mid41.lo(), self.quad23.hi())
        self.check_interval_ops(self.quad23, self.mid12,
                                "FFTT", quadeps23, quad2lo)
        self.check_interval_ops(self.mid12, self.quad23,
                                "FFTT", quadeps23, quad2lo)
        self.check_interval_ops(self.quad23, self.mid23,
                                "TTTT", self.quad23, self.mid23)
        self.check_interval_ops(self.mid23, self.quad23,
                                "FFTT", self.quad23, self.mid23)
        self.check_interval_ops(self.quad23, self.mid34,
                                "FFTT", quad23eps, quad3hi)
        self.check_interval_ops(self.mid34, self.quad23,
                                "FFTT", quad23eps, quad3hi)
        self.check_interval_ops(self.quad23, self.mid41,
                                "FFFF", quadeps123, self.empty)
        self.check_interval_ops(self.mid41, self.quad23,
                                "FFFF", quadeps123, self.empty)

    def testFromPointPair(self):
        self.assertEqual(SphereInterval.from_point_pair(-math.pi, math.pi),
                         self.pi)
        self.assertEqual(SphereInterval.from_point_pair(math.pi, -math.pi),
                         self.pi)
        self.assertEqual(SphereInterval.from_point_pair(
                self.mid34.hi(), self.mid34.lo()), self.mid34)
        self.assertEqual(SphereInterval.from_point_pair(
            self.mid23.lo(), self.mid23.hi()), self.mid23)

    def testExpanded(self):
        self.assertEqual(self.empty.expanded(1), self.empty)
        self.assertEqual(self.full.expanded(1), self.full)
        self.assertEqual(self.zero.expanded(1), SphereInterval(-1, 1))
        self.assertEqual(self.mipi.expanded(0.01),
                         SphereInterval(math.pi - 0.01, -math.pi + 0.01))
        self.assertEqual(self.pi.expanded(27), self.full)
        self.assertEqual(self.pi.expanded(math.pi / 2.0), self.quad23)
        self.assertEqual(self.pi2.expanded(math.pi / 2.0), self.quad12)
        self.assertEqual(self.mipi2.expanded(math.pi / 2.0), self.quad34)

    def testApproxEquals(self):

        self.assertTrue(self.empty.approx_equals(self.empty))
        self.assertTrue(self.zero.approx_equals(self.empty) and
                        self.empty.approx_equals(self.zero))
        self.assertTrue(self.pi.approx_equals(self.empty) and
                        self.empty.approx_equals(self.pi))
        self.assertTrue(self.mipi.approx_equals(self.empty) and
                        self.empty.approx_equals(self.mipi))
        self.assertTrue(self.pi.approx_equals(self.mipi) and
                        self.mipi.approx_equals(self.pi))
        self.assertTrue(self.pi.union(self.mipi).approx_equals(self.pi))
        self.assertTrue(self.mipi.union(self.pi).approx_equals(self.pi))
        self.assertTrue(self.pi.union(
            self.mid12).union(self.zero).approx_equals(self.quad12))
        self.assertTrue(self.quad2.intersection(
            self.quad3).approx_equals(self.pi))
        self.assertTrue(self.quad3.intersection(
            self.quad2).approx_equals(self.pi))

    def testGetDirectedHausdorffDistance(self):
        self.assertEqual(
            0.0,
            self.empty.get_directed_hausdorff_distance(self.empty))
        self.assertEqual(
            0.0,
            self.empty.get_directed_hausdorff_distance(self.mid12))
        self.assertEqual(
            math.pi,
            self.mid12.get_directed_hausdorff_distance(self.empty))

        self.assertEqual(
            0.0,
            self.quad12.get_directed_hausdorff_distance(self.quad123))

        interval = SphereInterval(3.0, -3.0)
        self.assertEqual(
            3.0,
            SphereInterval(-0.1, 0.2)
            .get_directed_hausdorff_distance(interval))
        self.assertEqual(
            3.0 - 0.1,
            SphereInterval(0.1, 0.2)
            .get_directed_hausdorff_distance(interval))
        self.assertEqual(
            3.0 - 0.1,
            SphereInterval(-0.2, -0.1)
            .get_directed_hausdorff_distance(interval))


class TestCap(unittest.TestCase):

    def setUp(self):
        # self.eps = 1e-15
        self.eps = 1e-14

    def get_lat_lon_point(self, lat_degrees, lon_degrees):
        return LatLon.from_degrees(lat_degrees, lon_degrees).to_point()

    def testBasic(self):
        empty = Cap.empty()
        full = Cap.full()
        self.assertTrue(empty.is_valid())
        self.assertTrue(empty.is_empty())
        self.assertTrue(empty.complement().is_full())
        self.assertTrue(full.is_valid())
        self.assertTrue(full.is_full())
        self.assertTrue(full.complement().is_empty())
        self.assertEqual(2, full.height())
        self.assertEqual(180.0, full.angle().degrees)

        default_empty = Cap()
        self.assertTrue(default_empty.is_valid())
        self.assertTrue(default_empty.is_empty())
        self.assertEqual(empty.axis(), default_empty.axis())
        self.assertEqual(empty.height(), default_empty.height())

        # Containment and intersection of empty and full caps.
        self.assertTrue(empty.contains(empty))
        self.assertTrue(full.contains(empty))
        self.assertTrue(full.contains(full))
        self.assertFalse(empty.interior_intersects(empty))
        self.assertTrue(full.interior_intersects(full))
        self.assertFalse(full.interior_intersects(empty))

        # Singleton cap containing the x-axis.
        xaxis = Cap.from_axis_height(Point(1, 0, 0), 0)
        self.assertTrue(xaxis.contains(Point(1, 0, 0)))
        self.assertFalse(xaxis.contains(Point(1, 1e-20, 0)))
        self.assertEqual(0, xaxis.angle().radians)

        # Singleton cap containing the y-axis.
        yaxis = Cap.from_axis_angle(Point(0, 1, 0), Angle.from_radians(0))
        self.assertFalse(yaxis.contains(xaxis.axis()))
        self.assertEqual(0, xaxis.height())

        # Check that the complement of a singleton cap is the full cap.
        xcomp = xaxis.complement()
        self.assertTrue(xcomp.is_valid())
        self.assertTrue(xcomp.is_full())
        self.assertTrue(xcomp.contains(xaxis.axis()))

        # Check that the complement of the complement is *not* the original.
        self.assertTrue(xcomp.complement().is_valid())
        self.assertTrue(xcomp.complement().is_empty())
        self.assertFalse(xcomp.complement().contains(xaxis.axis()))

        # Check that very small caps can be represented accurately.
        # Here "kTinyRad" is small enough that unit vectors perturbed by this
        # amount along a tangent do not need to be renormalized.
        kTinyRad = 1e-10
        tiny = Cap.from_axis_angle(Point(1, 2, 3).normalize(),
                                   Angle.from_radians(kTinyRad))
        tangent = tiny.axis().cross_prod(Point(3, 2, 1)).normalize()
        self.assertTrue(tiny.contains(tiny.axis() + 0.99 * kTinyRad * tangent))
        self.assertFalse(tiny.contains(tiny.axis() + 1.01 * kTinyRad * tangent))

        # Basic tests on a hemispherical cap.
        hemi = Cap.from_axis_height(Point(1, 0, 1).normalize(), 1)
        self.assertEqual(-hemi.axis(), hemi.complement().axis())
        self.assertEqual(1, hemi.complement().height())
        self.assertTrue(hemi.contains(Point(1, 0, 0)))
        self.assertFalse(hemi.complement().contains(Point(1, 0, 0)))
        self.assertTrue(hemi.contains(Point(1, 0, -(1-self.eps)).normalize()))
        self.assertFalse(hemi.interior_contains(
            Point(1, 0, -(1+self.eps)).normalize()))

        # A concave cap.
        concave = Cap.from_axis_angle(self.get_lat_lon_point(80, 10),
                                             Angle.from_degrees(150))
        self.assertTrue(
            concave.contains(self.get_lat_lon_point(-70 * (1 - self.eps), 10)))
        self.assertFalse(
            concave.contains(self.get_lat_lon_point(-70 * (1 + self.eps), 10)))
        self.assertTrue(
            concave.contains(
                self.get_lat_lon_point(-50 * (1 - self.eps), -170)))
        self.assertFalse(
            concave.contains(
                self.get_lat_lon_point(-50 * (1 + self.eps), -170)))

        # Cap containment tests.
        self.assertFalse(empty.contains(xaxis))
        self.assertFalse(empty.interior_intersects(xaxis))
        self.assertTrue(full.contains(xaxis))
        self.assertTrue(full.interior_intersects(xaxis))
        self.assertFalse(xaxis.contains(full))
        self.assertFalse(xaxis.interior_intersects(full))
        self.assertTrue(xaxis.contains(xaxis))
        self.assertFalse(xaxis.interior_intersects(xaxis))
        self.assertTrue(xaxis.contains(empty))
        self.assertFalse(xaxis.interior_intersects(empty))
        self.assertTrue(hemi.contains(tiny))
        self.assertTrue(hemi.contains(Cap.from_axis_angle(Point(1, 0, 0),
                                 Angle.from_radians(math.pi / 4.0 - self.eps))))
        self.assertFalse(hemi.contains(Cap.from_axis_angle(Point(1, 0, 0),
                                  Angle.from_radians(math.pi / 4.0 + self.eps))))
        self.assertTrue(concave.contains(hemi))
        self.assertTrue(concave.interior_intersects(hemi.complement()))
        self.assertFalse(concave.contains(
            Cap.from_axis_height(-concave.axis(), 0.1)))

    def testGetRectBound(self):

        # Empty and full caps.
        self.assertTrue(Cap.empty().get_rect_bound().is_empty())
        self.assertTrue(Cap.full().get_rect_bound().is_full())

        degree_eps = 1e-13
        # Maximum allowable error for latitudes and longitudes measured in
        # degrees.  (EXPECT_DOUBLE_EQ isn't sufficient.)

        # Cap that includes the south pole.
        rect = Cap.from_axis_angle(self.get_lat_lon_point(-45, 57),
                                         Angle.from_degrees(50)).get_rect_bound()
        self.assertAlmostEqual(rect.lat_lo().degrees, -90, delta=degree_eps)
        self.assertAlmostEqual(rect.lat_hi().degrees, 5, delta=degree_eps)
        self.assertTrue(rect.lon().is_full())

        # Cap that is tangent to the north pole.
        rect = Cap.from_axis_angle(Point(1, 0, 1).normalize(),
                    Angle.from_radians(math.pi / 4.0 + 1e-16)).get_rect_bound()
        self.assertAlmostEqual(rect.lat().lo(), 0, delta=self.eps)
        self.assertAlmostEqual(rect.lat().hi(), math.pi / 2.0, delta=self.eps)
        self.assertTrue(rect.lon().is_full())

        rect = Cap.from_axis_angle(Point(1, 0, 1).normalize(),
                                    Angle.from_degrees(45 + 5e-15)).get_rect_bound()
        self.assertAlmostEqual(rect.lat_lo().degrees, 0, delta=degree_eps)
        self.assertAlmostEqual(rect.lat_hi().degrees, 90, delta=degree_eps)
        self.assertTrue(rect.lon().is_full())

        # The eastern hemisphere.
        rect = Cap.from_axis_angle(Point(0, 1, 0),
                        Angle.from_radians(math.pi / 2.0 + 2e-16)).get_rect_bound()
        self.assertAlmostEqual(rect.lat_lo().degrees, -90, delta=degree_eps)
        self.assertAlmostEqual(rect.lat_hi().degrees, 90, delta=degree_eps)
        self.assertTrue(rect.lon().is_full())

        # A cap centered on the equator.
        rect = Cap.from_axis_angle(self.get_lat_lon_point(0, 50),
                                    Angle.from_degrees(20)).get_rect_bound()
        self.assertAlmostEqual(rect.lat_lo().degrees, -20, delta=degree_eps)
        self.assertAlmostEqual(rect.lat_hi().degrees, 20, delta=degree_eps)
        self.assertAlmostEqual(rect.lon_lo().degrees, 30, delta=degree_eps)
        self.assertAlmostEqual(rect.lon_hi().degrees, 70, delta=degree_eps)

        # A cap centered on the north pole.
        rect = Cap.from_axis_angle(self.get_lat_lon_point(90, 123),
                                    Angle.from_degrees(10)).get_rect_bound()
        self.assertAlmostEqual(rect.lat_lo().degrees, 80, delta=degree_eps)
        self.assertAlmostEqual(rect.lat_hi().degrees, 90, delta=degree_eps)
        self.assertTrue(rect.lon().is_full())

    def testCellMethods(self):
        face_radius = math.atan(math.sqrt(2))

        for face in range(6):
            # The cell consisting of the entire face.
            root_cell = Cell.from_face_pos_level(face, 0, 0)

            # A leaf cell at the midpoint of the v=1 edge.
            edge_cell = Cell.from_point(
                    s2sphere.face_uv_to_xyz(face, 0, 1 - self.eps))

            # A leaf cell at the u=1, v=1 corner.
            corner_cell = Cell.from_point(
                s2sphere.face_uv_to_xyz(face, 1 - self.eps, 1 - self.eps))

            # Quick check for full and empty caps.
            self.assertTrue(Cap.full().contains(root_cell))
            self.assertFalse(Cap.empty().may_intersect(root_cell))

            # Check intersections with the bounding caps of the leaf cells
            # that are adjacent to 'corner_cell' along the Hilbert curve.
            # Because this corner is at (u=1,v=1), the curve stays locally
            # within the same cube face.
            first = corner_cell.id().advance(-3)
            last = corner_cell.id().advance(4)
            id = first
            while id < last:
                cell = Cell(id)
                self.assertEqual(id == corner_cell.id(),
                          cell.get_cap_bound().contains(corner_cell))
                self.assertEqual(id.parent().contains(corner_cell.id()),
                          cell.get_cap_bound().may_intersect(corner_cell))

                id = id.next()

            anti_face = (face + 3) % 6  # Opposite face.
            for cap_face in range(6):
                # A cap that barely contains all of 'cap_face'.
                center = s2sphere.get_norm(cap_face)
                covering = Cap.from_axis_angle(center,
                    Angle.from_radians(face_radius + self.eps))
                self.assertEqual(cap_face == face, covering.contains(root_cell))
                self.assertEqual(cap_face != anti_face,
                    covering.may_intersect(root_cell))
                self.assertEqual(center.dot_prod(edge_cell.get_center()) > 0.1,
                    covering.contains(edge_cell))
                self.assertEqual(covering.may_intersect(edge_cell),
                    covering.contains(edge_cell))
                self.assertEqual(cap_face == face,
                    covering.contains(corner_cell))
                self.assertEqual(center.dot_prod(
                    corner_cell.get_center()) > 0,
                          covering.may_intersect(corner_cell))

                # A cap that barely intersects the edges of 'cap_face'.
                bulging = Cap.from_axis_angle(
                    center, Angle.from_radians(math.pi / 4.0 + self.eps))
                self.assertFalse(bulging.contains(root_cell))
                self.assertEqual(cap_face != anti_face,
                        bulging.may_intersect(root_cell))
                self.assertEqual(cap_face == face, bulging.contains(edge_cell))
                self.assertEqual(center.dot_prod(edge_cell.get_center()) > 0.1,
                          bulging.may_intersect(edge_cell))
                self.assertFalse(bulging.contains(corner_cell))
                self.assertFalse(bulging.may_intersect(corner_cell))

                # A singleton cap.
                singleton = Cap.from_axis_angle(center, Angle.from_radians(0))
                self.assertEqual(cap_face == face,
                        singleton.may_intersect(root_cell))
                self.assertFalse(singleton.may_intersect(edge_cell))
                self.assertFalse(singleton.may_intersect(corner_cell))

    def testExpanded(self):
        self.assertTrue(Cap.empty().expanded(Angle.from_radians(2)).is_empty())
        self.assertTrue(Cap.full().expanded(Angle.from_radians(2)).is_full())
        cap50 = Cap.from_axis_angle(Point(1, 0, 0), Angle.from_degrees(50))
        cap51 = Cap.from_axis_angle(Point(1, 0, 0), Angle.from_degrees(51))
        self.assertTrue(cap50.expanded(Angle.from_radians(0)).approx_equals(cap50))
        self.assertTrue(cap50.expanded(Angle.from_degrees(1)).approx_equals(cap51))
        self.assertFalse(cap50.expanded(Angle.from_degrees(129.99)).is_full())
        self.assertTrue(cap50.expanded(Angle.from_degrees(130.01)).is_full())


class TestLatLonRect(unittest.TestCase):

    def rect_from_degrees(self, lat_lo, lon_lo, lat_hi, lon_hi):
        return LatLonRect(LatLon.from_degrees(lat_lo, lon_lo),
                LatLon.from_degrees(lat_hi, lon_hi))

    def testEmptyAndFull(self):
        empty = LatLonRect.empty()
        full = LatLonRect.full()
        self.assertTrue(empty.is_valid())
        self.assertTrue(empty.is_empty())
        self.assertFalse(empty.is_point())
        self.assertTrue(full.is_valid())
        self.assertTrue(full.is_full())
        self.assertFalse(full.is_point())

        default_empty = LatLonRect()
        self.assertTrue(default_empty.is_valid())
        self.assertTrue(default_empty.is_empty())
        self.assertEqual(empty.lat().bounds(), default_empty.lat().bounds())
        self.assertEqual(empty.lon().bounds(), default_empty.lon().bounds())

    def testAccessors(self):
        d1 = self.rect_from_degrees(-90, 0, -45, 180)
        self.assertEqual(d1.lat_lo().degrees, -90)
        self.assertEqual(d1.lat_hi().degrees, -45)
        self.assertEqual(d1.lon_lo().degrees, 0)
        self.assertEqual(d1.lon_hi().degrees, 180)
        self.assertEqual(d1.lat(),
                         LineInterval(-math.pi / 2.0, -math.pi / 4.0))
        self.assertEqual(d1.lon(),
                         SphereInterval(0, math.pi))

    def testFromCenterSize(self):
        self.assertTrue(
            LatLonRect.from_center_size(
                LatLon.from_degrees(80, 170),
                LatLon.from_degrees(40, 60),
            ).approx_equals(self.rect_from_degrees(60, 140, 90, -160))
        )

        self.assertTrue(LatLonRect.from_center_size(
            LatLon.from_degrees(10, 40),
            LatLon.from_degrees(210, 400)).is_full()) \

        self.assertTrue(
            LatLonRect.from_center_size(
                LatLon.from_degrees(-90, 180),
                LatLon.from_degrees(20, 50),
            ).approx_equals(self.rect_from_degrees(-90, 155, -80, -155))
        )

    def testFromPoint(self):
        p = LatLon.from_degrees(23, 47)
        self.assertEqual(LatLonRect.from_point(p), LatLonRect(p, p))
        self.assertTrue(LatLonRect.from_point(p).is_point())

    def testFromPointPair(self):
        self.assertEqual(LatLonRect.from_point_pair(
            LatLon.from_degrees(-35, -140), LatLon.from_degrees(15, 155)),
            self.rect_from_degrees(-35, 155, 15, -140))
        self.assertEqual(LatLonRect.from_point_pair(
            LatLon.from_degrees(25, -70), LatLon.from_degrees(-90, 80)),
            self.rect_from_degrees(-90, -70, 25, 80))

    def testGetCenterSize(self):
        r1 = LatLonRect(LineInterval(0, math.pi / 2.0),
                        SphereInterval(-math.pi, 0))
        self.assertEqual(r1.get_center(),
                         LatLon.from_radians(math.pi / 4.0, -math.pi / 2.0))
        self.assertEqual(r1.get_size(),
                         LatLon.from_radians(math.pi / 2.0, math.pi))
        self.assertLess(
            LatLonRect.empty().get_size().lat().radians, 0)
        self.assertLess(
            LatLonRect.empty().get_size().lon().radians, 0)

    def testGetVertex(self):
        r1 = LatLonRect(LineInterval(0, math.pi / 2.0),
                        SphereInterval(-math.pi, 0))
        self.assertEqual(r1.get_vertex(0), LatLon.from_radians(0, math.pi))
        self.assertEqual(r1.get_vertex(1), LatLon.from_radians(0, 0))
        self.assertEqual(r1.get_vertex(2),
                         LatLon.from_radians(math.pi / 2.0, 0))
        self.assertEqual(r1.get_vertex(3),
                         LatLon.from_radians(math.pi / 2.0, math.pi))

        # Make sure the get_vertex() returns vertices in CCW order.
        for i in range(4):
            lat = math.pi / 4.0 * (i - 2)
            lon = math.pi / 2.0 * (i - 2) + 0.2
            r = LatLonRect(LineInterval(lat, lat + math.pi / 4.0),
                           SphereInterval(s2sphere.drem(lon, 2 * math.pi),
                           s2sphere.drem(lon + math.pi / 2.0, 2 * math.pi)))
            for k in range(4):
                self.assertTrue(
                    s2sphere.simple_ccw(r.get_vertex((k - 1) & 3).to_point(),
                                        r.get_vertex(k).to_point(),
                                        r.get_vertex((k + 1) & 3).to_point())
                )

    def testContains(self):
        eq_m180 = LatLon.from_radians(0, -math.pi)
        north_pole = LatLon.from_radians(math.pi / 2.0, 0)
        r1 = LatLonRect(eq_m180, north_pole)

        self.assertTrue(r1.contains(LatLon.from_degrees(30, -45)))
        self.assertTrue(r1.interior_contains(LatLon.from_degrees(30, -45)))
        self.assertFalse(r1.contains(LatLon.from_degrees(30, 45)))
        self.assertFalse(r1.interior_contains(LatLon.from_degrees(30, 45)))
        self.assertTrue(r1.contains(eq_m180))
        self.assertFalse(r1.interior_contains(eq_m180))
        self.assertTrue(r1.contains(north_pole))
        self.assertFalse(r1.interior_contains(north_pole))
        self.assertTrue(r1.contains(Point(0.5, -0.3, 0.1)))
        self.assertFalse(r1.contains(Point(0.5, 0.2, 0.1)))

    def check_interval_ops(self, x, y, expected_relation,
                           expected_union, expected_intersection):
        self.assertEqual(x.contains(y), expected_relation[0] == 'T')
        self.assertEqual(x.interior_contains(y), expected_relation[1] == 'T')
        self.assertEqual(x.intersects(y), expected_relation[2] == 'T')
        self.assertEqual(x.interior_intersects(y),
                         expected_relation[3] == 'T')

        self.assertEqual(x.contains(y), x.union(y) == x)
        self.assertEqual(x.intersects(y), not x.intersection(y).is_empty())

        self.assertEqual(x.union(y), expected_union)
        self.assertEqual(x.intersection(y), expected_intersection)

    def testIntervalOps(self):
        r1 = self.rect_from_degrees(0, -180, 90, 0)

        # Test operations where one rectangle consists of a single point.
        r1_mid = self.rect_from_degrees(45, -90, 45, -90)
        self.check_interval_ops(r1, r1_mid, "TTTT", r1, r1_mid)

        req_m180 = self.rect_from_degrees(0, -180, 0, -180)
        self.check_interval_ops(r1, req_m180, "TFTF", r1, req_m180)

        rnorth_pole = self.rect_from_degrees(90, 0, 90, 0)
        self.check_interval_ops(r1, rnorth_pole, "TFTF", r1, rnorth_pole)

        self.check_interval_ops(
            r1,
            self.rect_from_degrees(-10, -1, 1, 20), "FFTT",
            self.rect_from_degrees(-10, 180, 90, 20),
            self.rect_from_degrees(0, -1, 1, 0),
        )
        self.check_interval_ops(
            r1,
            self.rect_from_degrees(-10, -1, 0, 20), "FFTF",
            self.rect_from_degrees(-10, 180, 90, 20),
            self.rect_from_degrees(0, -1, 0, 0))
        self.check_interval_ops(
            r1,
            self.rect_from_degrees(-10, 0, 1, 20), "FFTF",
            self.rect_from_degrees(-10, 180, 90, 20),
            self.rect_from_degrees(0, 0, 1, 0),
        )

        self.check_interval_ops(
            self.rect_from_degrees(-15, -160, -15, -150),
            self.rect_from_degrees(20, 145, 25, 155), "FFFF",
            self.rect_from_degrees(-15, 145, 25, -150),
            LatLonRect.empty(),
        )
        self.check_interval_ops(
            self.rect_from_degrees(70, -10, 90, -140),
            self.rect_from_degrees(60, 175, 80, 5), "FFTT",
            self.rect_from_degrees(60, -180, 90, 180),
            self.rect_from_degrees(70, 175, 80, 5),
        )

        # Check that the intersection of two rectangles that overlap in
        # latitude but not longitude is valid, and vice versa.
        self.check_interval_ops(
            self.rect_from_degrees(12, 30, 60, 60),
            self.rect_from_degrees(0, 0, 30, 18), "FFFF",
            self.rect_from_degrees(0, 0, 60, 60),
            LatLonRect.empty(),
        )
        self.check_interval_ops(
            self.rect_from_degrees(0, 0, 18, 42),
            self.rect_from_degrees(30, 12, 42, 60), "FFFF",
            self.rect_from_degrees(0, 0, 42, 60),
            LatLonRect.empty(),
        )

    def testExpanded(self):
        self.assertTrue(self.rect_from_degrees(70, 150, 80, 170)
                        .expanded(LatLon.from_degrees(20, 30))
                        .approx_equals(
                            self.rect_from_degrees(50, 120, 90, -160)
                        ))
        self.assertTrue(LatLonRect.empty().expanded(
            LatLon.from_degrees(20, 30)).is_empty())
        self.assertTrue(LatLonRect.full().expanded(
            LatLon.from_degrees(20, 30)).is_full())

        self.assertTrue(self.rect_from_degrees(-90, 170, 10, 20)
                        .expanded(LatLon.from_degrees(30, 80))
                        .approx_equals(
                            self.rect_from_degrees(-90, -180, 40, 180)
                        ))

    def testConvolveWithCap(self):
        self.assertTrue(self.rect_from_degrees(0, 170, 0, -170)
                        .convolve_with_cap(Angle.from_degrees(15))
                        .approx_equals(
                            self.rect_from_degrees(-15, 155, 15, -155)
                        ))

        self.assertTrue(self.rect_from_degrees(60, 150, 80, 10)
                        .convolve_with_cap(Angle.from_degrees(15))
                        .approx_equals(
                            self.rect_from_degrees(45, -180, 90, 180)
                        ))

    def testGetCapBound(self):

        # Bounding cap at center is smaller:
        self.assertTrue(
                self.rect_from_degrees(-45, -45, 45, 45).get_cap_bound().
                    approx_equals(Cap.from_axis_height(Point(1, 0, 0), 0.5)))

        # Bounding cap at north pole is smaller:
        self.assertTrue(self.rect_from_degrees(88, -80, 89, 80).get_cap_bound().
                    approx_equals(Cap.from_axis_angle(Point(0, 0, 1),
                                                      Angle.from_degrees(2))))

        # Longitude span > 180 degrees:
        self.assertTrue(
                self.rect_from_degrees(-30, -150, -10, 50).get_cap_bound().
                    approx_equals(Cap.from_axis_angle(Point(0, 0, -1),
                                                      Angle.from_degrees(80))))

    def check_cell_ops(self, r, cell, level):
        # Test the relationship between the given rectangle and cell:
        # 0 == no intersection, 1 == MayIntersect, 2 == Intersects,
        # 3 == Vertex Containment, 4 == Contains
        vertex_contained = False
        for i in range(4):
            if r.contains(cell.get_vertex_raw(i)) \
                    or (not r.is_empty() \
                    and cell.contains(r.get_vertex(i).to_point())):
                vertex_contained = True
        self.assertEqual(r.may_intersect(cell), level >=1)
        self.assertEqual(r.intersects(cell), level >=2)
        self.assertEqual(vertex_contained, level >=3)
        self.assertEqual(r.contains(cell), level >=4)

    def testCellOps(self):

        # Contains(S2Cell), MayIntersect(S2Cell), Intersects(S2Cell)

        # Special cases.
        self.check_cell_ops(LatLonRect.empty(),
            Cell.from_face_pos_level(3, 0, 0), 0)
        self.check_cell_ops(LatLonRect.full(),
            Cell.from_face_pos_level(2, 0, 0), 4)
        self.check_cell_ops(LatLonRect.full(),
            Cell.from_face_pos_level(5, 0, 25), 4)

        # This rectangle includes the first quadrant of face 0.  It's expanded
        # slightly because cell bounding rectangles are slightly conservative.
        r4 = self.rect_from_degrees(-45.1, -45.1, 0.1, 0.1)
        self.check_cell_ops(r4, Cell.from_face_pos_level(0, 0, 0), 3)
        self.check_cell_ops(r4, Cell.from_face_pos_level(0, 0, 1), 4)
        self.check_cell_ops(r4, Cell.from_face_pos_level(1, 0, 1), 0)

        # This rectangle intersects the first quadrant of face 0.
        r5 = self.rect_from_degrees(-10, -45, 10, 0)
        self.check_cell_ops(r5, Cell.from_face_pos_level(0, 0, 0), 3)
        self.check_cell_ops(r5, Cell.from_face_pos_level(0, 0, 1), 3)
        self.check_cell_ops(r5, Cell.from_face_pos_level(1, 0, 1), 0)

        # Rectangle consisting of a single point.
        self.check_cell_ops(self.rect_from_degrees(4, 4, 4, 4),
                    Cell.from_face_pos_level(0, 0, 0), 3)

        # Rectangles that intersect the bounding rectangle of a face
        # but not the face itself.
        self.check_cell_ops(self.rect_from_degrees(41, -87, 42, -79),
                    Cell.from_face_pos_level(2, 0, 0), 1)
        self.check_cell_ops(self.rect_from_degrees(-41, 160, -40, -160),
                    Cell.from_face_pos_level(5, 0, 0), 1)

        # This is the leaf cell at the top right hand corner of face 0.
        # It has two angles of 60 degrees and two of 120 degrees.
        cell0tr = Cell.from_point(Point(1 + 1e-12, 1, 1))
        bound0tr = cell0tr.get_rect_bound()
        v0 = LatLon.from_point(cell0tr.get_vertex_raw(0))
        self.check_cell_ops(self.rect_from_degrees(v0.lat().degrees - 1e-8,
                                    v0.lon().degrees - 1e-8,
                                    v0.lat().degrees - 2e-10,
                                    v0.lon().degrees + 1e-10),
                                    cell0tr, 1)

        # Rectangles that intersect a face but where no vertex of one region
        # is contained by the other region.  The first one passes through
        # a corner of one of the face cells.
        self.check_cell_ops(self.rect_from_degrees(-37, -70, -36, -20),
                    Cell.from_face_pos_level(5, 0, 0), 2)

        # These two intersect like a diamond and a square.
        cell202 = Cell.from_face_pos_level(2, 0, 2)
        bound202 = cell202.get_rect_bound()
        self.check_cell_ops(
                self.rect_from_degrees(bound202.lo().lat().degrees + 3,
                                    bound202.lo().lon().degrees + 3,
                                    bound202.hi().lat().degrees - 3,
                                    bound202.hi().lon().degrees - 3),
                                    cell202, 2)

class TestCrossings(unittest.TestCase):

    def setUp(self):
        self.degen = -2

    def compare_result(self, actual, expected):
        if expected == self.degen:
            self.assertLessEqual(actual, 0)
        else:
            self.assertEqual(expected, actual)

    def check_crossing(self, a, b, c, d, robust, edge_or_vertex, simple):
        a = a.normalize()
        b = b.normalize()
        c = c.normalize()
        d = d.normalize()
        # CompareResult(S2EdgeUtil::RobustCrossing(a, b, c, d), robust)
        if simple:
            self.assertEqual(robust > 0, s2sphere.simple_crossing(a, b, c, d))
        # S2EdgeUtil::EdgeCrosser crosser(&a, &b, &c)
        # CompareResult(crosser.RobustCrossing(&d), robust)
        # CompareResult(crosser.RobustCrossing(&c), robust)

        # EXPECT_EQ(edge_or_vertex,
        #           S2EdgeUtil::EdgeOrVertexCrossing(a, b, c, d))
        # EXPECT_EQ(edge_or_vertex, crosser.EdgeOrVertexCrossing(&d))
        # EXPECT_EQ(edge_or_vertex, crosser.EdgeOrVertexCrossing(&c))

    def check_crossings(self, a, b, c, d, robust, edge_or_vertex, simple):

        self.check_crossing(a, b, c, d, robust, edge_or_vertex, simple)
        self.check_crossing(b, a, c, d, robust, edge_or_vertex, simple)
        self.check_crossing(a, b, d, c, robust, edge_or_vertex, simple)
        self.check_crossing(b, a, d, c, robust, edge_or_vertex, simple)
        self.check_crossing(a, a, c, d, self.degen, 0, False)
        self.check_crossing(a, b, c, c, self.degen, 0, False)
        self.check_crossing(a, b, a, b, 0, 1, False)
        self.check_crossing(
            c, d, a, b, robust, edge_or_vertex ^ (robust == 0), simple)

    def testCrossings(self):
        # The real tests of edge crossings are in s2{loop,polygon}_unittest,
        # but we do a few simple tests here.

        # Two regular edges that cross.
        self.check_crossings(Point(1, 2, 1), Point(1, -3, 0.5),
                      Point(1, -0.5, -3), Point(0.1, 0.5, 3), 1, True, True)

        # Two regular edges that cross antipodal points.
        self.check_crossings(Point(1, 2, 1), Point(1, -3, 0.5),
                      Point(-1, 0.5, 3), Point(-0.1, -0.5, -3), -1, False, True)

        # Two edges on the same great circle.
        self.check_crossings(Point(0, 0, -1), Point(0, 1, 0),
                      Point(0, 1, 1), Point(0, 0, 1), -1, False, True)

        # Two edges that cross where one vertex is S2::Origin().
        self.check_crossings(Point(1, 0, 0), s2sphere.origin(),
                      Point(1, -0.1, 1), Point(1, 1, -0.1), 1, True, True)

        # Two edges that cross antipodal points where one vertex is S2::Origin().
        self.check_crossings(Point(1, 0, 0), Point(0, 1, 0),
                      Point(0, 0, -1), Point(-1, -1, 1), -1, False, True)

        # Two edges that share an endpoint.  The Ortho() direction is (-4,0,2),
        # and edge CD is further CCW around (2,3,4) than AB.
        self.check_crossings(Point(2, 3, 4), Point(-1, 2, 5),
                      Point(7, -2, 3), Point(2, 3, 4), 0, False, True)

        # Two edges that barely cross each other near the middle of one edge.
        # The
        # edge AB is approximately in the x=y plane, while CD is approximately
        # perpendicular to it and ends exactly at the x=y plane.
        self.check_crossings(Point(1, 1, 1), Point(1, np.nextafter(1, 0), -1),
                      Point(11, -12, -1), Point(10, 10, 1), 1, True, False)

        # In this version, the edges are separated by a distance of about 1e-15.
        self.check_crossings(Point(1, 1, 1), Point(1, np.nextafter(1, 2), -1),
                      Point(1, -1, 0), Point(1, 1, 0), -1, False, False)

        # Two edges that barely cross each other near the end of both edges.
        # This example cannot be handled using regular double-precision
        # arithmetic due to floating-point underflow.
        self.check_crossings(Point(0, 0, 1), Point(2, -1e-323, 1),
                      Point(1, -1, 1), Point(1e-323, 0, 1), 1, True, False)

        # In this version, the edges are separated by a dist of about 1e-640.
        self.check_crossings(Point(0, 0, 1), Point(2, 1e-323, 1),
                      Point(1, -1, 1), Point(1e-323, 0, 1), -1, False, False)

        # Two edges that barely cross each other near the middle of one edge.
        # Computing the exact determinant of some of the triangles in this test
        # requires more than 2000 bits of precision.
        self.check_crossings(Point(1, -1e-323, -1e-323),
            Point(1e-323, 1, 1e-323), Point(1, -1, 1e-323), Point(1, 1, 0),
            1, True, False)

        # In this version, the edges are separated by a dist of about 1e-640.
        self.check_crossings(Point(1, 1e-323, -1e-323),
            Point(-1e-323, 1, 1e-323), Point(1, -1, 1e-323), Point(1, 1, 0),
              -1, False, False)


class TestUtils(unittest.TestCase):
    def testDrem(self):
        self.assertAlmostEqual(s2sphere.drem(6.5, 2.3), -0.4)
        self.assertAlmostEqual(s2sphere.drem(1.0, 2.0), 1.0)
        self.assertAlmostEqual(s2sphere.drem(1.0, 3.0), 1.0)


class TestCellUnion(unittest.TestCase):
    def testBasic(self):
        empty = CellUnion([])
        self.assertEqual(0, empty.num_cells())

        face1_id = CellId.from_face_pos_level(1, 0, 0)
        face1_union = CellUnion([face1_id])
        self.assertEqual(1, face1_union.num_cells())
        self.assertEqual(face1_id, face1_union.cell_id(0))

        face2_id = CellId.from_face_pos_level(2, 0, 0)
        face2_union = CellUnion([face2_id.id()])

        self.assertEqual(1, face2_union.num_cells())
        self.assertEqual(face2_id, face2_union.cell_id(0))

        face1_cell = Cell(face1_id)
        face2_cell = Cell(face2_id)
        self.assertTrue(face1_union.contains(face1_cell))
        self.assertFalse(face1_union.contains(face2_cell))

    def add_cells(self, cell_id, selected, input, expected):
        if cell_id == CellId.none():
            for face in range(6):
                self.add_cells(CellId.from_face_pos_level(face, 0, 0),
                               False, input, expected)
            return

        if cell_id.is_leaf():
            assert selected
            input.append(cell_id)
            return

        if not selected and random.randrange(
                CellId.MAX_LEVEL - cell_id.level()) == 0:
            expected.append(cell_id)
            selected = True

        added = False
        if selected and random.randrange(6) != 0:
            input.append(cell_id)
            added = True

        num_children = 0
        for child in cell_id.children():
            crange = 4
            if selected:
                crange = 12
            if random.randrange(crange) == 0 and num_children < 3:
                self.add_cells(child, selected, input, expected)
                num_children += 1

            if selected and not added:
                self.add_cells(child, selected, input, expected)

    def testNormalize(self):
        for i in xrange(NORMALIZE_ITERATIONS):
            input_, expected = [], []
            self.add_cells(CellId.none(), False, input_, expected)

            cellunion = CellUnion(input_)
            self.assertEqual(len(expected), cellunion.num_cells())

            for i in xrange(len(expected)):
                self.assertEqual(expected[i], cellunion.cell_id(i))

            # should test getcapbound here

            for input_j in input_:
                self.assertTrue(cellunion.contains(input_j))
                self.assertTrue(cellunion.contains(input_j.to_point()))
                self.assertTrue(cellunion.intersects(input_j))

                if not input_j.is_face():
                    self.assertTrue(cellunion.intersects(input_j.parent()))
                    if input_j.level() > 1:
                        self.assertTrue(cellunion.intersects(
                            input_j.parent().parent()))
                        self.assertTrue(cellunion.intersects(
                            input_j.parent(0)))

                if not input_j.is_leaf():
                    self.assertTrue(cellunion.contains(
                                        input_j.child_begin()))
                    self.assertTrue(cellunion.intersects(
                                        input_j.child_begin()))
                    self.assertTrue(cellunion.contains(
                                        input_j.child_end().prev()))
                    self.assertTrue(cellunion.intersects(
                                        input_j.child_end().prev()))
                    self.assertTrue(cellunion.contains(
                                        input_j.child_begin(CellId.MAX_LEVEL)))
                    self.assertTrue(cellunion.intersects(
                                        input_j.child_begin(CellId.MAX_LEVEL)))

            for expected_j in expected:
                if not expected_j.is_face():
                    self.assertFalse(cellunion.contains(expected_j.parent()))
                    self.assertFalse(cellunion.contains(expected_j.parent(0)))

            x, y, x_or_y, x_and_y = [], [], [], []
            for input_j in input_:
                in_x = random.randrange(2) == 0
                in_y = random.randrange(2) == 0
                if in_x:
                    x.append(input_j)
                if in_y:
                    y.append(input_j)
                if in_x or in_y:
                    x_or_y.append(input_j)

            xcells = CellUnion(x)
            ycells = CellUnion(y)
            x_or_y_expected = CellUnion(x_or_y)

            x_or_y_cells = CellUnion.get_union(xcells, ycells)
            self.assertEqual(x_or_y_cells, x_or_y_expected)

            for j in xrange(ycells.num_cells()):
                yid = ycells.cell_id(j)
                u = CellUnion.get_intersection(xcells, yid)
                for k in xrange(xcells.num_cells()):
                    xid = xcells.cell_id(k)
                    if xid.contains(yid):
                        self.assertEqual(1, u.num_cells())
                        self.assertEqual(u.cell_id(0), yid)
                    elif yid.contains(xid):
                        self.assertTrue(u.contains(xid))

                for k in xrange(u.num_cells()):
                    self.assertTrue(xcells.contains(u.cell_id(k)))
                    self.assertTrue(yid.contains(u.cell_id(k)))
                x_and_y.extend(u.cell_ids())
            x_and_y_expected = CellUnion(x_and_y)

            x_and_y_cells = CellUnion.get_intersection(xcells, ycells)
            self.assertEqual(x_and_y_cells.cell_ids(),
                             x_and_y_expected.cell_ids())

            x_minus_y_cells = CellUnion.get_difference(xcells, ycells)
            y_minus_x_cells = CellUnion.get_difference(ycells, xcells)
            self.assertTrue(xcells.contains(x_minus_y_cells))
            self.assertFalse(x_minus_y_cells.intersects(ycells))
            self.assertTrue(ycells.contains(y_minus_x_cells))
            self.assertFalse(y_minus_x_cells.intersects(xcells))
            self.assertFalse(x_minus_y_cells.intersects(y_minus_x_cells))

            diff_union = CellUnion.get_union(x_minus_y_cells, y_minus_x_cells)
            diff_intersection_union = CellUnion.get_union(diff_union,
                                                          x_and_y_cells)
            self.assertEqual(diff_intersection_union, x_or_y_cells)

            test, dummy = [], []
            self.add_cells(CellId.none(), False, test, dummy)
            for test_j in test:
                contains, intersects = False, False
                for expected_k in expected:
                    if expected_k.contains(test_j):
                        contains = True
                    if expected_k.intersects(test_j):
                        intersects = True

                self.assertEqual(contains, cellunion.contains(test_j))
                self.assertEqual(intersects, cellunion.intersects(test_j))

    def testEmpty(self):
        empty_cell_union = CellUnion()
        face1_id = CellId.from_face_pos_level(1, 0, 0)

        # Normalize()
        empty_cell_union.normalize()
        self.assertEqual(0, empty_cell_union.num_cells())

        # Denormalize(...)
        self.assertEqual(0, empty_cell_union.num_cells())

        # Contains(...)
        self.assertFalse(empty_cell_union.contains(face1_id))
        self.assertTrue(empty_cell_union.contains(empty_cell_union))

        # Intersects(...)
        self.assertFalse(empty_cell_union.intersects(face1_id))
        self.assertFalse(empty_cell_union.intersects(empty_cell_union))

        # GetUnion(...)
        cell_union = CellUnion.get_union(empty_cell_union, empty_cell_union)
        self.assertEqual(0, cell_union.num_cells())

        # GetIntersection(...)
        intersection = CellUnion.get_intersection(empty_cell_union, face1_id)
        self.assertEqual(0, intersection.num_cells())
        intersection = CellUnion.get_intersection(empty_cell_union,
                                                  empty_cell_union)
        self.assertEqual(0, intersection.num_cells())

        # GetDifference(...)
        difference = CellUnion.get_difference(empty_cell_union,
                                              empty_cell_union)
        self.assertEqual(0, difference.num_cells())

        # Expand(...)
        empty_cell_union.expand(Angle.from_radians(1), 20)
        self.assertEqual(0, empty_cell_union.num_cells())
        empty_cell_union.expand(10)
        self.assertEqual(0, empty_cell_union.num_cells())


class TestRegionCoverer(unittest.TestCase):

    def setUp(self):
        random.seed(20)

    def testRandomCells(self):

        for i in xrange(REGION_COVERER_ITERATIONS):
            coverer = RegionCoverer()
            coverer.max_cells = 1
            cell_id = TestCellId.get_random_cell_id()

            # cell_id = CellId(7981803829394669568)
            covering = coverer.get_covering(Cell(cell_id))
            self.assertEqual(1, len(covering))
            self.assertEqual(cell_id, covering[0])

    def skewed(self, max_long):
        base = random.randint(0, 0xffffffff) % (max_long + 1)
        return random.randint(0, 0xffffffff) & ((1 << base) - 1)

    def random_point(self):
        x = 2 * random.random() - 1
        y = 2 * random.random() - 1
        z = 2 * random.random() - 1
        return Point(x, y, z).normalize()

    def get_random_cap(self, min_area, max_area):
        cap_area = max_area * math.pow(min_area / max_area, random.random())
        assert cap_area >= min_area
        assert cap_area <= max_area
        return Cap.from_axis_area(self.random_point(), cap_area)

    # this is from S2Testing.cc and is called CheckCovering
    def check_cell_union_covering(self, region, covering, check_tight,
                                  cell_id):
        if not cell_id.is_valid():
            for face in range(6):
                self.check_cell_union_covering(
                    region, covering, check_tight,
                    CellId.from_face_pos_level(face, 0, 0))
            return
        if not region.may_intersect(Cell(cell_id)):
            if check_tight:
                self.assertFalse(covering.intersects(cell_id))
        elif not covering.contains(cell_id):
            self.assertFalse(region.contains(Cell(cell_id)))
            self.assertFalse(cell_id.is_leaf())
            for child in cell_id.children():
                self.check_cell_union_covering(
                        region, covering, check_tight, child)

    def check_covering(self, coverer, region, covering, interior):
        min_level_cells = defaultdict(int)
        covering = list(covering)
        for i in xrange(len(covering)):
            level = covering[i].level()
            self.assertGreaterEqual(level, coverer.min_level)
            self.assertLessEqual(level, coverer.max_level)
            self.assertEqual(
                    (level - coverer.min_level) % coverer.level_mod, 0)
            min_level_cells[covering[i].parent(coverer.min_level)] += 1
        if len(covering) > coverer.max_cells:
            for count in min_level_cells.values():
                self.assertEqual(count, 1)

        if interior:
            for i in xrange(len(covering)):
                self.assertTrue(region.contains(Cell(covering[i])))
        else:
            cell_union = CellUnion(covering)
            self.check_cell_union_covering(region, cell_union, True, CellId())

    def testRandomCaps(self):
        for i in xrange(RANDOM_CAPS_ITERATIONS):
            coverer = RegionCoverer()

            coverer.min_level = random.randrange(CellId.MAX_LEVEL + 1)
            coverer.max_level = random.randrange(CellId.MAX_LEVEL + 1)

            while coverer.min_level > coverer.max_level:
                coverer.min_level = random.randrange(CellId.MAX_LEVEL + 1)
                coverer.max_level = random.randrange(CellId.MAX_LEVEL + 1)
            coverer.max_cells = self.skewed(10)
            coverer.level_mod = 1 + random.randrange(1, 3)

            max_area = min(4 * math.pi, (3 * coverer.max_cells + 1) *
                           CellId.avg_area().get_value(coverer.min_level))

            cap = self.get_random_cap(
                0.1 * CellId.avg_area().get_value(CellId.MAX_LEVEL), max_area)
            covering = coverer.get_covering(cap)
            self.check_covering(coverer, cap, covering, False)
            interior = coverer.get_interior_covering(cap)
            self.check_covering(coverer, cap, interior, True)

            # Check deterministic.
            # For some unknown reason the results can be in a different
            # sort order.
            covering2 = coverer.get_covering(cap)
            self.assertEqual(covering.sort(), covering2.sort())

            cells = CellUnion(covering)
            denormalized = cells.denormalize(
                       coverer.min_level, coverer.level_mod)
            self.check_covering(coverer, cap, denormalized, False)

    def testSimpleCoverings(self):
        for i in xrange(SIMPLE_COVERINGS_ITERATIONS):
            coverer = RegionCoverer()
            coverer.max_cells = 0x7fffffff
            level = random.randrange(CellId.MAX_LEVEL + 1)
            coverer.min_level = level
            coverer.max_level = level
            max_area = min(
                    4 * math.pi, 1000 * CellId.avg_area().get_value(level))
            cap = self.get_random_cap(
                0.1 * CellId.avg_area().get_value(CellId.MAX_LEVEL), max_area)
            covering = RegionCoverer.get_simple_covering(cap, cap.axis(),
                                                         level)
            self.check_covering(coverer, cap, covering, False)


if __name__ == '__main__':
    unittest.main()
