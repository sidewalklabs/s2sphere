from cpython cimport array
import array

# as in s2-geometry/geometry/base/integral_types.h
ctypedef unsigned short uint16
ctypedef unsigned long long int64

# Constants for CellId
DEF LOOKUP_BITS = 4
DEF SWAP_MASK = 0x01
DEF INVERT_MASK = 0x02

POS_TO_IJ = ((0, 1, 3, 2),
             (0, 2, 3, 1),
             (3, 2, 0, 1),
             (3, 1, 0, 2))
cdef array.array POS_TO_ORIENTATION = array.array(
    'i', [SWAP_MASK, 0, 0, INVERT_MASK | SWAP_MASK])
cdef uint16 LOOKUP_POS[1 << (2 * LOOKUP_BITS + 2)]
cdef uint16 LOOKUP_IJ[1 << (2 * LOOKUP_BITS + 2)]


def _init_lookup_cell(level, i, j, orig_orientation, pos, orientation):
    if level == LOOKUP_BITS:
        ij = (i << LOOKUP_BITS) + j
        LOOKUP_POS[(ij << 2) + orig_orientation] = (pos << 2) + orientation
        LOOKUP_IJ[(pos << 2) + orig_orientation] = (ij << 2) + orientation
    else:
        level = level + 1
        i <<= 1
        j <<= 1
        pos <<= 2
        r = POS_TO_IJ[orientation]
        for index in range(4):
            _init_lookup_cell(
                level, i + (r[index] >> 1),
                j + (r[index] & 1), orig_orientation,
                pos + index, orientation ^ POS_TO_ORIENTATION[index],
            )

_init_lookup_cell(0, 0, 0, 0, 0, 0)
_init_lookup_cell(0, 0, 0, SWAP_MASK, 0, SWAP_MASK)
_init_lookup_cell(0, 0, 0, INVERT_MASK, 0, INVERT_MASK)
_init_lookup_cell(0, 0, 0, SWAP_MASK | INVERT_MASK, 0, SWAP_MASK | INVERT_MASK)


def to_face_ij_orientation(id_, face, lsb, max_level):
    cdef int64 cid = id_
    cdef int64 i = 0
    cdef int64 j = 0
    cdef int bits = (face & SWAP_MASK)
    cdef int nbits = 0

    for k in range(7, -1, -1):
        if k == 7:
            nbits = (max_level - 7 * LOOKUP_BITS)
        else:
            nbits = LOOKUP_BITS

        bits += (
            cid >> (k * 2 * LOOKUP_BITS + 1) &
            ((1 << (2 * nbits)) - 1)
        ) << 2

        bits = LOOKUP_IJ[bits]
        i += (bits >> (LOOKUP_BITS + 2)) << (k * LOOKUP_BITS)
        j += ((bits >> 2) & ((1 << LOOKUP_BITS) - 1)) << (k * LOOKUP_BITS)
        bits &= (SWAP_MASK | INVERT_MASK)

    assert 0 == POS_TO_ORIENTATION[2]
    assert SWAP_MASK == POS_TO_ORIENTATION[0]
    if (lsb & 0x1111111111111110) != 0:
        bits ^= SWAP_MASK
    cdef int64 orientation = bits

    return face, i, j, orientation
