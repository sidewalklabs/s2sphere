.. _quickstart:


Examples
========

.. code-block:: python

    import s2sphere

    r = s2sphere.RegionCoverer()
    p1 = s2sphere.LatLon.from_degrees(33, -122)
    p2 = s2sphere.LatLon.from_degrees(33.1, -122.1)
    cell_ids = r.get_covering(s2sphere.LatLonRect.from_point_pair(p1, p2))
    print(cell_ids)
