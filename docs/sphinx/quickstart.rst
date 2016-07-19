.. _quickstart:


Examples
========

One of the standard applications is to get a set of S2 cells at various levels
covering a rectangle in :math:`(lat, lng)` coordinates:

.. code-block:: python

    import s2sphere

    r = s2sphere.RegionCoverer()
    p1 = s2sphere.LatLng.from_degrees(33, -122)
    p2 = s2sphere.LatLng.from_degrees(33.1, -122.1)
    cell_ids = r.get_covering(s2sphere.LatLngRect.from_point_pair(p1, p2))
    print(cell_ids)

which prints this list:

.. code-block:: sh

    [9291041754864156672, 9291043953887412224, 9291044503643226112, 9291045878032760832, 9291047252422295552, 9291047802178109440, 9291051650468806656, 9291052200224620544]

