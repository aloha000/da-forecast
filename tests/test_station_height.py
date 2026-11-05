"""Coordinate conversion, geoid sign/interpolation and exact-station header parsing."""
import gzip
import math
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from preprocessing.build_station_height import ecef_height, geoid_at, read_station

class StationHeightTest(unittest.TestCase):
    def test_ecef_roundtrip(self):
        a=6378137.
        f=1/298.257223563
        e2=f*(2-f)
        for latitude,longitude,h in ((0,0,0),(45,123,1000),(-75,359.9,40),(89.999,-180,-100)):
            phi,lam=math.radians(latitude),math.radians(longitude)
            n=a/math.sqrt(1-e2*math.sin(phi)**2)
            xyz=((n+h)*math.cos(phi)*math.cos(lam),(n+h)*math.cos(phi)*math.sin(lam),
                 (n*(1-e2)+h)*math.sin(phi))
            lat,lon,height=ecef_height(*xyz)
            self.assertAlmostEqual(lat,latitude,places=7)
            self.assertAlmostEqual((lon-longitude+180)%360-180,0,places=7)
            self.assertAlmostEqual(height,h,places=4)

    def test_grid_nodes_midpoints_poles_and_periodic_longitude(self):
        grid=np.broadcast_to(np.arange(721)[:,None],(721,1440)).astype(float).copy()
        grid+=np.arange(1440)[None,:]*.01
        self.assertAlmostEqual(float(geoid_at(grid,45,0)),grid[180,720])
        self.assertAlmostEqual(float(geoid_at(grid,44.875,.125)),
                               grid[180:182,720:722].mean())
        self.assertAlmostEqual(float(geoid_at(grid,-90,-180)),grid[720,0])
        self.assertAlmostEqual(float(geoid_at(grid,90,180)),grid[0,0])
        self.assertAlmostEqual(float(geoid_at(grid,0,359.9)),float(geoid_at(grid,0,-.1)))
        with self.assertRaises(ValueError):
            geoid_at(grid,91,0)

    def test_station_is_selected_by_exact_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"TEST").mkdir()
            header=("+SITE/ID\n TEST A --------- P x 0 0 0 0 0 0 100.0\n-SITE/ID\n"
                    "+TROP/STA_COORDINATES\n OTHR A 1 P 6378137 0 0 IGS20_ NGL\n"
                    " TEST A 1 P 6378237 0 0 IGS20_ NGL\n-TROP/STA_COORDINATES\n")
            with zipfile.ZipFile(root/"TEST/TEST.2023.trop.zip","w") as archive:
                archive.writestr("test.gz",gzip.compress(header.encode("ascii")))
            row=read_station((root,"TEST"))
            self.assertEqual(row["station"],"TEST")
            self.assertAlmostEqual(row["ellipsoidal_height_m"],100)
            n=float(geoid_at(np.full((721,1440),20.),row["latitude"],row["longitude"]))
            self.assertAlmostEqual(row["ellipsoidal_height_m"]-n,80)
            with self.assertRaisesRegex(ValueError,"no valid"):
                read_station((root,"NONE"))

if __name__=="__main__":
    unittest.main()
