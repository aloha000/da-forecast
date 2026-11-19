import obstore
import xarray as xr
import zarr

# Open an hourly SatWeave Zarr store on Google Cloud Storage
# (For the 2016-2025 30-minute archive, use prefix="zarr/2016_to_2025/mosaic.zarr")
gcs_store = obstore.store.GCSStore(
    bucket="satweave_1",
    prefix="zarr/2026_to_present/20260901T120000/mosaic.zarr",
    skip_signature=True,
)
ds = xr.open_zarr(zarr.storage.ObjectStore(gcs_store), chunks={})
print(ds)
