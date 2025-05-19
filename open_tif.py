import rasterio
import matplotlib.pyplot as plt
from tif_utils import TifProcessor
import os
import xarray as xr


# Path to the TIFF file
input_path = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/aridity/Larissa/data/input"
tif_name = "VegDRI_7Day_eVIIRS_Apr_11_2022_Apr_17_2022.tiff"
tif_path = os.path.join(input_path, tif_name)

output_path_3857 = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/aridity/Larissa/data/output/3857"
nc_name_3857 = "VegDRI_7Day_eVIIRS_Apr_11_2022_Apr_17_2022.nc"
nc_path_3857 = os.path.join(output_path_3857, nc_name_3857)

output_path_4326 = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/aridity/Larissa/data/output/4326"
nc_name_4326 = "VegDRI_7Day_eVIIRS_Apr_11_2022_Apr_17_2022.nc"
nc_path_4326 = os.path.join(output_path_4326, nc_name_4326)



processor = TifProcessor()



# Open the TIFF file
#processor.inspect_and_plot_tif(tif_path, cmap='viridis', title='VegDRI 7-Day Product (Band 1)')



if not os.path.exists(nc_path_3857):
    ds_3857 = processor.tif_to_coards_netcdf(
    input_tif=tif_path,
    output_nc=nc_path_3857,
    var_name="VegDRI",
    long_name="Vegetation Drought Response Index",
    units="index"
    )
else:
    ds_3857 = xr.open_dataset(nc_path_3857)


if not os.path.exists(nc_path_4326):
    ds4326 = processor.reproject_to_4326(ds_3857, nc_path_4326, 'VegDRI')
    ds4326 = processor.mask_non_conus_and_save(ds4326, var_name='VegDRI', output_nc=nc_path_4326)
else:
    ds4326 = xr.open_dataset(nc_path_4326)



fig_path = ''
processor.plot_ds_epsg4326(ds4326, 
                           var_name='VegDRI', 
                           colormap='viridis', 
                           bounding_box=None, 
                           save_path=None, 
                           grid_thickness="0", 
                           show_colorbar=None, 
                           colorbar_limits=None)