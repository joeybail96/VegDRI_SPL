import rasterio
import matplotlib.pyplot as plt
from tif_utils import TifProcessor
import os
import xarray as xr
import pandas as pd


# directories
input_path = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/aridity/Larissa/data/input"
output_path_3857 = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/aridity/Larissa/data/output/3857"
output_path_4326 = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/aridity/Larissa/data/output/4326"
fig_output_dir = "/uufs/chpc.utah.edu/common/home/hallar-group2/climatology/aridity/Larissa/figures"


# specify a quadarant that will be used for analysis
specified_quadrant = 'SW'

# initialize processor class object
processor = TifProcessor()

# datasets that will be used for statistics
combined_datasets = {
    "2022": [],
    "2025": []
}
combined_datasets_trimmed = {
    "2022": [],
    "2025": []
}


# loop through all files in input path
for tif_name in os.listdir(input_path):
    
    # vegdri data is stored in tif files
    if tif_name.endswith(".tiff") or tif_name.endswith(".tif"):
        
        # retrieve the full path to the tif file
        tif_path = os.path.join(input_path, tif_name)

        # generate output NetCDF names based on input tif filename
        # # same file name with .nc at the end
        base_name = os.path.splitext(tif_name)[0] + ".nc"
        nc_path_3857 = os.path.join(output_path_3857, base_name) # stored in output/3857
        nc_path_4326 = os.path.join(output_path_4326, base_name) # stored in output/4326

        print(f"\n--- Processing: {tif_name} ---")

        # Step 1: convert TIFF to NetCDF (EPSG:3857) if not already converted
        if not os.path.exists(nc_path_3857):
            ds_3857 = processor.tif_to_coards_netcdf(
                input_tif=tif_path,
                output_nc=nc_path_3857,
                var_name="VegDRI",
                long_name="Vegetation Drought Response Index",
                units="index"
            )
        # if already converted, open existing .nc file
        else:
            ds_3857 = xr.open_dataset(nc_path_3857)

        # Step 2: reproject and mask to CONUS (EPSG:4326)
        if not os.path.exists(nc_path_4326):
            ds4326 = processor.reproject_to_4326(ds_3857, nc_path_4326, 'VegDRI')
        # if already converted, open existing .nc file
        else:
            ds4326 = xr.open_dataset(nc_path_4326)

        # Step 3: trim data data
        # # if quadrant is specified, it will trim to NE, NW, SE, SW quadrant around SPL
        ds4326_trimmed = processor.trim_map(ds4326, quadrant=specified_quadrant)
        
        # Step 4: accumulate data into 2022 or 2025 dataset
        date = processor.extract_week_end_date(tif_name)
        if date:
            # extract year from date
            year = str(date.year)
            if year in combined_datasets:
                # add a time dimension to the 2D DataArray
                data_array_trimmed = ds4326_trimmed["VegDRI"].expand_dims(time=[pd.Timestamp(date)])
                combined_datasets_trimmed[year].append(data_array_trimmed)
                # add 2022 and 2025 data to corresponding year
                data_array = ds4326["VegDRI"].expand_dims(time=[pd.Timestamp(date)])
                combined_datasets[year].append(data_array)
                

        # # Step 5: Plot (optional — can skip or add save_path)
        # fig_filename = os.path.splitext(tif_name)[0] + "_lines.png"
        # fig_save_path = os.path.join(fig_output_dir, fig_filename)
        # processor.plot_ds_epsg4326(
        #     ds4326, 
        #     var_name='VegDRI', 
        #     bounding_box=None, 
        #     save_path=fig_save_path,  # Or provide a path like f"/some/dir/{base_name}.png"
        #     grid_thickness="0", 
        #     show_colorbar=None, 
        #     colorbar_limits=None,
        #     draw_quadrant_lines=True
        # )
        
        
        # fig_filename = os.path.splitext(tif_name)[0] + "_lines_westONLY.png"
        # fig_save_path = os.path.join(fig_output_dir, fig_filename)   
        # processor.plot_ds_epsg4326(
        #     ds4326_trimmed, 
        #     var_name='VegDRI', 
        #     bounding_box=None, 
        #     save_path=fig_save_path,  # Or provide a path like f"/some/dir/{base_name}.png"
        #     grid_thickness="0", 
        #     show_colorbar=None, 
        #     colorbar_limits=None,
        #     draw_quadrant_lines=True
        # )
        


# Combine datatsets
ds_2022 = xr.concat(combined_datasets["2022"], dim="time") if combined_datasets["2022"] else None
ds_2022 = xr.Dataset({"VegDRI": ds_2022})

ds_2025 = xr.concat(combined_datasets["2025"], dim="time") if combined_datasets["2025"] else None
ds_2025 = xr.Dataset({"VegDRI": ds_2025})

ds_2022_trimmed = xr.concat(combined_datasets_trimmed["2022"], dim="time") if combined_datasets_trimmed["2022"] else None
ds_2022_trimmed = xr.Dataset({"VegDRI": ds_2022_trimmed})

ds_2025_trimmed = xr.concat(combined_datasets_trimmed["2025"], dim="time") if combined_datasets_trimmed["2025"] else None
ds_2025_trimmed = xr.Dataset({"VegDRI": ds_2025_trimmed})


csv_filename = f'2022_{specified_quadrant}_stats.csv'
processor.compute_and_export_vegdri_stats(ds_2022_trimmed, csv_name=csv_filename)
csv_filename = f'2025_{specified_quadrant}_stats.csv'
processor.compute_and_export_vegdri_stats(ds_2025_trimmed, csv_name=csv_filename)


# if specified_quadrant is not None:
#     save_filename = f'2week_histogram_CONUS_Quadrant_{specified_quadrant}.png'

# else:
#     save_filename = '2week_histogram_CONUS.png'
# save_path = os.path.join(fig_output_dir, save_filename)
# # processor.plot_histogram_vegdri(ds_2022, ds_2025, "Histogram for US", output_path=save_path)

# save_path = os.path.join(fig_output_dir, save_filename)
# processor.plot_histogram_vegdri(ds_2022_trimmed, ds_2025_trimmed, "Histogram for Western US", output_path=save_path)


# summary_df = pd.DataFrame(summary_list)
# summary_df.to_csv(f"{specified_quadrant}_VegDRI_summary_stats.csv", index=False)
# print(summary_df.to_string(index=False))


# save_filename = 'percent_change_lines.png'
# save_path = os.path.join(fig_output_dir, save_filename)
# processor.plot_percent_change(ds_2022, ds_2025, output_path=save_path, draw_quadrant_lines=True)


# save_filename = 'percent_change_CONUS.png'
# save_path = os.path.join(fig_output_dir, save_filename)
# processor.plot_percent_change(ds_2022, ds_2025, output_path=save_path)
