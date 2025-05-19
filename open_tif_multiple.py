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
# # 'NE', 'NW', 'SW', 'SE' centered around SPL
# # 'west' == west of specified longitude
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


plot_response = input("Would you like to generate VegDRI plots? (y/n): ").strip().lower()
stats_response = input("Would you like to generate statistics for trimmed US? (y/n): ").strip().lower()
histo_response = input("Would you like to generate VegDRI histograms? (y/n): ").strip().lower()
perc_diff_response = input("Would you like to generate % difference maps? (y/n): ").strip().lower()


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
                

                # Step 5: plot (optional — prompts user to proceed)               
                if plot_response == 'y':
                    # Plot full dataset with quadrant lines
                    filename = os.path.splitext(tif_name)[0]
                    save_path = f'../figures/maps/{filename}.png'
                    processor.plot_ds_epsg4326(
                        ds4326, 
                        var_name='VegDRI', 
                        bounding_box=None, 
                        save_path=save_path,
                        grid_thickness="0", 
                        show_colorbar=None, 
                        colorbar_limits=None,
                        draw_quadrant_lines=True
                    )
                    
                    # plot trimmed (west only) dataset with quadrant lines
                    filename = os.path.splitext(tif_name)[0]
                    save_path = f'../figures/maps/{filename}_{specified_quadrant}.png'
                    processor.plot_ds_epsg4326(
                        ds4326_trimmed, 
                        var_name='VegDRI', 
                        bounding_box=None, 
                        save_path=save_path,
                        grid_thickness="0", 
                        show_colorbar=None, 
                        colorbar_limits=None,
                        draw_quadrant_lines=True
                    )
                else:
                    print("Skipping plot generation.")


# combine datasets
# # 2022 full CONUS
ds_2022 = xr.concat(combined_datasets["2022"], dim="time") if combined_datasets["2022"] else None
ds_2022 = xr.Dataset({"VegDRI": ds_2022})
# # 2025 full CONUS
ds_2025 = xr.concat(combined_datasets["2025"], dim="time") if combined_datasets["2025"] else None
ds_2025 = xr.Dataset({"VegDRI": ds_2025})
# # 2022 trimmed section of CONUS
ds_2022_trimmed = xr.concat(combined_datasets_trimmed["2022"], dim="time") if combined_datasets_trimmed["2022"] else None
ds_2022_trimmed = xr.Dataset({"VegDRI": ds_2022_trimmed})
# # 2025 trimmed section of CONUS
ds_2025_trimmed = xr.concat(combined_datasets_trimmed["2025"], dim="time") if combined_datasets_trimmed["2025"] else None
ds_2025_trimmed = xr.Dataset({"VegDRI": ds_2025_trimmed})


# compute statistics on trimmed section for 2022 and 2025
if stats_response == 'y':
    # # this can be edited in the future to also include CONUS statistics
    csv_filename = f'../data/output/stats/2022_{specified_quadrant}_stats.csv'
    processor.compute_and_export_vegdri_stats(ds_2022_trimmed, csv_name=csv_filename)
    csv_filename = f'../data/output/stats/2025_{specified_quadrant}_stats.csv'
    processor.compute_and_export_vegdri_stats(ds_2025_trimmed, csv_name=csv_filename)


# generate histogram analysis for 2022 and 2025 trimmed and full CONUS
if histo_response == 'y':
    save_path = '../figures/histograms/CONUS_pdf.png'
    processor.plot_histogram_vegdri(ds_2022, ds_2025, "Histogram for US", output_path=save_path)
    save_path = f'../figures/histograms/{specified_quadrant}_pdf.png'
    processor.plot_histogram_vegdri(ds_2022_trimmed, ds_2025_trimmed, "Histogram for Western US", output_path=save_path)


# generate % difference maps for 2022 and 2025 trimmed and full CONUS
if perc_diff_response == 'y':
    # generate % difference maps
    # # full CONUS
    save_path = '../figures/maps/percent_change_CONUS.png'
    processor.plot_percent_change(ds_2022, ds_2025, output_path=save_path)
    processor.plot_percent_change(ds_2022, ds_2025, output_path=save_path, draw_quadrant_lines=True)
    # # western CONUS
    if specified_quadrant == 'west':
        save_path = '../figures/maps/percent_change_western_CONUS.png'
        processor.plot_percent_change(ds_2022_trimmed, ds_2025_trimmed, output_path=save_path)
        processor.plot_percent_change(ds_2022_trimmed, ds_2025_trimmed, output_path=save_path, draw_quadrant_lines=True)