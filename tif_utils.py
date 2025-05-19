import numpy as np
import xarray as xr
import rioxarray
from osgeo import gdal
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import natural_earth
import geopandas as gpd
import shapely.vectorized
from shapely.geometry import box
import rasterio
import pandas as pd
from datetime import datetime
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
from cartopy.feature import ShapelyFeature
import cartopy.io.shapereader as shpreader
from cartopy.io.shapereader import Reader, natural_earth
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

class TifProcessor:
    def __init__(self):
        pass

    def inspect_and_plot_tif(self, tif_path, cmap='viridis', title='VegDRI 7-Day Product (Band 1)'):
        """
        Opens a GeoTIFF, prints metadata, and plots the first band.
    
        Parameters:
        - tif_path (str): Path to the .tif file
        - cmap (str): Colormap to use for plotting
        - title (str): Title for the plot
        """
        with rasterio.open(tif_path) as src:
            print("TIFF Metadata:")
            print(f" - Driver: {src.driver}")
            print(f" - Width: {src.width}")
            print(f" - Height: {src.height}")
            print(f" - Count (Bands): {src.count}")
            print(f" - CRS: {src.crs}")
            print(f" - Transform: {src.transform}")
            
            # Read the first band
            band1 = src.read(1)
    
        # Plot the first band
        plt.figure(figsize=(10, 6))
        plt.imshow(band1, cmap=cmap)
        plt.colorbar(label="Pixel Value")
        plt.title(title)
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.tight_layout()
        plt.show()


    def tif_to_coards_netcdf(self, input_tif, output_nc, var_name, long_name, units, reproject_to_4326=True):
        # Open the TIFF using GDAL
        dataset = gdal.Open(input_tif)
        geotransform = dataset.GetGeoTransform()
    
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        x_indices = np.arange(width)
        y_indices = np.arange(height)
    
        # Calculate x and y coordinates
        x = geotransform[0] + x_indices * geotransform[1]
        y = geotransform[3] + y_indices * geotransform[5]
    
        # Read the data from the first band
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray().astype(np.float32)
        #data[data == 99999] = np.nan  # Replace NoData with NaN
        # Mask values >= 255
        #data[data >= 255] = np.nan
    
        # Build the xarray Dataset
        x_array = xr.DataArray(x, dims="x", name="x",
                               attrs={"long_name": "Easting", "units": "meters", "axis": "X"})
        y_array = xr.DataArray(y, dims="y", name="y",
                               attrs={"long_name": "Northing", "units": "meters", "axis": "Y"})
        data_array = xr.DataArray(data, dims=["y", "x"], name=var_name,
                                  attrs={"long_name": long_name, "units": units})
        ds = xr.Dataset({var_name: data_array}, coords={"x": x_array, "y": y_array})
        
        # Add CRS
        ds = ds.rio.write_crs("EPSG:3857")  # Original projection
    
        # Save to NetCDF
        ds.to_netcdf(output_nc)
        return ds
    
    
    
    
    def reproject_to_4326(self, ds, output_nc, var_name):
        da = ds[var_name]
        da = da.rio.set_spatial_dims(x_dim='x', y_dim='y')  # Use x and y as spatial dimensions initially
        da = da.rio.write_crs("EPSG:3857")  # Set the CRS to EPSG:3857 (original projection)
        
        # Reproject to EPSG:4326
        reprojected = da.rio.reproject("EPSG:4326")
        
        # Rename x and y to lon and lat
        reprojected = reprojected.rename({'x': 'lon', 'y': 'lat'})
        
        # Drop spatial_ref since it's no longer needed
        reprojected = reprojected.drop_vars('spatial_ref')
        
        # Set the variable name and save to new dataset
        reprojected.name = var_name
        ds_out = reprojected.to_dataset()
        
        # Save the reprojected dataset to NetCDF
        ds_out.to_netcdf(output_nc)
        
        return ds_out
    


    def mask_non_conus_and_save(self, ds, var_name, output_nc):

    
        # Load country boundaries
        shp_path = natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
        countries = gpd.read_file(shp_path)
        usa = countries[countries['ADMIN'] == 'United States of America']
    
        # Define CONUS bounding box
        conus_box = box(-125, 24, -66.5, 50)
        conus_gdf = gpd.GeoDataFrame(geometry=[conus_box], crs=usa.crs)
    
        # Clip to CONUS
        conus = gpd.overlay(usa, conus_gdf, how='intersection')
        conus_geom = conus.unary_union
    
        # Prepare the mask
        lat = ds['lat'].values
        lon = ds['lon'].values
        Lon, Lat = np.meshgrid(lon, lat)
        mask = shapely.vectorized.contains(conus_geom, Lon, Lat)
    
        # Apply mask
        data = ds[var_name].values
        data_masked = np.where(mask, data, np.nan)
    
        # Replace data in the dataset
        ds[var_name].values = data_masked
    
        # Save new NetCDF
        ds.to_netcdf(output_nc)
        return ds
    
    
    
    
    def mask_above_252(self, ds, var_name, output_nc=None):
        """
        Returns a copy of the input Dataset where all data values >252
        in all data variables are set to NaN.
        """
        ds_masked = ds.copy()
    
        ds_masked[var_name] = ds_masked[var_name].where(ds_masked[var_name] < 252)
        
        if output_nc:
            ds_masked.to_netcdf(output_nc)
            
        return ds_masked


    
 
    
    def plot_ds_epsg4326(
        self, ds, var_name,
        bounding_box=None,
        save_path=None, grid_thickness="0", show_colorbar=True,
        colorbar_limits=None, draw_quadrant_lines=False
    ):
        crs_proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': crs_proj})
    
        # Apply bounding box if provided
        if bounding_box:
            min_lon, max_lon, min_lat, max_lat = bounding_box
            ds = ds.where(
                (ds.lon >= min_lon) & (ds.lon <= max_lon) &
                (ds.lat >= min_lat) & (ds.lat <= max_lat),
                drop=True
            )
    
        lon = ds['lon'].values
        lat = ds['lat'].values
        z = ds[var_name].values
        Lon, Lat = np.meshgrid(lon, lat)
    
        boundaries = [0, 64, 81, 97, 113, 161, 178, 193, 253, 254, 255, 256]
        colors = [
            '#800000', '#FF0000', '#FFA500', '#FFFF00', '#FFFFFF',
            '#90EE90', '#008000', '#006400', '#00008B', '#D3D3D3', '#FFFFFF'
        ]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(boundaries, ncolors=len(colors))
    
        # Add water
        ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#3b9b9b', zorder=1)
        ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='#3b9b9b', zorder=1)
    
        # Mask Canada & Mexico
        shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
        for record in shpreader.Reader(shapefile).records():
            if record.attributes['NAME'] in ['Mexico', 'Canada']:
                geometry = record.geometry
                feature = ShapelyFeature([geometry], crs=crs_proj, facecolor='gray', edgecolor='none')
                ax.add_feature(feature, zorder=3)
    
        # Plot data
        img = ax.pcolormesh(
            Lon, Lat, z,
            transform=crs_proj,
            cmap=cmap,
            norm=norm,
            shading='auto',
            edgecolors='black',
            linewidth=float(grid_thickness),
            zorder=2
        )
    
        # Add borders
        ax.add_feature(cfeature.BORDERS, linewidth=1, zorder=4)
        ax.add_feature(cfeature.STATES.with_scale('10m'), linewidth=0.5, edgecolor='black', zorder=4)
        ax.coastlines(resolution='10m', linewidth=0.5, zorder=4)
    
        # Oceans & Lakes (background)
        for name in ['ocean', 'lakes']:
            shp = natural_earth(resolution='10m', category='physical', name=name)
            geom = Reader(shp).geometries()
            color = '#3b9b9b'
            edge = 'none' if name == 'ocean' else 'black'
            feature = ShapelyFeature(geom, crs_proj, facecolor=color, edgecolor=edge)
            ax.add_feature(feature, zorder=4)
    
        # Optional: Draw quadrant lines
        if draw_quadrant_lines:
            # Get the dataset boundaries (min_lon, max_lon, min_lat, max_lat, lon_midpoint, lat_midpoint)
            min_lon, max_lon, min_lat, max_lat, lon_midpoint, lat_midpoint = self.calculate_boundaries(ds)
        
            # Vertical dividing line at the lon_midpoint (longitude)
            ax.plot([lon_midpoint, lon_midpoint], [min_lat, max_lat], color='black', linestyle='--', linewidth=6, transform=crs_proj, zorder=5)
        
            # Horizontal dividing line at the lat_midpoint (latitude)
            ax.plot([min_lon, max_lon], [lat_midpoint, lat_midpoint], color='black', linestyle='--', linewidth=6, transform=crs_proj, zorder=5)

        # Colorbar
        if show_colorbar:
            cbar = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.05, boundaries=boundaries)
            cbar.set_label(var_name)
            cbar.set_ticks(boundaries[:-3])
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()       
    

    def calculate_boundaries(self, ds, spl=True):
            """
            Calculates the boundaries (min_lon, max_lon, min_lat, max_lat, lon_midpoint, lat_midpoint)
            from the dataset to be used in plotting and further processing.
            
            Parameters:
                ds (xarray.Dataset): Input dataset with 'lat' and 'lon' coordinates.
                lon_boundary (float): Longitude that separates west from east (default is -100).
                
            Returns:
                tuple: (min_lon, max_lon, min_lat, max_lat, lon_midpoint, lat_midpoint)
            """
            
            ds = self.mask_above_252(ds, 'VegDRI')
            ds = ds.where(ds['VegDRI'].notnull(), drop=True)
            
            
            # Get min and max latitudes and longitudes from the dataset
            min_lon = ds.lon.min().values
            max_lon = ds.lon.max().values
            min_lat = ds.lat.min().values
            max_lat = ds.lat.max().values
            
            # Calculate midpoints for longitude and latitude
            if spl:
                lon_midpoint = -106.744
                lat_midpoint = 40.455
            else:
                lon_midpoint = (min_lon + max_lon) / 2.0
                lat_midpoint = (min_lat + max_lat) / 2.0
            
            return min_lon, max_lon, min_lat, max_lat, lon_midpoint, lat_midpoint
            


    def trim_map(self, ds, lon_boundary=-100.0, quadrant='west'):
        """
        Trims the dataset to the western U.S., or a specified quadrant within the western U.S.
        
        Parameters:
            ds (xarray.Dataset): Input dataset with 'lat' and 'lon' coordinates.
            lon_boundary (float): Longitude that separates west from east (default is -100).
            quadrant (str, optional): One of 'NW', 'NE', 'SW', 'SE' for subregions of western U.S.
        
        Returns:
            xarray.Dataset: Subset of the dataset.
        """
        # Subset the dataset based on the lon_boundary
        western_ds = ds.sel(lon=ds.lon[(ds.lon >= ds.lon.min().values) & (ds.lon <= lon_boundary)])
    
        # If no quadrant is specified, return the full western dataset
        if quadrant == 'west':
            return western_ds
        
        # Calculate the boundaries only if quadrant is specified
        min_lon, max_lon, min_lat, max_lat, lon_midpoint, lat_midpoint = self.calculate_boundaries(ds, spl=True)

        # Subset the dataset by quadrant
        if quadrant == 'NW':
            quad_ds = ds.sel(
                lat=ds.lat[ds.lat >= lat_midpoint],
                lon=ds.lon[ds.lon <= lon_midpoint]
            )
        elif quadrant == 'NE':
            quad_ds = ds.sel(
                lat=ds.lat[ds.lat >= lat_midpoint],
                lon=ds.lon[ds.lon > lon_midpoint]
            )
        elif quadrant == 'SW':
            quad_ds = ds.sel(
                lat=ds.lat[ds.lat < lat_midpoint],
                lon=ds.lon[ds.lon <= lon_midpoint]
            )
        elif quadrant == 'SE':
            quad_ds = ds.sel(
                lat=ds.lat[ds.lat < lat_midpoint],
                lon=ds.lon[ds.lon > lon_midpoint]
            )
        else:
            raise ValueError(f"Invalid quadrant: {quadrant}. Use 'NW', 'NE', 'SW', 'SE', or None.")
        
        return quad_ds
        



    def summarize_dataset_stats(self, ds, var_name, filename):
        """
        Computes basic statistics for a variable in an xarray Dataset.
        
        Parameters:
            ds (xarray.Dataset): Input dataset.
            var_name (str): Name of the variable to compute statistics for.
        
        Returns:
            pandas.Series: Summary statistics.
        """
        data = ds[var_name].values
        data_flat = data.flatten()
        # Mask out NaNs and values > 252
        data_valid = data_flat[(~np.isnan(data_flat)) & (data_flat <= 252)]
    
        stats = {
            'count': data_valid.size,
            'nan_count': np.isnan(data_flat).sum(),
            'mean': np.mean(data_valid),
            'median': np.median(data_valid),
            'min': np.min(data_valid),
            'max': np.max(data_valid),
            'std': np.std(data_valid),
            '25th_percentile': np.percentile(data_valid, 25),
            '75th_percentile': np.percentile(data_valid, 75),
        }
        
        # Plot histogram
        plt.figure(figsize=(6, 4))
        plt.hist(data_valid, bins=50, color='skyblue', edgecolor='black')
        plt.title(f"{filename} Histogram of {var_name}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    
        return pd.Series(stats)


    def extract_week_end_date(self, tif_name):
        try:
            # Example: VegDRI_7Day_eVIIRS_Apr_4_2022_Apr_10_2022
            parts = tif_name.split("_")
            end_month = parts[-3]
            end_day = parts[-2]
            end_year = parts[-1].split(".")[0]  # remove .tif or .tiff
            return datetime.strptime(f"{end_month} {end_day} {end_year}", "%b %d %Y")
        except Exception as e:
            print(f"Failed to extract date from: {tif_name} ({e})")
            return None




    def accumulate_by_year(self, ds, year, accumulators, var_name="VegDRI"):
        """
        Adds the trimmed western-US VegDRI data for a given year into an accumulator list.
        """
        if year not in accumulators:
            accumulators[year] = []
        accumulators[year].append(ds[var_name])


    def plot_histogram_vegdri(self, ds_2022, ds_2025, title, output_path=None):
        data_2022 = ds_2022["VegDRI"].values.flatten()
        data_2025 = ds_2025["VegDRI"].values.flatten()
        data_2022 = data_2022[~np.isnan(data_2022) & (data_2022 <= 252)]
        data_2025 = data_2025[~np.isnan(data_2025) & (data_2025 <= 252)]
    
        bins = 51
        range_vals = (0, 252)
    
        hist_2022_count, _ = np.histogram(data_2022, bins=bins, range=range_vals, density=False)
        hist_2025_count, _ = np.histogram(data_2025, bins=bins, range=range_vals, density=False)
        ymax_count = max(hist_2022_count.max(), hist_2025_count.max()) * 1.1
    
        hist_2022_density, _ = np.histogram(data_2022, bins=bins, range=range_vals, density=True)
        hist_2025_density, _ = np.histogram(data_2025, bins=bins, range=range_vals, density=True)
        ymax_density = max(hist_2022_density.max(), hist_2025_density.max()) * 1.1
    
        # Prepare figure
        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])
    
        # Top Left: 2022 Count
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.hist(data_2022, bins=bins, range=range_vals, color='blue', edgecolor='white', alpha=0.7, density=False)
        ax00.set_title("2022 - Pixel Count")
        ax00.set_ylim(0, ymax_count)
        ax00.set_ylabel("Pixel Count (1e6)")
        ax00.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 1e-6:.1f}'))
        ax00.set_xticks(np.arange(0, 260, 20))
    
        # Top Right: 2022 Density
        ax01 = fig.add_subplot(gs[0, 1])
        ax01.hist(data_2022, bins=bins, range=range_vals, color='blue', edgecolor='white', alpha=0.7, density=True)
        ax01.set_title("2022 - Density")
        ax01.set_ylim(0, ymax_density)
        ax01.set_ylabel("Probability")
        ax01.set_xticks(np.arange(0, 260, 20))
    
        # Middle Left: 2025 Count
        ax10 = fig.add_subplot(gs[1, 0])
        ax10.hist(data_2025, bins=bins, range=range_vals, color='red', edgecolor='white', alpha=0.7, density=False)
        ax10.set_title("2025 - Pixel Count")
        ax10.set_ylim(0, ymax_count)
        ax10.set_ylabel("Pixel Count (1e6)")
        ax10.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 1e-6:.1f}'))
        ax10.set_xticks(np.arange(0, 260, 20))
    
        # Middle Right: 2025 Density
        ax11 = fig.add_subplot(gs[1, 1])
        ax11.hist(data_2025, bins=bins, range=range_vals, color='red', edgecolor='white', alpha=0.7, density=True)
        ax11.set_title("2025 - Density")
        ax11.set_ylim(0, ymax_density)
        ax11.set_ylabel("Probability")
        ax11.set_xticks(np.arange(0, 260, 20))
    
        # Bottom Full-width: KDE Line Plot
        ax20 = fig.add_subplot(gs[2, :])
        kde_2022 = gaussian_kde(data_2022)
        kde_2025 = gaussian_kde(data_2025)
        x_vals = np.linspace(0, 252, 500)
        ax20.plot(x_vals, kde_2022(x_vals), label='2022', color='blue')
        ax20.plot(x_vals, kde_2025(x_vals), label='2025', color='red')
        ax20.set_title("2022 vs 2025 - PDF (Smoothed KDE)")
        ax20.set_ylabel("Density")
        ax20.set_xlabel("VegDRI Value")
        ax20.set_xticks(np.arange(0, 260, 20))
        ax20.set_ylim(0, max(kde_2022(x_vals).max(), kde_2025(x_vals).max()) * 1.1)
        ax20.legend()
    
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
        if output_path:
            plt.savefig(output_path, dpi=300)
        else:
            plt.show()
    


    def plot_percent_change(self, ds_2022, ds_2025, output_path=None, draw_quadrant_lines=False):
        """
        Averages VegDRI over the two time steps in each dataset,
        calculates percent change between 2022 and 2025,
        and plots the result with consistent styling.
        
        Parameters:
            draw_dividing_lines (bool): Whether to draw vertical/horizontal lines dividing the western US.
        """
        crs_proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': crs_proj})
    
        # Step 1: Average over time
        avg_2022 = ds_2022['VegDRI'].mean(dim='time')
        avg_2025 = ds_2025['VegDRI'].mean(dim='time')
    
        # Step 2: Mask out fill values (e.g., 255)
        valid_mask = (avg_2022 != 255.0) & (avg_2025 != 255.0)
        avg_2022 = avg_2022.where(valid_mask)
        avg_2025 = avg_2025.where(valid_mask)
    
        # Step 3: Percent change calculation
        percent_change = ((avg_2025 - avg_2022) / avg_2022) * 100
    
        # Step 4: Create gray mask where either dataset has a 254
        gray_mask = ((avg_2022 == 254.0) | (avg_2025 == 254.0)).astype(float)
        gray_mask = gray_mask.where(gray_mask == 1)
    
        # Meshgrid for plotting
        lon = ds_2022['lon'].values
        lat = ds_2022['lat'].values
        Lon, Lat = np.meshgrid(lon, lat)
    
        # Add dark blue water
        ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='lightblue', zorder=1)
        ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='lightblue', zorder=1)
    
        # Plot percent change
        img = ax.pcolormesh(
            Lon, Lat, percent_change,
            transform=crs_proj,
            shading='auto',
            cmap='RdBu',
            vmin=-100, vmax=100,
            zorder=2
        )
    
        # Plot gray 254 mask
        ax.pcolormesh(
            Lon, Lat, gray_mask,
            transform=crs_proj,
            shading='auto',
            cmap=ListedColormap(['white']),
            zorder=2
        )
    
        # Mask out Mexico and Canada
        shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
        reader = shpreader.Reader(shapefile)
        for record in reader.records():
            if record.attributes['NAME'] in ['Mexico', 'Canada']:
                geometry = record.geometry
                feature = ShapelyFeature([geometry], crs=crs_proj, facecolor='gray', edgecolor='none')
                ax.add_feature(feature, zorder=3)
    
        # Add borders and coastlines
        ax.add_feature(cfeature.BORDERS, linewidth=1, zorder=4)
        ax.add_feature(cfeature.STATES.with_scale('10m'), linewidth=0.5, edgecolor='black', zorder=4)
        ax.coastlines(resolution='10m', linewidth=0.5, zorder=4)
    
        # Add high-quality water features again on top
        ocean_shp = natural_earth(resolution='10m', category='physical', name='ocean')
        ocean_geom = Reader(ocean_shp).geometries()
        ocean_feature = ShapelyFeature(ocean_geom, crs_proj, facecolor='#3b9b9b', edgecolor='none')
        ax.add_feature(ocean_feature, zorder=4)
    
        lakes_shp = natural_earth(resolution='10m', category='physical', name='lakes')
        lakes_geom = Reader(lakes_shp).geometries()
        lakes_feature = ShapelyFeature(lakes_geom, crs_proj, facecolor='#3b9b9b', edgecolor='black')
        ax.add_feature(lakes_feature, zorder=4)
    
        # Draw vertical and horizontal dividing lines for the western U.S.
        if draw_quadrant_lines:
            # Get the dataset boundaries (min_lon, max_lon, min_lat, max_lat, lon_midpoint, lat_midpoint)
            min_lon, max_lon, min_lat, max_lat, lon_midpoint, lat_midpoint = self.calculate_boundaries(ds_2022)
        
            # Vertical dividing line at the lon_midpoint (longitude)
            ax.plot([lon_midpoint, lon_midpoint], [min_lat, max_lat], color='black', linestyle='--', linewidth=6, transform=crs_proj, zorder=5)
        
            # Horizontal dividing line at the lat_midpoint (latitude)
            ax.plot([min_lon, max_lon], [lat_midpoint, lat_midpoint], color='black', linestyle='--', linewidth=6, transform=crs_proj, zorder=5)
            
     
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label('Percent Change in VegDRI (2025 vs 2022)')
    
        ax.set_title("Percent Change in VegDRI (Avg April Weeks 1–2: 2022 to 2025)")
    
        plt.tight_layout()
    
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
        return percent_change



    def compute_and_export_vegdri_stats(self, ds, csv_name):
        """
        Computes overall statistics (mean, std, min, max, median) for the VegDRI variable
        across all time and spatial coordinates (lat, lon), excluding NaN and values >= 252.
        The results are exported to a CSV file.
    
        Parameters:
        - ds (xarray.Dataset): Dataset containing VegDRI variable (time, lat, lon)
        - csv_name (str): The name of the output CSV file
        """
        if 'VegDRI' not in ds:
            raise ValueError("Dataset must contain 'VegDRI' variable.")
    
        # Apply the condition: exclude NaNs and values >= 252
        filtered_vegdri = ds['VegDRI'].where((ds['VegDRI'] <= 252) & ds['VegDRI'].notnull())
    
        # Compute the statistics across all time and spatial coordinates
        stats = {
            'mean': filtered_vegdri.mean().item(),
            'std': filtered_vegdri.std().item(),
            'min': filtered_vegdri.min().item(),
            'max': filtered_vegdri.max().item(),
            'median': filtered_vegdri.median().item()
        }
    
        # Convert stats to DataFrame
        stats_df = pd.DataFrame(stats, index=[0])
    
        # Export the stats to CSV
        stats_df.to_csv(csv_name, index=False)
        print(f"Exported stats to {csv_name}")

