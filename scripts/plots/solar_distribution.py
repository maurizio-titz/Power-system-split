import rasterio
import numpy as np
from rasterio.sample import sample_gen
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys


sys.path.append("./")

from utils.plot_style import save_figure
from utils import data_handling
from utils.config import path_to_figures_sclopf

population_data_dir = "./population_data"


def get_countries_from_pypsa_network(network):
    """
    Extract country codes from PyPSA network bus names.

    Parameters:
    network: PyPSA network object

    Returns:
    list: Unique country codes found in the network
    """
    # PyPSA-EUR typically names buses with country codes as the first 2-3 characters
    # e.g., "DE1 0", "FR1 0", "ES1 0", etc.
    bus_names = network.buses.index.tolist()

    # Extract potential country codes from bus names
    countries = set()

    for bus in bus_names:
        # Try different patterns for country code extraction
        bus_str = str(bus)

        # Pattern 1: First 2 characters (e.g., "DE", "FR", "ES")
        if len(bus_str) >= 2:
            potential_code = bus_str[:2]
            if potential_code.isalpha():
                countries.add(potential_code.upper())

        # Pattern 2: First 3 characters if they include letters (e.g., "DEU", "FRA")
        if len(bus_str) >= 3:
            potential_code = bus_str[:3]
            if potential_code.isalpha():
                countries.add(potential_code.upper())

    # Convert 2-letter codes to 3-letter ISO codes for WorldPop
    country_mapping = {
        "AT": "AUT",
        "BE": "BEL",
        "BG": "BGR",
        "CH": "CHE",
        "CZ": "CZE",
        "DE": "DEU",
        "DK": "DNK",
        "EE": "EST",
        "ES": "ESP",
        "FI": "FIN",
        "FR": "FRA",
        "GB": "GBR",
        "GR": "GRC",
        "HR": "HRV",
        "HU": "HUN",
        "IE": "IRL",
        "IT": "ITA",
        "LT": "LTU",
        "LU": "LUX",
        "LV": "LVA",
        "NL": "NLD",
        "NO": "NOR",
        "PL": "POL",
        "PT": "PRT",
        "RO": "ROU",
        "SE": "SWE",
        "SI": "SVN",
        "SK": "SVK",
        "UK": "GBR",
    }

    # Map to 3-letter codes
    iso3_countries = set()
    for country in countries:
        if len(country) == 2:
            iso3_countries.add(country_mapping.get(country, country))
        elif len(country) == 3:
            iso3_countries.add(country)
        else:
            iso3_countries.add(country)

    return sorted(list(iso3_countries))


def download_all_pypsa_countries_population_data(
    networks, year=2020, output_dir="./population_data", resolution_km=10
):
    """
    Download population data for all countries found in PyPSA networks.

    Parameters:
    networks: dict of PyPSA networks
    year: int, year of data (default 2020)
    output_dir: str, directory to save the downloaded files
    resolution_km: int, resolution in kilometers (1 or 10, default 10)

    Returns:
    list: List of successfully downloaded file paths
    """
    # Get countries from the first available network
    first_network = list(networks.values())[0]
    countries = get_countries_from_pypsa_network(first_network)

    print(f"Found countries in PyPSA network: {countries}")

    population_files = []
    successful_downloads = 0

    for country in countries:
        print(f"Downloading population data for {country}...")
        filepath = download_worldpop_data(
            country, year=year, output_dir=output_dir, resolution_km=resolution_km
        )
        if filepath:
            population_files.append(filepath)
            successful_downloads += 1

    print(
        f"Successfully downloaded population data for {successful_downloads}/{len(countries)} countries"
    )
    return population_files


def aggregate_raster_to_coarser_resolution(input_path, output_path, factor=10):
    """
    Aggregate a raster to a coarser resolution by averaging pixels.

    Parameters:
    input_path: str, path to input high-resolution raster
    output_path: str, path to save the aggregated raster
    factor: int, aggregation factor (e.g., 10 means 10x10 pixels -> 1 pixel)

    Returns:
    str: Path to aggregated file, or None if failed
    """
    try:
        import rasterio
        from rasterio.enums import Resampling

        if os.path.exists(output_path):
            print(f"Aggregated raster already exists: {output_path}")
            return output_path

        print(f"Aggregating raster from {input_path} with factor {factor}...")

        with rasterio.open(input_path) as src:
            # Calculate new dimensions
            new_height = src.height // factor
            new_width = src.width // factor

            # Read and resample data
            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=Resampling.average,
            )

            # Update transform for new resolution
            transform = src.transform * src.transform.scale(
                (src.width / new_width), (src.height / new_height)
            )

            # Update metadata
            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "transform": transform,
                    "width": new_width,
                    "height": new_height,
                }
            )

            # Write aggregated data
            with rasterio.open(output_path, "w", **kwargs) as dst:
                dst.write(data)

        print(f"Aggregated raster saved to: {output_path}")
        return output_path

    except ImportError:
        print("rasterio not available for aggregation")
        return None
    except Exception as e:
        print(f"Error aggregating raster: {e}")
        return None


def merge_population_rasters(
    population_files,
    output_path="population_data/europe_combined_2020.tif",
    resolution_km=10,
):
    """
    Merge multiple population raster files into a single European raster.

    Parameters:
    population_files: list of paths to individual country population rasters
    output_path: path where the merged raster should be saved
    resolution_km: int, resolution in kilometers

    Returns:
    str: path to the merged raster file, or None if failed
    """
    # Update output path to include resolution
    base_name, ext = os.path.splitext(output_path)
    output_path = f"{base_name}_{resolution_km}km{ext}"

    # Check if merged file already exists
    if os.path.exists(output_path):
        print(f"Merged population raster already exists: {output_path}")
        return output_path

    try:
        from rasterio.merge import merge
        from rasterio.plot import show
        import numpy as np

        if not population_files:
            raise ValueError("No population files to merge")

        # Open all raster files
        raster_datasets = []
        for file_path in population_files:
            try:
                if os.path.exists(file_path):
                    raster_datasets.append(rasterio.open(file_path))
            except Exception as e:
                print(f"Warning: Could not open {file_path}: {e}")
                continue

        if not raster_datasets:
            raise ValueError("No valid raster files found")

        print(f"Merging {len(raster_datasets)} population raster files...")

        # Merge all rasters
        merged_array, merged_transform = merge(raster_datasets)

        # Get metadata from the first raster
        merged_meta = raster_datasets[0].meta.copy()
        merged_meta.update(
            {
                "height": merged_array.shape[1],
                "width": merged_array.shape[2],
                "transform": merged_transform,
            }
        )

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save merged raster
        with rasterio.open(output_path, "w", **merged_meta) as dest:
            dest.write(merged_array)

        # Close all datasets
        for dataset in raster_datasets:
            dataset.close()

        print(f"Merged population raster saved to: {output_path}")
        return output_path

    except ImportError:
        print("rasterio.merge not available. Using individual files instead.")
        return None
    except ValueError as e:
        print(f"Error merging raster files: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error merging raster files: {e}")
        return None


def get_population_density_from_raster(coordinates_df, raster_path=None):
    """
    Get population density for coordinates from a raster dataset.

    Parameters:
    coordinates_df: DataFrame with 'longitude' and 'latitude' columns
    raster_path: Path to population density raster file

    Returns:
    list: Population density values for each coordinate
    """

    # Check if raster path is provided
    if raster_path is None:
        print("You need to download a population density raster dataset first")
        print("Options:")
        print("1. WorldPop: https://www.worldpop.org/")
        print("2. GPW: https://sedac.ciesin.columbia.edu/data/collection/gpw-v4")
        print("3. LandScan: https://landscan.ornl.gov/")
        return None

    # Check if raster file exists
    try:
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"Raster file not found: {raster_path}")
    except (OSError, TypeError) as e:
        print(f"Error accessing raster file: {e}")
        return None

    try:
        # Open the raster dataset
        with rasterio.open(raster_path) as dataset:
            print(f"Opened raster: {raster_path}")
            print(f"Raster CRS: {dataset.crs}")
            print(f"Raster bounds: {dataset.bounds}")

            # Prepare coordinates for sampling
            coords = [
                (lon, lat)
                for lon, lat in zip(
                    coordinates_df["longitude"], coordinates_df["latitude"]
                )
            ]

            # Sample population density values at coordinates
            pop_density_values = list(sample_gen(dataset, coords))

            # Extract the values (sample_gen returns generator of arrays)
            pop_densities = []
            for val in pop_density_values:
                if len(val) > 0 and not np.isnan(val[0]) and val[0] >= 0:
                    pop_densities.append(val[0])
                else:
                    pop_densities.append(0)  # Set to 0 for invalid/missing data

            print(f"Successfully sampled {len(pop_densities)} points")
            print(
                f"Population density range: {min(pop_densities):.2f} - {max(pop_densities):.2f}"
            )

            return pop_densities

    except rasterio.RasterioIOError as e:
        print(f"Error reading raster file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing raster file: {e}")
        return None


def download_worldpop_data(
    country_iso3, year=2020, output_dir=population_data_dir, resolution_km=10
):
    """
    Download WorldPop population density data for a specific country.

    Parameters:
    country_iso3: str, 3-letter ISO country code (e.g., 'DEU' for Germany)
    year: int, year of data (default 2020)
    output_dir: str, directory to save the downloaded file
    resolution_km: int, resolution in kilometers (1 or 10, default 10)

    Returns:
    str: Path to the downloaded file, or None if failed
    """
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir, f"{country_iso3.lower()}_population_{year}_{resolution_km}km.tif"
    )

    # Check if file already exists
    if os.path.exists(output_path):
        print(f"Population data for {country_iso3} already exists: {output_path}")
        return output_path

    # WorldPop URL pattern - different for 1km vs 10km resolution
    if resolution_km == 1:
        # 1km resolution URL (original)
        url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/{year}/{country_iso3.upper()}/{country_iso3.lower()}_ppp_{year}_1km_Aggregated_UNadj.tif"

        try:
            print(
                f"Downloading population data for {country_iso3} ({year}) at {resolution_km}km resolution..."
            )
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded: {output_path}")
            return output_path

        except requests.RequestException as e:
            print(f"Error downloading WorldPop data for {country_iso3}: {e}")
        except OSError as e:
            print(f"Error saving file for {country_iso3}: {e}")

    elif resolution_km == 10:
        # 10km resolution URL - try multiple potential patterns
        url_patterns = [
            f"https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/{year}/maxar_v1/{country_iso3.upper()}/{country_iso3.lower()}_ppp_{year}_UNadj.tif",
            f"https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{country_iso3.upper()}/{country_iso3.lower()}_ppp_{year}.tif",
            f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/{year}/{country_iso3.upper()}/{country_iso3.lower()}_ppp_{year}_1km.tif",
        ]

        # Try multiple URL patterns for 10km resolution
        for i, url in enumerate(url_patterns):
            try:
                print(
                    f"Downloading population data for {country_iso3} ({year}) at {resolution_km}km resolution (attempt {i+1})..."
                )
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"Downloaded: {output_path}")
                return output_path

            except requests.RequestException as e:
                print(f"Attempt {i+1} failed for {country_iso3}: {e}")
                if i < len(url_patterns) - 1:
                    continue
                else:
                    print(f"All download attempts failed for {country_iso3}")
                    break
            except OSError as e:
                print(f"Error saving file for {country_iso3}: {e}")
                break
    else:
        raise ValueError(f"Unsupported resolution: {resolution_km}km. Supported: 1, 10")

    return None


# Modified version of your existing code with population density analysis
def combine_solar_and_population_density_data(
    selected_co2ls, networks, raster_path, resolution_km=10
):
    """
    Analyze the relationship between solar capacity and population density.

    Parameters:
    selected_co2ls: list of CO2 levels to analyze
    networks: dict of PyPSA networks
    raster_path: path to population density raster file
    resolution_km: int, resolution in kilometers
    """

    # Create figure with subplots for each CO2 level
    f, axs = plt.subplots(
        len(selected_co2ls), 1, figsize=(12, 4 * len(selected_co2ls)), sharex=True
    )
    if len(selected_co2ls) == 1:
        axs = [axs]

    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(selected_co2ls)))

    all_data = []  # Store all data for correlation analysis

    for i, co2l in enumerate(selected_co2ls):
        network = networks[co2l]

        # Get bus coordinates
        bus_coords = network.buses[["x", "y"]].copy()
        bus_coords.columns = ["longitude", "latitude"]

        # Get solar generators
        solar_gens = network.generators[
            network.generators.carrier.str.contains("solar", case=False)
        ]
        solar_capacity_by_bus = solar_gens.groupby("bus")["p_nom"].sum()

        # Merge with coordinates
        solar_data = bus_coords.merge(
            solar_capacity_by_bus, left_index=True, right_index=True, how="inner"
        )

        # Get population densities
        print(f"Processing CO2 level {co2l}...")
        pop_densities = get_population_density_from_raster(solar_data, raster_path)

        try:
            if pop_densities is None:
                raise ValueError(f"Failed to get population data for CO2 level {co2l}")

            solar_data["population_density"] = pop_densities

            # Remove zero population density points for log scale
            solar_data_filtered = solar_data[
                solar_data["population_density"] > 0
            ].copy()

            if len(solar_data_filtered) == 0:
                raise ValueError(f"No valid data points for CO2 level {co2l}")

            # Create 2D histogram: Population density vs Solar capacity
            hb = axs[i].hexbin(
                solar_data_filtered["population_density"],
                solar_data_filtered["p_nom"] / 1000,
                gridsize=30,
                cmap="YlOrRd",
                mincnt=1,
                xscale="log",
            )

            axs[i].set_xlabel("Population Density [people/km2]")
            axs[i].set_ylabel("Installed Solar Capacity [GW]")
            axs[i].set_title(rf"CO$_2$ {int(co2l*100)}\%")
            axs[i].grid(True, alpha=0.3)

            # Add colorbar
            plt.colorbar(hb, ax=axs[i], label="Count")

            # Calculate correlation
            correlation = solar_data_filtered["p_nom"].corr(
                solar_data_filtered["population_density"]
            )
            axs[i].text(
                0.02,
                0.98,
                f"Correlation: {correlation:.3f}",
                transform=axs[i].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Store data for overall analysis
            solar_data_filtered["co2l"] = co2l
            # Ensure bus names are preserved - add bus name as a column too for easier access
            # solar_data_filtered["bus_name"] = solar_data_filtered.index
            all_data.append(solar_data_filtered)

            print(
                f"CO2 {co2l}: {len(solar_data_filtered)} points, correlation: {correlation:.3f}"
            )

        except ValueError as e:
            print(f"Warning: {e}")
            continue

    plt.suptitle(r"Solar Capacity vs Population Density by CO$_2$ Level")
    plt.tight_layout()
    plt.show()

    # Combined analysis
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=False)
        # Round all float columns to 1 decimal
        float_cols = combined_data.select_dtypes(include=["float"]).columns
        combined_data[float_cols] = combined_data[float_cols].round(1)

        # Verify bus names are preserved
        print(f"Combined data created with {len(combined_data)} rows")
        print(f"Index (bus names) sample: {list(combined_data.index[:5])}")
        print(f"Columns: {list(combined_data.columns)}")

        # Overall correlation analysis
        overall_correlation = combined_data["p_nom"].corr(
            combined_data["population_density"]
        )
        print(f"\nOverall correlation: {overall_correlation:.3f}")

        # Create scatter plot with regression line
        plt.figure(figsize=(10, 6))
        colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(selected_co2ls)))
        for i, co2l in enumerate(selected_co2ls):
            data_subset = combined_data[combined_data["co2l"] == co2l]
            if len(data_subset) > 0:
                plt.scatter(
                    data_subset["population_density"],
                    data_subset["p_nom"] / 1000,
                    alpha=0.6,
                    color=colors[i],
                    label=rf"CO$_2$ {int(co2l*100)}\%",
                )

        plt.xscale("log")
        plt.xlabel("Population Density [people/km²]")
        plt.ylabel("Solar Capacity [GW]")
        plt.title("Solar Capacity vs Population Density - Combined Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        combined_data.to_csv(
            population_data_dir
            + f"solar_capacity_vs_population_density_{resolution_km}km.csv",
        )
        print(
            f"Combined data saved to: {population_data_dir}solar_capacity_vs_population_density_{resolution_km}km.csv"
        )
        return combined_data

    return None


# Usage example with your existing code:
def get_solar_and_population_data(networks, selected_co2ls=None, resolution_km=10):
    """
    Main function to run the solar vs population density analysis.

    Parameters:
    selected_co2ls: list of CO2 levels to analyze (if None, will use default)
    networks: dict of PyPSA networks (if None, function will just download data)
    resolution_km: int, resolution in kilometers (1 or 10, default 10)
    """

    data_file_path = (
        population_data_dir
        + f"solar_capacity_vs_population_density_{resolution_km}km.csv"
    )
    if os.path.exists(data_file_path):
        print(f"Loading existing data from {data_file_path}")
        return pd.read_csv(data_file_path)

    # Default CO2 levels if not provided
    if selected_co2ls is None:
        selected_co2ls = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        print("Using default CO2 levels:", selected_co2ls)

    if networks is not None:
        # Extract countries from the PyPSA network and download population data
        print("Extracting countries from PyPSA network...")
        population_files = download_all_pypsa_countries_population_data(
            networks, year=2020, resolution_km=resolution_km
        )

        # Try to merge all country rasters into a single European raster
        merged_raster_path = merge_population_rasters(
            population_files, resolution_km=resolution_km
        )

        if merged_raster_path and os.path.exists(merged_raster_path):
            raster_path = merged_raster_path
        else:
            # Fall back to using individual country files if merging fails
            print("Using first available population raster file...")
            if population_files:
                raster_path = population_files[0]
            else:
                print("No population data files available!")
                return None

    else:
        # Fallback: download data for common European countries
        print(
            "Networks not provided. Downloading data for common European countries..."
        )
        european_countries = [
            "AUT",
            "BEL",
            "BGR",
            "CHE",
            "CZE",
            "DEU",
            "DNK",
            "EST",
            "ESP",
            "FIN",
            "FRA",
            "GBR",
            "GRC",
            "HRV",
            "HUN",
            "IRL",
            "ITA",
            "LTU",
            "LUX",
            "LVA",
            "NLD",
            "NOR",
            "POL",
            "PRT",
            "ROU",
            "SWE",
            "SVN",
            "SVK",
        ]

        population_files = []
        for country in european_countries:
            filepath = download_worldpop_data(
                country, year=2020, resolution_km=resolution_km
            )
            if filepath:
                population_files.append(filepath)

        # Try to merge all country rasters
        merged_raster_path = merge_population_rasters(
            population_files, resolution_km=resolution_km
        )

        if merged_raster_path:
            raster_path = merged_raster_path
        elif population_files:
            raster_path = population_files[0]
        else:
            raster_path = (
                f"population_data/europe_population_2020_{resolution_km}km.tif"
            )

    print(f"Using population raster: {raster_path}")

    try:
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"Population raster file not found: {raster_path}")

        if networks is None:
            raise ValueError(
                "Networks data not provided. Please load PyPSA networks first."
            )

        # Run the analysis
        combined_data = combine_solar_and_population_density_data(
            selected_co2ls, networks, raster_path, resolution_km
        )

        if combined_data is not None:
            # Additional analysis
            print("\nSummary Statistics:")
            print(
                combined_data.groupby("co2l")[
                    ["p_nom", "population_density"]
                ].describe()
            )
            return combined_data

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the download was successful or update the file path")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error during analysis: {e}")

    return None


# Integration with your existing plotting code:
def create_population_density_plots(combined_data, selected_co2ls):
    """
    Create population density plots using pre-processed combined data.

    Parameters:
    combined_data: DataFrame with columns 'co2l', 'population_density', 'p_nom'
    selected_co2ls: list of CO2 levels to analyze
    """
    if combined_data is None or len(combined_data) == 0:
        print("No combined data provided for plotting")
        return

    f, axs = plt.subplots(1, len(selected_co2ls), figsize=(15, 5), sharey=True)

    # Handle case where there's only one subplot
    if len(selected_co2ls) == 1:
        axs = [axs]

    for i, co2l in enumerate(selected_co2ls):
        # Filter data for this CO2 level
        data_subset = combined_data[combined_data["co2l"] == co2l]

        try:
            if len(data_subset) == 0:
                raise ValueError(f"No data available for CO2 level {co2l}")

            # Filter out zero values for log scale
            solar_data_filtered = data_subset[data_subset["population_density"] > 0]

            if len(solar_data_filtered) == 0:
                raise ValueError(
                    f"No valid data points with population density > 0 for CO2 level {co2l}"
                )

            # Create 2D histogram: Population density vs Solar capacity
            # cmap = "cividis"
            # hb = axs[i].hexbin(
            #     solar_data_filtered["population_density"],
            #     solar_data_filtered["p_nom"] / 1000,
            #     gridsize=25,
            #     cmap=cmap,
            #     mincnt=1,
            #     xscale="log",
            # )
            axs[i].scatter(
                solar_data_filtered["population_density"],
                solar_data_filtered["p_nom"] / 1000,
                color="black",
                alpha=0.5,
                s=10,
            )

            axs[i].set_xlabel("Population Density [people/km²]")
            axs[i].set_title(rf"CO$_2$ {int(co2l*100)}\%")
            axs[i].grid(True, alpha=0.3)

            # Add colorbar
            # plt.colorbar(hb, ax=axs[i], label="Count")

        except ValueError as e:
            print(f"Warning: {e}")
            # Create empty plot
            axs[i].set_xlabel("Population Density [people/km²]")
            axs[i].set_title(rf"CO$_2$ {int(co2l*100)}\% - No Data")
            axs[i].text(
                0.5,
                0.5,
                "No valid data",
                transform=axs[i].transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )

    # Set y-label only for leftmost subplot
    axs[0].set_ylabel("Installed Solar Capacity [GW]")

    plt.suptitle(r"Solar Capacity vs Population Density by CO$_2$ Level")
    plt.tight_layout()
    save_figure(f, path_to_figures_sclopf, "solar_capacity_vs_population_density")


def analyze_solar_population_for_pypsa_networks(
    networks, selected_co2ls=None, resolution_km=10
):
    """
    Complete analysis function that performs data analysis and creates plots.

    Parameters:
    networks: dict of PyPSA networks
    selected_co2ls: list of CO2 levels to analyze (if None, uses all available)
    resolution_km: int, resolution in kilometers (default 10)

    Returns:
    combined_data: DataFrame with all analysis results
    """
    if selected_co2ls is None:
        selected_co2ls = list(networks.keys())
        print(f"Using all available CO2 levels: {selected_co2ls}")

    # Get the combined data from analysis
    combined_data = get_solar_and_population_data(
        networks, selected_co2ls, resolution_km
    )

    if combined_data is not None:
        # Create the plots using the combined data
        create_population_density_plots(combined_data, selected_co2ls)

        print("\nAnalysis complete! Combined data and plots have been generated.")
        return combined_data
    else:
        print("Analysis failed - no data available for plotting.")
        return None


# Run the main function if this script is executed directly
if __name__ == "__main__":
    n_nodes = 600
    resolution_km = 10  # Use 10km resolution for coarser population data
    co2ls = data_handling.get_co2_levels(n_nodes)
    data_file_path = (
        population_data_dir
        + f"solar_capacity_vs_population_density_{resolution_km}km.csv"
    )

    if not os.path.exists(data_file_path):
        networks = {
            co2l: data_handling.load_pypsa_network(
                n_nodes=600, co2lvl=co2l, use_sclopf=True
            )
            for co2l in co2ls
        }
    else:
        networks = None

    selected_co2ls = [0.6, 0.2, 0.0]
    analyze_solar_population_for_pypsa_networks(
        networks, selected_co2ls=selected_co2ls, resolution_km=resolution_km
    )
