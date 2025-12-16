# Individual Power System Split Plotting Scripts

This directory contains individual Python scripts for generating each specific plot from the power system split analysis.

## Available Scripts

### 1. `plot_generation_map.py`
**Figure 2: Generation Map**
- Creates maps showing generation mix by carrier type and month
- Plots total annual generation versus CO2 levels
- Function: `create_generation_map()`

### 2. `plot_storage_map.py`
**Storage Capacity Analysis**
- Maps showing battery and H2 storage capacity distribution
- Line plot of total storage capacity versus CO2 levels
- Function: `create_storage_map()`

### 3. `plot_spi_analysis.py`
**Figure 3: SPI (System Polarization Index)**
- Histograms of power flow distance and rotational energy
- SPI vector plots across months and CO2 levels
- Function: `create_spi_plot()`

### 4. `plot_line_failure_probs.py`
**Figure 7: Line Failure Probabilities**
- Maps showing primary and secondary line failure probabilities
- Both logarithmic and linear colorscale versions
- Functions: `create_line_failure_plot()`, `create_line_failure_plot_linear()`

### 5. `plot_nodal_blackout_probs.py`
**Nodal Blackout Probabilities**
- Maps showing blackout probabilities at each network node
- Visualization across different CO2 levels
- Function: `create_nodal_blackout_plot()`

### 6. `plot_split_statistics.py`
**Figure 5: Split Statistics**
- Histograms of power imbalance and rotational energy in split components
- Distribution of loss of load share across CO2 levels
- Function: `create_split_statistics_plot()`

### 7. `plot_inertia_mitigation.py`
**Inertia Mitigation Analysis**
- Shows inertia requirements for different mitigation scenarios
- Maps of synthetic inertia placement
- Function: `create_inertia_mitigation_plot()`

### 8. `plot_line_extension_mitigation.py`
**Line Extension Mitigation**
- Cost analysis for grid reinforcement
- Maps showing which lines to reinforce for optimal cost-benefit
- Function: `create_line_extension_mitigation_plot()`

## Usage

### Running Individual Scripts

Each script can be run independently:

```bash
cd /srv/data/mtitz/power-system-split
python scripts/plot_generation_map.py
python scripts/plot_storage_map.py
python scripts/plot_spi_analysis.py
# ... etc
```

### Running Multiple Scripts

Use the main runner script to execute all or specific plots:

```bash
# Run all plots
python scripts/run_all_plots.py --plot all

# Run specific plot
python scripts/run_all_plots.py --plot generation
python scripts/run_all_plots.py --plot storage
python scripts/run_all_plots.py --plot spi
python scripts/run_all_plots.py --plot line_failures
python scripts/run_all_plots.py --plot nodal_blackout
python scripts/run_all_plots.py --plot split_stats
python scripts/run_all_plots.py --plot inertia_mitigation
python scripts/run_all_plots.py --plot line_extension
```

## Dependencies

All scripts require the same dependencies as the original notebook:
- numpy, pandas, matplotlib, networkx, pypsa
- cartopy (for map projections)
- Custom utility modules from the `utils/` directory

## Output

- All plots are saved as PDF files in the directory specified by `path_to_figures_sclopf`
- Each script contains configuration options that can be modified at the top of the script
- Scripts include both display (`plt.show()`) and save (`plt.savefig()`) functionality

## Notes

- Scripts maintain the same matplotlib styling and LaTeX formatting as the original
- Each script is self-contained and includes necessary imports and setup
- Configuration parameters (like CO2 levels, figure sizes, etc.) can be easily modified within each script
- The modular structure makes it easier to debug, modify, or extend individual plots

## Customization

To customize plots:
1. Edit the relevant script directly
2. Modify parameters at the top of each script (CO2 levels, figure sizes, colors, etc.)
3. Add or remove subplot panels as needed
4. Adjust styling, labels, and formatting within each function

This modular approach allows for better maintenance, easier debugging, and more flexible usage of the plotting functionality.
