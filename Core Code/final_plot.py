import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def load_results(base_path='./Stage2_Final'):
    """
    Load results by traversing subdirectories under the base path.
    Extracts dataset year, color mode, brightness, Huber loss, RMSE, and directory path from directory names and log files.
    
    Args:
        base_path (str): Path to the base directory containing experiment subdirectories.
    
    Returns:
        dict: Nested dictionary with structure:
              results[dataset_year][color_mode][huber_loss][brightness] = {'rmse': float, 'path': str}
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for entry in os.listdir(base_path):
        subdir = os.path.join(base_path, entry)

        if os.path.isdir(subdir):
            dir_name = entry

            # Remove 'stage_two_' prefix if present
            if dir_name.startswith('stage_two_'):
                dir_name = dir_name[len('stage_two_'):]

            parts = dir_name.split('_')

            # Initialize variables
            dataset_year = None
            color_mode = None
            brightness = None
            huber_loss = None

            i = 0
            while i < len(parts):
                part = parts[i]
                if part in ['2013', '2014']:
                    dataset_year = part
                elif part in ['Red', 'Green', 'RedGreen']:
                    color_mode = part.lower()  # Convert to lowercase
                elif part == 'bright':
                    # Expect the next part to be the brightness value
                    if i + 1 < len(parts):
                        bright_val = parts[i + 1]
                        brightness = f'bright_{bright_val}'
                        i += 1  # Skip the next part as it's already processed
                    else:
                        print(f"Warning: 'bright' specified but no value found in '{entry}'. Skipping.")
                        brightness = None
                        break
                elif part == 'original':
                    brightness = 'original'
                elif part == 'huber':
                    # Expect the next part to be the huber loss value
                    if i + 1 < len(parts):
                        huber_val = parts[i + 1]
                        try:
                            huber_loss = float(huber_val)
                        except ValueError:
                            huber_loss = None
                        i += 1  # Skip the next part as it's already processed
                    else:
                        print(f"Warning: 'huber' specified but no value found in '{entry}'. Skipping.")
                        huber_loss = None
                        break
                i += 1

            # Check if all necessary information was extracted
            if not all([dataset_year, color_mode, brightness, huber_loss is not None]):
                print(f"Warning: Could not parse directory name '{entry}'. Skipping.")
                continue

            # Find the .log file in the subdirectory
            log_files = [f for f in os.listdir(subdir) if f.endswith('.log')]
            if not log_files:
                print(f"Warning: No .log file found in '{entry}'. Skipping.")
                continue

            log_file_path = os.path.join(subdir, log_files[0])

            try:
                with open(log_file_path, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        print(f"Warning: Empty .log file in '{entry}'. Skipping.")
                        continue
                    last_line = lines[-1].strip()
                    # Extract RMSE using regex
                    match = re.search(r'Best RMSE:\s*([0-9.]+)', last_line)
                    if match:
                        rmse = float(match.group(1))
                    else:
                        print(f"Warning: Could not parse RMSE from last line in '{entry}'. Skipping.")
                        continue
            except Exception as e:
                print(f"Error reading log file in '{entry}': {e}. Skipping.")
                continue

            # Store the extracted RMSE and path in the results dictionary
            results[dataset_year][color_mode][huber_loss][brightness] = {
                'rmse': rmse,
                'path': subdir
            }

    return results

def process_best_performances(results):
    """
    Process the results to count the number of times each brightness parameter achieves the best RMSE
    for each dataset year across all color modes and Huber deltas. Also, identify the overall best
    parameter combination per dataset year.
    
    Args:
        results (dict): Nested dictionary containing RMSE values and paths.
    
    Returns:
        dict: Dictionary with structure:
              best_counts[dataset_year][brightness] = count
        dict: Dictionary with structure:
              best_combinations[dataset_year] = {'brightness': str, 'huber_loss': float, 'rmse': float, 'path': str}
    """
    best_counts = defaultdict(lambda: defaultdict(int))
    best_combinations = {}

    for dataset_year, color_modes in results.items():
        for color_mode, huber_losses in color_modes.items():
            for huber_loss, brightness_dict in huber_losses.items():
                # Find the brightness parameter with the lowest RMSE for this combination
                best_brightness = None
                best_rmse = float('inf')
                best_path = ''

                for brightness, data in brightness_dict.items():
                    rmse = data['rmse']
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_brightness = brightness
                        best_path = data['path']
                
                if best_brightness:
                    best_counts[dataset_year][best_brightness] += 1

                    # Update overall best combination for the dataset year
                    if dataset_year not in best_combinations or best_rmse < best_combinations[dataset_year]['rmse']:
                        best_combinations[dataset_year] = {
                            'brightness': best_brightness,
                            'huber_loss': huber_loss,
                            'rmse': best_rmse,
                            'path': best_path
                        }

    return best_counts, best_combinations

def set_common_style():
    """Set common style parameters for all plots"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.grid': True,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'figure.dpi': 300,
    })

def plot_best_performances(best_counts, dataset_years=['2013', '2014']):
    """Plot the number of times each brightness parameter achieves the best RMSE"""
    set_common_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Professional color scheme
    colors = {
        '2013': '#2c5985',  # Deeper blue
        '2014': '#8b2635'   # Deeper red
    }

    # Setup data
    all_brightness = ['bright_0.4', 'bright_0.5', 'bright_0.6', 'bright_0.7', 'original']
    all_brightness = [b for b in all_brightness if any(b in best_counts[year] for year in dataset_years)]
    x = np.arange(len(all_brightness))
    width = 0.35

    # Create bars
    counts_2013 = [best_counts['2013'].get(brightness, 0) for brightness in all_brightness]
    counts_2014 = [best_counts['2014'].get(brightness, 0) for brightness in all_brightness]

    rects1 = ax.bar(x - width/2, counts_2013, width, label='2013', 
                    color=colors['2013'], edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, counts_2014, width, label='2014', 
                    color=colors['2014'], edgecolor='black', linewidth=1)

    # Customize plot
    ax.set_xlabel('Brightness Parameters', fontsize=16)
    ax.set_ylabel('Number of Best Performances', fontsize=16)
    ax.set_xticks(x)
    xtick_labels = [b.replace('bright_', '') if b != 'original' else 'origin' for b in all_brightness]
    ax.set_xticklabels(xtick_labels, rotation=45)
    
    # Add value labels above bars
    def autolabel(rects, offset=0):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, offset),  # Vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=12)

    autolabel(rects1, 8)  # Increased offset
    autolabel(rects2, 8)

    # Customize grid and legend
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('best_performances_comparison.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_rmse_trend(results, dataset_years=['2013', '2014']):
   """Plot the trend of average RMSE"""
   set_common_style()
   
   fig, ax = plt.subplots(figsize=(12, 8))
   
   # Nature style colors
   colors = {
       '2013': '#2b5a89',  # Nature blue
       '2014': '#a63603'   # Nature red
   }

   # Prepare data
   light_params = ['0.4', '0.5', '0.6', '0.7', 'origin']
   brightness_keys = ['bright_0.4', 'bright_0.5', 'bright_0.6', 'bright_0.7', 'original']
   avg_rmse = {year: [] for year in dataset_years}

   # Calculate average RMSE
   for year in dataset_years:
       for brightness in brightness_keys:
           rmse_list = []
           color_modes = results.get(year, {})
           for color_mode in color_modes:
               huber_losses = color_modes[color_mode]
               for huber_loss in huber_losses:
                   brightness_dict = huber_losses[huber_loss]
                   if brightness in brightness_dict:
                       rmse_list.append(brightness_dict[brightness]['rmse'])
           avg_rmse[year].append(np.mean(rmse_list) if rmse_list else np.nan)

   # Adjust plot margins to prevent overlapping
   plt.subplots_adjust(bottom=0.15, left=0.12)

   # Plot lines with refined style
   for year in dataset_years:
       line = ax.plot(light_params, avg_rmse[year],
                     marker='o', label=year,
                     color=colors[year],
                     linewidth=2.5, markersize=8,
                     markeredgecolor='white',
                     markeredgewidth=2)
       
       # Optimize label positions to avoid overlapping
       for i, rmse in enumerate(avg_rmse[year]):
           if not np.isnan(rmse):
               # Adjust vertical offset to avoid overlapping with lines and other labels
               if year == '2013':
                   y_offset = 25 if i == 0 else 15  # Extra offset for first point
               else:
                   y_offset = -25 if i == len(light_params)-1 else -15  # Extra offset for last point

               ax.annotate(f"{rmse:.2f}",
                         xy=(i, rmse),
                         xytext=(0, y_offset),
                         textcoords="offset points",
                         ha='center',
                         va='bottom' if y_offset > 0 else 'top',
                         color=colors[year],
                         fontsize=12,
                         fontweight='normal')

   # Customize axes and grid
   ax.set_xlabel('Lighting Parameter', fontsize=16)
   ax.set_ylabel('Average RMSE', fontsize=16)
   ax.set_xticks(range(len(light_params)))
   ax.set_xticklabels(light_params, rotation=0)
   
   # Adjust axis range to leave space for labels
   y_min, y_max = ax.get_ylim()
   ax.set_ylim(y_min - 0.5, y_max + 0.5)
   
   # Refined grid
   ax.yaxis.grid(True, linestyle='--', alpha=0.3)
   ax.set_axisbelow(True)

   # Customize legend with increased spacing from the border
   ax.legend(loc='upper right', frameon=True, framealpha=0.9, 
            bbox_to_anchor=(0.98, 0.98))

   # Adjust layout and save with high resolution
   plt.tight_layout()
   plt.savefig('performance_trend.png', dpi=600, bbox_inches='tight', pad_inches=0.2)
   plt.close()

def print_best_combinations(best_combinations):
    """
    Print the best parameter combinations for each dataset year, including the path to the model weights.
    
    Args:
        best_combinations (dict): Dictionary with best combinations.
    """
    for dataset_year, combo in best_combinations.items():
        brightness = combo['brightness']
        huber_loss = combo['huber_loss']
        rmse = combo['rmse']
        path = combo['path']
        print(f"Dataset {dataset_year} - Best Combination: "
              f"Brightness={brightness.replace('bright_', '') if brightness != 'original' else 'origin'}, "
              f"Huber δ={huber_loss}, RMSE={rmse}, Path='{path}'")

def main():
    """
    Main function to load results, process best performances, plot statistics, and print best combinations.
    """
    base_path = './Stage2_Final'

    if not os.path.exists(base_path):
        print(f"Error: Base path '{base_path}' does not exist.")
        return

    # Load all results from the directory structure
    results = load_results(base_path)

    if not results:
        print("No valid results found. Please check the directory structure and log files.")
        return

    # Process best performances
    best_counts, best_combinations = process_best_performances(results)

    # Plot and save the best performances
    plot_best_performances(best_counts, dataset_years=['2013', '2014'])

    # Print the best parameter combinations
    print_best_combinations(best_combinations)
    print("Plots have been generated and saved.")

    # Plot and save the RMSE trend with lighting parameters
    plot_rmse_trend(results, dataset_years=['2013', '2014'])

if __name__ == '__main__':
    main()