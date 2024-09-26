import os
import re
import csv
from collections import OrderedDict


log_dir = os.getcwd()  # Current working directory
summary_data = OrderedDict()  # Dict to store the extracted data

# Iterate over all files in the directory
for file_name in sorted(os.listdir(log_dir)):
    if file_name.endswith(".log"):
        file_path = os.path.join(log_dir, file_name)
        with open(file_path, 'r') as file:
            content = file.readlines()

            # Reverse iterate to find the 'val_loss_step' as the end of the summary
            end_index = None
            for i in range(len(content) - 1, -1, -1):
                if 'val_loss_step' in content[i]:
                    end_index = i
                    break
            # If 'val_loss_step' is not found, look for 'val_final_loss_step'
            if end_index is None:
                for i in range(len(content) - 1, -1, -1):
                    if 'val_final_loss_step' in content[i]:
                        end_index = i
                        break
            
            # Find the 'Run summary:' as the start of the summary
            for i in range(end_index, -1, -1):
                if 'Run summary:' in content[i]:
                    start_index = i + 1
                    break
            
            # Extract the summary values
            summary = OrderedDict()
            pattern = r"[-+]?\d*\.\d+|\d+"
            for line in content[start_index: end_index+1]:
                line = line.strip()
                if line:
                    # Extract last number in the line
                    numbers = re.findall(pattern, line)
                    if numbers:
                        value = numbers[-1]
                        row_name = line[:line.rfind(value)].replace('wandb: ', '').strip() if value else line.replace('wandb: ', '').strip()
                        summary[row_name] = value
            
            # Add extracted summary to the dictionary under the file name
            summary_data[file_name] = summary

# Get the union of all keys to ensure all rows are included
all_keys = OrderedDict()
for summary in summary_data.values():
    for key in summary:
        if key not in all_keys:
            all_keys[key] = None

# Define the CSV file name
output_csv = os.path.join(log_dir, "run_summary_results.csv")

# Write results to a CSV file on the same working directory as the .log files
with open(output_csv, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Metric"] + list(summary_data.keys()))  # Header

    # Write the data
    for key in all_keys:
        row = [key]
        for file_name in summary_data.keys():
            row.append(summary_data[file_name].get(key, ''))
        writer.writerow(row)

print(f"Results saved to {output_csv}")

# run the following in the targeted log directory:
# python /Users/macuser/Desktop/Imperial/70078_MSc_AI_Individual_Project/code/cxr-fmkd/utils/logs_utils/extract_run_summary.py
