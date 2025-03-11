import os
import csv
import random
import argparse

def generate_file_data_csv(directory, output_file="output.csv", file_extension=".png", 
                           visibility_range=(1, 3), x_range=(100, 800), y_range=(100, 600)):
    """
    Scans a directory for files with a specified extension and generates a CSV file
    with random values for visibility, coordinates, and status.
    
    Args:
        directory (str): Path to the directory containing the files
        output_file (str): Name of the output CSV file
        file_extension (str): File extension to filter by
        visibility_range (tuple): Min and max values for visibility
        x_range (tuple): Min and max values for x-coordinate
        y_range (tuple): Min and max values for y-coordinate
    
    Returns:
        str: Path to the created CSV file
    """
    # Check if directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"Directory '{directory}' does not exist")
    
    # Get all files with the specified extension
    files = [f for f in os.listdir(directory) if f.endswith(file_extension)]
    
    # Sort files to ensure consistent ordering
    files.sort()
    
    # Create the output file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['file name', 'visibility', 'x-coordinate', 'y-coordinate', 'status'])
        
        # Write data for each file
        for file in files:
            visibility = random.randint(visibility_range[0], visibility_range[1])
            x_coord = random.randint(x_range[0], x_range[1])
            y_coord = random.randint(y_range[0], y_range[1])
            status = random.randint(0, 1)
            
            writer.writerow([file, visibility, x_coord, y_coord, status])
    
    print(f"CSV file created: {output_file}")
    return output_file

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate CSV data for files in a directory')
    parser.add_argument('directory', help='Directory containing the files')
    parser.add_argument('-o', '--output', default='output.csv', help='Output CSV file name')
    parser.add_argument('-e', '--extension', default='.png', help='File extension to filter by')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate CSV file
    generate_file_data_csv(args.directory, args.output, args.extension)

if __name__ == "__main__":
    main()
