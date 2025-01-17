# DischargeSpctrumModelling
Computational Modelling for Bsc Project

## Tools and Utilities

### 1. File Conversion Tool

The `file_conversion` tool, located in the `utils` folder, is a small utility designed to convert `.b16` or `.pcoraw` files into formats like `.tif` or `BMP` using the `pco.fileconversion` command-line tool. This tool is particularly useful for processing raw camera data generated during experiments.

#### Prerequisites for File Conversion

1. **PCO File Conversion Tool**  
   Download and install the `pco.fileconversion` command-line tool.  
   You can download it from the official website here:  
   [Download PCO File Conversion Tool](https://excelitas.com/de/product/pco-add-on-software)

   It is recommended to use the default installation path:  
   `C:\Program Files\PCO Digital Camera Toolbox\pco.fileconversion`

#### Setting Up for File Conversion

1. **Prepare the Data Folder**  
   Create a `data` folder and place your `.b16` files (or other compatible formats) into the `data` folder. Converted files will be saved in the `data/output` folder.

2. **Using the File Conversion Tool**  
   The `file_conversion` function in `utils/fileConversion.py` can be used to convert files with options for bit depth, file ranges, and multi-page TIFF creation.

   Example usage:
   ```python
   from utils.fileConversion import file_conversion

   file_conversion(
       input_filename='input/test_00001.b16', 
       output_filename='test_converted.BMP', 
       bit_depth=16,  # Choose between 8, 16 (default), or 24
       start_num=10,  # Optional: starting number for file sequence
       stop_num=100,  # Optional: ending number for file sequence
       digit_range=4,  # Optional: number of digits for file numbering
       multi_tiff=False  # Set to True for multi-page TIFF
   )