import subprocess
import os
from utils import get_root

def file_conversion(input_filename, output_filename, fileconversion_path=None, bit_depth=16, 
                    start_num=None, stop_num=None, digit_range=None, multi_tiff=False):
    """
    Convert a .b16 or .pcoraw file to a different format (e.g., .tif) using the PCO file conversion command-line tool.

    Parameters:
    ----------
    input_filename : str
        The relative path to the input file (from the project root) within the 'data' folder.
        When using a scanning range (i.e., with `start_num` and `stop_num`), it is preferred to use
        a filename in the format `<filename>_0000` (or similar) so that the tool can correctly scan the sequence of files.
        
    output_filename : str
        The relative path to the output file (from the project root) within the 'data/output' folder.
        
    fileconversion_path : str, optional
        The full path to the 'pco_file_cmd.exe' file. If not provided, it defaults to 
        'C:\\Program Files\\PCO Digital Camera Toolbox\\pco.fileconversion\\pco_file_cmd.exe' or the environment variable 'PCO_CONVERTER_PATH'.
        
    bit_depth : int, optional
        Bit depth for the output file. It can be 8, 16 (default), or 24.
        
    start_num : int, optional
        The start number for a range of files to be processed. Only used if converting a sequence of files.
        
    stop_num : int, optional
        The stop number for a range of files to be processed. Only used if converting a sequence of files.
        
    digit_range : int, optional
        The number of digits for the input file numbering (e.g., 4 for `file_0001.b16`).
        
    multi_tiff : bool, optional
        If True, the output will be a multi-page .tif file.

    Raises:
    ------
    subprocess.CalledProcessError
        If the file conversion command fails, an error message is printed.

    Returns:
    -------
    None
        Prints a success message if the conversion is successful.
    """
    # Set default path for file conversion tool if not provided
    if fileconversion_path is None:
        fileconversion_path = os.getenv('PCO_CONVERTER_PATH', r'C:\Program Files\PCO Digital Camera Toolbox\pco.fileconversion\pco_file_cmd.exe')
    
    # Define input and output file paths
    project_root = get_root()
    data_folder_dir = os.path.join(project_root, 'data')
    input_file = os.path.join(data_folder_dir, input_filename)
    output_file = os.path.join(data_folder_dir, 'output', output_filename)

    # Construct the command for file conversion
    command = [fileconversion_path, '-i', input_file, '-o', output_file, '-b', str(bit_depth)]

    # Add optional parameters if provided
    if start_num is not None and stop_num is not None:
        command.extend(['-n', f'{start_num}', f'{stop_num}'])
    if digit_range is not None:
        command.extend(['-s', str(digit_range)])
    if multi_tiff:
        command.append('-m')
    
    # Execute the file conversion command
    try:
        subprocess.run(command, check=True)
        print(f"File {input_file} converted to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
