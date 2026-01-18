"""
Simple script to list SAVEE dataset files.

This script can be run directly to list all files in the SAVEE dataset directory.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import from backend
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from run_models import list_savee_dataset_files
import json


def main():
    """Main function to list dataset files."""
    print("=" * 80)
    print("SAVEE Dataset File Lister")
    print("=" * 80)
    print()
    
    # Call the function
    result = list_savee_dataset_files()
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    
    if result.get('error'):
        print(f"❌ ERROR: {result['error']}")
        print(f"Path checked: {result['path']}")
        return
    
    print(f"✅ Path: {result['path']}")
    print(f"✅ Exists: {result['exists']}")
    print(f"✅ Total files: {result['total_files']}")
    print(f"✅ Directories: {len(result['directories'])}")
    print()
    
    # Print file extensions summary
    if result['file_extensions']:
        print("File Extensions:")
        for ext, count in sorted(result['file_extensions'].items()):
            print(f"  {ext}: {count} files")
        print()
    
    # Print directories
    if result['directories']:
        print("Subdirectories:")
        for directory in result['directories']:
            print(f"  📁 {directory}")
        print()
    
    # Print files (first 20)
    if result['files']:
        print(f"Files (showing first 20 of {result['total_files']}):")
        for i, file_info in enumerate(result['files'][:20], 1):
            print(f"  {i:3d}. {file_info['name']:<50} ({file_info['size_mb']:>8.2f} MB) [{file_info['extension']}]")
        
        if len(result['files']) > 20:
            print(f"  ... and {len(result['files']) - 20} more files")
    
    print("\n" + "=" * 80)
    print("Complete results available as JSON below:")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

