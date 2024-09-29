import os
import fnmatch
def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename:
                continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files
if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__) , './saved_data/')
    skip_mirrored_data = True
    hdf5_files = find_all_hdf5(dataset_dir, skip_mirrored_data)
    print(hdf5_files)