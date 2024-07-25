import numpy as np
import tensorflow as tf
from os import listdir, makedirs
from os.path import isfile, join
from tqdm import tqdm
from argparse import ArgumentParser



def parse_tfrecord_file(tfrecord_filepath):
    """Extract embeddings from a single TFRecord file."""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_filepath)
    for raw_record in raw_dataset.take(1):   # Only take the first record
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        embedding = np.array(example.features.feature['embedding'].float_list.value, dtype=np.float32)   # should specify dtype=np.float32 for proper embeddings handling
        return embedding   # Return embedding after parsing the first record
    

def convert_tfrecords_to_dat(tf_dir, np_dir):
    """Convert all TFRecord files in a directory to NumPy .dat files in another directory."""
    makedirs(np_dir, exist_ok=True)   # Ensure the output directory exists
    files = [f for f in listdir(tf_dir) if isfile(join(tf_dir, f))]

    for f in tqdm(files, desc='Converting'):
        try:
            tfrecord_filepath = join(tf_dir, f)
            embedding = parse_tfrecord_file(tfrecord_filepath)
            if embedding is not None:
                output_filepath = join(np_dir, f.replace('.tfrecord', '.dat'))
                embedding.tofile(output_filepath)
        except Exception as e:
            print(f"Failed to process {f}: {e}")



if __name__ == '__main__':
    parser = ArgumentParser(description="Convert TFRecord files to NumPy .dat files.")
    parser.add_argument('--tf_dir', type=str, default='/vol/biomedic3/bglocker/cxr-foundation/outputs/chexpert/cxr_tfrecords/',
                        help='Directory containing TFRecord files.')
    parser.add_argument('--np_dir', type=str, default='/vol/biomedic3/bglocker/mscproj24/fz221/data/cxrfm_embeddings/chexpert/cxr_numpy/',
                        help='Output directory for NumPy .dat files.')
    
    args = parser.parse_args()

    # Run the conversion
    convert_tfrecords_to_dat(args.tf_dir, args.np_dir)