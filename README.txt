This is a library with one module to help you build a tf.data Dataset pipeline for the purpose of semantic image segmentation.
Refer 'demo.ipynb' at https://github.com/abhimanyu911/segmentation_pipeline_builder for instructions 

Modules:

build_pipeline(input_image_dir = None, output_image_dir = None , batch_sz = None ,repeat_shuffle = None): Returns a tf.data Datset pipeline with input image- true mask pairs

Arguments:


input_image_dir : (str) local directory containing input images
output_image_dir : (str)local directory containing ground truth output images
batch_sz : (int) batch size
repeat_shuffle : (boolean) option to repeat shuffling for multiple epochs
