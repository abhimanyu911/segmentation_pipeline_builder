import tensorflow as tf 
import os
import matplotlib.pyplot as plt

def build_pipeline(input_image_dir = None, output_image_dir = None, batch_sz = None):

    with tf.device('/cpu:0'):
        
        input_list = []
        output_list = []

        input_files = os.listdir(input_image_dir)
        output_files = os.listdir(output_image_dir)

        for input_file, output_file in zip(input_files, output_files):

            input_list.append(os.path.join(input_image_dir, input_file))
            output_list.append(os.path.join(output_image_dir, output_file))


        print('\n BUILDING DATASET ... \n')



        #Create the dataset from slices of the input and output filenames
        dataset = tf.data.Dataset.from_tensor_slices((input_list, output_list))
    
        #Parse the images from filename to the pixel values.
        #Use multiple threads to improve the speed of preprocessing
        dataset = dataset.map(process_path, num_parallel_calls=4)
    
        #Preprocess images
        #Use multiple threads to improve the speed of preprocessing
        dataset = dataset.map(preprocess, num_parallel_calls=4)

        print('\n SAMPLE INPUT-GROUND TRUTH OUPUT PAIR: \n')

        fig = plt.figure(figsize=(8, 8))

        for image, mask in dataset.take(1):

            arrays = [image, mask]
            titles = ['Input Image', 'True Mask']
            for i, array in enumerate(arrays):

                fig.add_subplot(1, 2, i+1)
                plt.imshow(tf.keras.preprocessing.image.array_to_img(array))
                plt.title(titles[i])
                plt.axis('off')
            plt.show()

    
        #cache, shuffle the data with a buffer size equal to the length of the dataset
        #(this ensures good shuffling), batch the images and prefetch one batch
        #to make sure that a batch is ready to be served at all time
        dataset = dataset.cache().shuffle(len(input_list)).batch(batch_sz).prefetch(1)

        print('\n DATASET READY AND RETURNED \n')
    
        return dataset

def process_path(image_path, mask_path):
    
    #read the content of the file
    img = tf.io.read_file(image_path)
    
    #decode using jpeg format
    img = tf.image.decode_png(img, channels=3)
    
    #convert to float values in [0, 1]
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    
    input_size = tf.constant(tuple(int(x.strip()) for x in input('Enter input image size as <rows>,<columns> for your model:\n').split(',')))
    output_size = tf.constant(tuple(int(x.strip()) for x in input('Enter output image size as <rows>,<columns> for your model:\n').split(',')))

    image = tf.image.resize(image, input_size, method='nearest')
    mask = tf.image.resize(mask, output_size, method='nearest')

    return image, mask
