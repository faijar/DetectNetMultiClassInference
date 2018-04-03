import argparse
import os
import time
import cv2
from google.protobuf import text_format
import numpy as np
import PIL.Image
import scipy.misc

os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output
import caffe
from caffe.proto import caffe_pb2

def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net
    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)

def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer
    Arguments:
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) 

    # color images
    if dims[1] == 3:
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        with open(mean_file,'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t


def non_max_suppression_slow(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	pick = []
        
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
		for pos in xrange(0, last):
			# grab the current index
			j = idxs[pos]
 
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
 
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
 
			overlap = float(w * h) / area[j]
 
			if overlap > overlapThresh:
				suppress.append(pos)
 
		idxs = np.delete(idxs, suppress)
	return boxes[pick]

def resize_img(image, height, width):
    """
    Resizes the image to detectnet inputs

    Arguments:
    image -- a single image
    height -- height of the network input
    width -- width of the network input
    """
    image = np.array(image)
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

def draw_bboxes(image, locations, i):
    """
    Draws the bounding boxes into an image

    Arguments:
    image -- a single image already resized
    locations -- the location of the bounding boxes
    """
    
 
    pick = non_max_suppression_slow(locations, 0.3)
    print(pick)
    for startX, startY, endX, endY, confidence in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image,str(i+1), (startX,startY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    #cv2.imwrite('bbox.png',image)#test on a single image
    return image
def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk
    Returns an np.ndarray (channels x width x height)
    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension
    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

def forward_pass(images, net, transformer, i, batch_size=None):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)
    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer
    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    if batch_size is None:
        batch_size = 1

    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:,:,np.newaxis])
        else:
            caffe_images.append(image)

    dims = transformer.inputs['data'][1:]

    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        start = time.time()
        output = net.forward()[net.outputs[i]]
        end = time.time()
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))
        print 'Processed %s/%s images in %f seconds ...' % (len(scores), len(caffe_images), (end - start))

    return scores

def read_labels(labels_file):
    """
    Returns a list of strings
    Arguments:
    labels_file -- path to a .txt file
    """
    if not labels_file:
        print 'WARNING: No labels file provided. Results will be difficult to interpret.'
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels

def classify(caffemodel, deploy_file, image_files, image, i,
        mean_file=None, labels_file=None, batch_size=None, use_gpu=True):
    """
    Classify some images against a Caffe model and print the results
    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images
    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    # Load the model and images
    net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
    images = [load_image(image_file, height, width, mode) for image_file in image_files]
    labels = read_labels(labels_file)

    # Classify the image
    
    scores = forward_pass(images, net, transformer,i, batch_size=batch_size)
    
    ### Process the results

    # Format of scores is [ batch_size x max_bbox_per_image x 5 (xl, yt, xr, yb, confidence) ]
    # https://github.com/NVIDIA/caffe/blob/v0.15.13/python/caffe/layers/detectnet/clustering.py#L81
    for j, image_results in enumerate(scores):
        print '==> Image #%d' % j
        for left, top, right, bottom, confidence in image_results:
            if confidence == 0:
                continue

            print 'Detected object at [(%d, %d), (%d, %d)] with "confidence" %f' % (
                int(round(left)),
                int(round(top)),
                int(round(right)),
                int(round(bottom)),
                confidence,
            )
    #img = resize_img(image,height,width)
    print(width)
    img_result = image
    
    img_result = draw_bboxes(image,image_results, i)
    img_result = resize_img(img_result,720,1280)
    return img_result

if __name__ == '__main__':
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Classification example - DIGITS')

    ### Positional arguments

    parser.add_argument('caffemodel',   help='Path to a .caffemodel')
    parser.add_argument('deploy_file',  help='Path to the deploy file')
    parser.add_argument('image_file',
                        nargs='+',
                        help='Path[s] to an image')

    ### Optional arguments

    parser.add_argument('-m', '--mean',
            help='Path to a mean file (*.npy)')
    parser.add_argument('-l', '--labels',
            help='Path to a labels file')
    parser.add_argument('--batch-size',
                        type=int)
    parser.add_argument('--nogpu',
            action='store_true',
            help="Don't use the GPU")

    args = vars(parser.parse_args())
   
    print 'Script took %f seconds.' % (time.time() - script_start_time,)
    print str(args['image_file'])
    str1 = ''.join(args['image_file'])
    img = cv2.imread(str1)
    image = resize_img(img,400,500) #resize to network input size
    for i in range(-1,8):
    	result = classify(args['caffemodel'], args['deploy_file'], args['image_file'], image, i,
           	  args['mean'], args['labels'], args['batch_size'], not args['nogpu'])
    cv2.imshow("image", result)
    cv2.waitKey(0)
