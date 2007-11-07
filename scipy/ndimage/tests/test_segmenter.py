
import numpy as N
import scipy.ndimage.segment as S

def shen_castan(image, IIRFilter=0.8, scLow=0.3, window=7, lowThreshold=220+2048, highThreshold=600+2048, dust=16):
	datatype = [('L', 'i', 1), ('R', 'i', 1), ('T', 'i', 1), ('B', 'i', 1),
            	    ('Label', 'i', 1), ('Area', 'i', 1), ('cX', 'f', 1), ('cY', 'f', 1),
            	    ('curveClose', 'i', 1), ('cXB', 'f', 1), ('cYB', 'f', 1), ('bLength', 'f', 1),
            	    ('minRadius', 'f', 1), ('maxRadius', 'f', 1), ('aveRadius', 'f', 1), ('ratio', 'f', 1),
            	    ('compactness', 'f', 1), ('voxelMean', 'f', 1), ('voxelVar', 'f', 1), ('TEM', 'f', 20)]
	labeledEdges, numberObjects = S.shen_castan_edges(scLow, IIRFilter, window, lowThreshold, highThreshold, image)
	# allocated struct array for edge object measures. for now just the rect bounding box
	ROIList = N.zeros(numberObjects, dtype=datatype)
	# return the bounding box for each connected edge
	S.get_object_stats(labeledEdges, ROIList)
	return labeledEdges, ROIList[ROIList['Area']>dust]

def sobel(image, sLow=0.3, tMode=1, lowThreshold=220+2048, highThreshold=600+2048, BPHigh=10.0, apearture=21, dust=16):
	datatype = [('L', 'i', 1), ('R', 'i', 1), ('T', 'i', 1), ('B', 'i', 1),
            	    ('Label', 'i', 1), ('Area', 'i', 1), ('cX', 'f', 1), ('cY', 'f', 1),
            	    ('curveClose', 'i', 1), ('cXB', 'f', 1), ('cYB', 'f', 1), ('bLength', 'f', 1),
            	    ('minRadius', 'f', 1), ('maxRadius', 'f', 1), ('aveRadius', 'f', 1), ('ratio', 'f', 1),
            	    ('compactness', 'f', 1), ('voxelMean', 'f', 1), ('voxelVar', 'f', 1), ('TEM', 'f', 20)]
	# get sobel edge points. return edges that are labeled (1..numberObjects)
	labeledEdges, numberObjects = S.sobel_edges(sLow, tMode, lowThreshold, highThreshold, BPHigh, apearture, image)
	# allocated struct array for edge object measures. for now just the rect bounding box
	ROIList = N.zeros(numberObjects, dtype=datatype)
	# return the bounding box for each connected edge
	S.get_object_stats(labeledEdges, ROIList)
	# thin (medial axis transform) of the sobel edges as the sobel produces a 'band edge'
	S.morpho_thin_filt(labeledEdges, ROIList)
	return labeledEdges, ROIList[ROIList['Area']>dust]

def canny(image, cSigma=1.0, cLow=0.5, cHigh=0.8, tMode=1, lowThreshold=220+2048, highThreshold=600+2048,
          BPHigh=10.0, apearture=21, dust=16):
	datatype = [('L', 'i', 1), ('R', 'i', 1), ('T', 'i', 1), ('B', 'i', 1),
            	    ('Label', 'i', 1), ('Area', 'i', 1), ('cX', 'f', 1), ('cY', 'f', 1),
            	    ('curveClose', 'i', 1), ('cXB', 'f', 1), ('cYB', 'f', 1), ('bLength', 'f', 1),
            	    ('minRadius', 'f', 1), ('maxRadius', 'f', 1), ('aveRadius', 'f', 1), ('ratio', 'f', 1),
            	    ('compactness', 'f', 1), ('voxelMean', 'f', 1), ('voxelVar', 'f', 1), ('TEM', 'f', 20)]
	# get canny edge points. return edges that are labeled (1..numberObjects)
	labeledEdges, numberObjects = S.canny_edges(cSigma, cLow, cHigh, tMode, lowThreshold, highThreshold, 
			                           BPHigh, apearture, image)
	# allocated struct array for edge object measures. for now just the rect bounding box
	ROIList = N.zeros(numberObjects, dtype=datatype)
	# return the bounding box for each connected edge
	S.get_object_stats(labeledEdges, ROIList)
	return labeledEdges, ROIList[ROIList['Area']>dust]

def get_shape_mask(labeledEdges, ROIList):
	# pass in Sobel morph-thinned labeled edge image (LEI) and ROIList
	# GetShapeMask will augment the ROI list
	# labeledEdges is the original edge image and overwritten as mask image
	# maskImage is the mask that is used for blob texture / pixel features
	S.build_boundary(labeledEdges, ROIList)
	return 

def get_voxel_measures(rawImage, labeledEdges, ROIList):
	#
	# pass raw image, labeled mask and the partially filled ROIList
	# VoxelMeasures will fill the voxel features in the list
	#
	S.voxel_measures(rawImage, labeledEdges, ROIList)
	return 

def get_texture_measures(rawImage, labeledEdges, ROIList):
	#
	# pass raw image, labeled mask and the partially filled ROIList
	# VoxelMeasures will fill the texture (Law's, co-occurence, Gabor) features in the list
	#
	S.texture_measures(rawImage, labeledEdges, ROIList)
	return 

def segment_regions():
	# get slice from the CT volume
	image = get_slice('slice112.raw')
	# need a copy of original image as filtering will occur on the extracted slice
    	sourceImage = image.copy()
	# Sobel is the first level segmenter. Sobel magnitude and MAT (medial axis transform)
	# followed by connected component analysis. What is returned is labeled edges and the object list
    	labeledMask, ROIList = sobel(image)
	# From the labeled edges and the object list get the labeled mask for each blob object
    	get_shape_mask(labeledMask, ROIList)
	# Use the labeled mask and source image (raw) to get voxel features 
    	get_voxel_measures(sourceImage, labeledMask, ROIList)
	# Use the labeled mask and source image (raw) to get texture features 
	get_texture_measures(sourceImage, labeledMask, ROIList)
	return sourceImage, labeledMask, ROIList

def grow_regions():
	# get slice from the CT volume
	image = get_slice('slice112.raw')
	regionMask, numberRegions = region_grow(image)
	return regionMask, numberRegions 


def region_grow(image, lowThreshold=220+2048, highThreshold=600+2048, open=7, close=7):
	# morphology filters need to be clipped to 11 max and be odd
	regionMask, numberRegions = S.region_grow(lowThreshold, highThreshold, close, open, image)
	return regionMask, numberRegions
          

def get_slice(imageName='junk.raw', bytes=2, rows=512, columns=512):
	# get a slice alrady extracted from the CT volume
	#image = open(imageName, 'rb')
	#slice = image.read(rows*columns*bytes)
	#values = struct.unpack('h'*rows*columns, slice)
	#ImageSlice = N.array(values, dtype=float).reshape(rows, columns)

	ImageSlice = N.fromfile(imageName, dtype=N.uint16).reshape(rows, columns);

	# clip the ends for this test CT image file as the spine runs off the end of the image
	ImageSlice[505:512, :] = 0
	return (ImageSlice).astype(float)

def get_slice2(image_name='junk.raw', bytes=2, shape=(512,512)):
        import mmap
        file = open(image_name, 'rb')
        mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        slice = N.frombuffer(mm, dtype='u%d' % bytes).reshape(shape) 
        slice = slice.astype(float)
        slice[505:512,:] = 0
        return slice

def save_slice(mySlice, filename='junk.raw', bytes=4):
	# just save the slice to a fixed file
	slice = mySlice.astype('u%d' % bytes)
        slice.tofile(filename)


# test 1
def test1():
    import Segmenter as S
    image = S.get_slice('slice112.raw')
    sourceImage = image.copy()
    edges, objects = S.sobel(image)
    S.get_shape_mask(edges, objects)
    S.get_voxel_measures(sourceImage, edges, objects)
    S.get_texture_measures(sourceImage, edges, objects)

# test 2
def test2():
    import volumeInput as V
    import Segmenter as S
    sourceImage, labeledMask, ROIList = S.segment_regions()


# test 3
def test3():
    import Segmenter as S
    regionMask, numberRegions = S.grow_regions()
    regionMask.max()
    S.save_slice(regionMask, 'regionMask.raw')



class TestSegmenter(NumpyTestCase):
    def test_1(self):


    