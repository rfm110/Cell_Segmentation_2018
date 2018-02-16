from skimage.measure import regionprops

def find_area_in_binary_array(labeled_binary_image):

    #using regionprops only gives the number of pixels in a given region
    #would need to combine it with image shape
    #alternatively, find the radius
    area_of_labeled_regions = []
    for segment in regionprops(labeled_binary_image):
        area_of_labeled_regions.append(segment.area)

    return area_of_labeled_regions
    #need it in list form?





