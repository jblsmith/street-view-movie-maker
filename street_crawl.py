import sys
from utils import *
from API_KEYS import API_KEY_DIRECTIONS, API_KEY_STREETVIEW

'''Google Street View Movie Maker

Usage is:
	python2 ./street_crawl.py lat1 lon1 lat2 lon2 output_filestem


For example, to make a one-second video of the entrance of Joshua Treet National Park:
	python2 ./street_crawl.py 33.669793 -115.802125 33.671796 -115.801851 joshua_tree

Note: usage requires your own API keys.

'''

def main(lat_lon_A, lat_lon_B, filestem, picsize):
	print "Tracing path from ({0}) to ({1})".format(lat_lon_A, lat_lon_B)
	# Request driving directions from A to B
	gd = googlemaps.Client(key=API_KEY_DIRECTIONS)
	directions_result = gd.directions(origin=lat_lon_A, destination=lat_lon_B, mode="driving")
	# Convert driving directions into sequence of GPS points
	path_points = polyline.decode(directions_result[0]['overview_polyline']['points'])
	dense_points = [interpolate_points(pt[0],pt[1],hop_size=10) for pt in zip(path_points[:-1],path_points[1:])]	
	look_points_rough = [item for sequence in dense_points for item in sequence]
	# Remove unnecessary points
	look_points = clean_look_points(look_points_rough)
	print "For this route, there are {0} images to download.\n".format(len(look_points))
	continue_opt = raw_input('Would you like to download them all Type yes to proceed; otherwise, program halts.\n')
	if continue_opt not in ['Yes','yes']:
		return
	# Download sequence of images (up to a limit? What's the limit in a day?)
	download_images_for_path(API_KEY_STREETVIEW, filestem, look_points, picsize=picsize)
	# Assign images new filenames (and remove bad images)
	line_up_files(filestem)
	# Convert sequence of images to video
	make_video(filestem, rate=20, video_string=filestem, picsize=picsize)
	# TODO: Delete downloaded images

if __name__ == "__main__":
	lat_A, lon_A, lat_B, lon_B = [float(x) for x in sys.argv[1:5]]
	filestem = sys.argv[5]
	picsize = sys.argv[6]
	main((lat_A, lon_A), (lat_B, lon_B), filestem, picsize)
