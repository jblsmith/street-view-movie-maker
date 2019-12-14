from __future__ import print_function
# Some useful Google API documentation:
# https://developers.google.com/maps/documentation/directions/
# https://developers.google.com/maps/documentation/roads/snap

import googlemaps
import urllib, os
import numpy as np
import json
import pandas as pd
import polyline
import glob
import subprocess
import math
PHOTO_FOLDER = "./photos/"

# Adapted directly from Andrew Wheeler:
# https://andrewpwheeler.wordpress.com/2015/12/28/using-python-to-grab-google-street-view-imagery/
# Usage example:
# >>> download_streetview_image((46.414382,10.012988))
def download_streetview_image(apikey_streetview, lat_lon, filename="image", savepath=PHOTO_FOLDER, picsize="600x300", heading=151.78, pitch=-0, fi=".jpg", fov=90, get_metadata=False, verbose=True, outdoor=True, radius=5):
	assert type(radius) is int
	# Any size up to 640x640 is permitted by the API
	# fov is the zoom level, effectively. Between 0 and 120.
	base = "https://maps.googleapis.com/maps/api/streetview"
	if get_metadata:
		base = base + "/metadata?parameters"
	if type(lat_lon) is tuple:
		lat_lon_str = str(lat_lon[0]) + "," + str(lat_lon[1])
	elif type(lat_lon) is str:
		# We expect a latitude/longitude tuple, but if you providing a string address works too.
		lat_lon_str = lat_lon
	if outdoor:
		outdoor_string = "&source=outdoor"
	else:
		outdoor_string = ""
	url = base + "?size=" + picsize + "&location=" + lat_lon_str + "&heading=" + str(heading) + "&pitch=" + str(pitch) + "&fov=" + str(fov) + outdoor_string + "&radius" + str(radius) + "&key=" + apikey_streetview
	if verbose:
		print(url)
	if get_metadata:
		# Description of metadata API: https://developers.google.com/maps/documentation/streetview/intro#size
		response = urllib.urlopen(url)
		data = json.loads(response.read())
		return data
	else:
		urllib.urlretrieve(url, savepath+filename+fi)
		return savepath+filename+fi

# Gist copied from https://gist.github.com/jeromer/2005586 which is in the public domain:
def calculate_initial_compass_bearing(pointA, pointB):
	if (type(pointA) != tuple) or (type(pointB) != tuple):
		raise TypeError("Only tuples are supported as arguments")
	lat1 = math.radians(pointA[0])
	lat2 = math.radians(pointB[0])
	diffLong = math.radians(pointB[1] - pointA[1])
	x = math.sin(diffLong) * math.cos(lat2)
	y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
			* math.cos(lat2) * math.cos(diffLong))
	initial_bearing = math.atan2(x, y)
	initial_bearing = math.degrees(initial_bearing)
	compass_bearing = (initial_bearing + 360) % 360
	return compass_bearing

def haversine(a_gps, b_gps):
	"""
	Calculate the great circle distance between two points 
	on the earth (specified in decimal degrees)
	"""
	lat1, lon1 = a_gps
	lat2, lon2 = b_gps
	# convert decimal degrees to radians 
	lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
	# haversine formula 
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
	c = 2 * math.asin(math.sqrt(a)) 
	km = 6367 * c
	m = 6367000.0 * c
	return m

# Given two GPS points (lat/lon), interpolate a sequence of GPS points in a straight line
def interpolate_points(a_gps,b_gps,n_points=10,hop_size=None):
	if hop_size is not None:
		distance = haversine(a_gps, b_gps)
		n_points = np.ceil(distance*1.0/hop_size)
	x = np.linspace(a_gps[0],b_gps[0],n_points)
	y = np.linspace(a_gps[1],b_gps[1],n_points)
	dense_points_list = zip(x,y)
	return dense_points_list
	# else:
	#	 print("You forgot to provide a hop parameter! Choose between:")
	#	 print("  n_points = number of points to interpolate;")
	#	 print("  hop_size = maximum distance between points in meters.")

# Short script to process the lookpoints from the above "interpolate points" function.
def clean_look_points(look_points):
	# Remove points that are the same
	pt_diffs = [np.array(a)-np.array(b) for (a,b) in zip(look_points[:-1],look_points[1:])]
	keepers = np.abs(np.array(pt_diffs))>0
	look_points_out = [look_points[i] for i in range(len(keepers)) if np.any(keepers[i])]
	return look_points_out

# Download street view images for a sequence of GPS points.
# The orientation is assumed to be towards the next point.
# Setting orientation to value N orients the camera to the Nth next point.
# If there isn't a point N points in the future, we just use the previous heading.
def download_images_for_path(apikey_streetview, filestem, look_points, orientation=1, picsize="640x320"):
	assert type(orientation) is int
	assert orientation >= 1
	for i in range(len(look_points)):
		gps_point = look_points[i]
		if i+orientation >= len(look_points):
			heading = prev_heading
		else:
			heading = calculate_initial_compass_bearing(gps_point, look_points[i+orientation])
		probe = download_streetview_image(apikey_streetview, gps_point, filename="", heading=heading, picsize=picsize, get_metadata=True)
		if probe['status']=="OK" and 'Google' in probe['copyright']:
			dest_file = download_streetview_image(apikey_streetview, gps_point, filename=filestem + str(i), heading=heading, picsize=picsize, get_metadata=False)
		prev_heading = heading

def get_turn_headings(h1, h2, stepsize=15):
	if h2 < h1:
		h2 += 360
	clockwise = (h2 - h1 < 180)
	if not clockwise:
		h1 += 360
	n_points = np.ceil(np.abs( (h1 - h2)*1.0 /stepsize))
	headings = np.linspace(h1,h2,n_points)
	return np.mod(headings,360)

# def execute_turn(apikey_streetview, filestem, gps_point, h1, h2, picsize="640x320", stepsize=15):
# 	if h2 < h1:
# 		h2 += 360
# 	clockwise = (h2 - h1 < 180)
# 	if not clockwise:
# 		h1 += 360
# 	n_points = np.ceil(np.abs( (h1 - h2)*1.0 /stepsize))
# 	headings = np.linspace(h1,h2,n_points)
# 	probe = download_streetview_image(apikey_streetview, gps_point, filename="", heading=headings[0], picsize=picsize, get_metadata=True)
# 	if probe['status']=="OK" and 'Google' in probe['copyright']:
# 		for h_i,h in enumerate(headings):
# 			dest_file = download_streetview_image(apikey_streetview, gps_point, filename="{0}_turn_{1}".format(filestem,h_i), heading=h, picsize=picsize, get_metadata=False)
#
def generate_download_sequence(gps_points, savename):
	# Create dataframe with GPS points
	pt_list = pd.DataFrame(index=range(len(gps_points)), data=gps_points, columns=["lat","lon"])
	# Compute basic headings
	headings = [calculate_initial_compass_bearing(pt[0], pt[1]) for pt in zip(gps_points[:-1],gps_points[1:])]
	pt_list['heading'] = headings + [headings[-1]]
	# Set up probes and collect all in raw form
	pt_list['probe'] = [{} for i in pt_list.index]
	for i in pt_list.index:
		pt_list['probe'][i] = download_streetview_image(apikey_streetview, (pt_list["lat"][i],pt_list["lon"][i]), filename="", heading=pt_list["heading"][i], get_metadata=True)
	# Assign probe items to their own columns:
	probe_items = ['copyright', 'date', 'location', 'pano_id', 'status']
	for p_item in probe_items:
		pt_list[p_item] = [x[p_item] for x in pt_list['probe']]
	pt_list.to_pickle(savename)
	return pt_list

def create_itinerary_df(gps_points):
	# Create dataframe with GPS points
	pt_list = pd.DataFrame(index=range(len(gps_points)), columns=["lat", "lon", "heading", "probe", "copyright", "date", "location", "pano_id", "status", "downloaded_1", "downloaded_array"])
	lats, lons = zip(*gps_points)
	pt_list['lat'] = lats
	pt_list['lon'] = lons
	pt_list['downloaded_1'] = False
	pt_list['downloaded_array'] = False
	# Compute basic headings
	headings = [calculate_initial_compass_bearing(pt[0], pt[1]) for pt in zip(gps_points[:-1],gps_points[1:])]
	pt_list['heading'] = headings + [headings[-1]]
	# pt_list['probe'] = [{} for i in pt_list.index]
	pt_list = pt_list.fillna('')
	return pt_list

def probe_itinerary_items(itinerary_df, indlist, apikey_streetview, redo=False):
	assert [i in itinerary_df.index for i in indlist]
	probe_items = ['copyright', 'date', 'location', 'pano_id', 'status']
	for i in indlist:
		if (itinerary_df['status'][i] == '') or (redo):
			print(i)
			probe_result = download_streetview_image(apikey_streetview, (itinerary_df["lat"].loc[i],itinerary_df["lon"][i]), filename="", heading=itinerary_df["heading"][i], get_metadata=True)
			# itinerary_df.loc[i]["probe"] = probe_result
			# Assign probe items to their own columns:
			for p_item in probe_result.keys():
				itinerary_df[p_item][i] = probe_result[p_item]

def process_pointlist(pt_list=None, pt_list_filename=None):
	if pt_list is None and pt_list_filename is not None:
		pt_list = pd.read_pickle(pt_list_filename)
	# Remove duplicate / invalid points:
	unique_panos = np.unique(pt_list.pano_id)
	panoid_to_ind = {panoid:pt_list.pano_id.eq(panoid).idxmax() for panoid in unique_panos}
	keepers = [i for i in sorted(panoid_to_ind.values()) if pt_list.status[i]=='OK' and 'Google' in pt_list.copyright[i]]
	new_list = pt_list.loc[keepers]
	new_list.index = np.arange(new_list.shape[0])
	crit_diff = 5
	turn_indices = new_list.loc[np.abs(np.diff(new_list.heading))>crit_diff].index
	new_rows = []
	for ti in turn_indices:
		h1 = new_list.headings[ti]
		h2 = new_list.headings[ti+1]
		headings = get_turn_headings(h1, h2, stepsize=1)[1:-1]
		tmp_df = pd.DataFrame(np.tile(new_list.loc[ti],(len(headings),1)))
		tmp_df.columns = new_list.columns
		tmp_df.heading = headings
		tmp_df.index = np.linspace(ti+0.01,ti+0.99,len(headings))
		new_rows += [tmp_df]
	final_list = pd.concat([new_list]+new_rows)
	final_list = final_list.sort_index()
	final_list.index = np.arange(final_list.shape[0])
	return final_list

def download_pics_from_list(item_list, apikey_streetview, filestem, picsize, redownload=False, index_filter=None):
	if index_filter is None:
		index_filter = item_list.index
	for i in index_filter:
		row = item_list.loc[i]
		lat, lon, heading, downloaded = row['lat'], row['lon'], row['heading'], row['downloaded_1']
		if (not downloaded) or redownload:
			download_streetview_image(apikey_streetview, (lat,lon), filename=filestem + str(i), heading=heading, picsize=picsize, get_metadata=False)
			item_list["downloaded_1"].loc[i] = True

def download_tableaux_from_list(item_list, apikey_streetview, filestem, picsize, fov, fov_step, pitch, grid_dim, index_filter=None):
	if index_filter is None:
		index_filter = item_list.index
	for i in index_filter:
		row = item_list.loc[i]
		lat, lon, heading, downloaded = row['lat'], row['lon'], row['heading'], row['downloaded_array']
		download_images_for_point(apikey_streetview, (lat,lon), filestem + str(i), "./photos/", heading, fov, fov_step, pitch, grid_dim)
		if (not downloaded) or redownload:
			assemble_grid_of_images(filestem + str(i), "./photos/", "./photos/composite-{0}-{1}".format(filestem,i), grid_dim, crop_dim="640x640+0+0")
			item_list["downloaded_array"].loc[i] = True

# Download set of zoomed-in views to be composited into a larger image

# Download set of zoomed-in views to be composited into a larger image
def download_images_for_point(apikey_streetview, lat_lon, filestem, savepath, heading, fov = 30, fov_step = 30, pitch = 15, grid_dim = [4,2]):
	horiz_points = (np.arange(grid_dim[0]) - (grid_dim[0]-1)/2.0) * fov_step
	vert_points = (np.arange(grid_dim[1])[::-1] - (grid_dim[1]-1)/2.0) * fov_step + pitch
	# horiz_points = np.linspace(-1, 1, grid_dim[0]) * (fov / 90.0)
	# vert_points = np.linspace(max_pitch, min_pitch, grid_dim[1]) * (fov / 90.0)
	# fov_angle_frac = 1.0 * fov / max(grid_dim)
	# fudge_factor = 5
	# assert fov_angle_frac >= 15
	panel_inds = np.reshape(np.arange(np.prod(grid_dim)), grid_dim, 1).transpose()
	for ix,x in enumerate(horiz_points):
		for iy,y in enumerate(vert_points):
			panel_ind = panel_inds[iy,ix]
			print(panel_ind)
			tmp_heading = heading + x
			tmp_pitch = y
			print(tmp_heading, tmp_pitch)
			download_streetview_image(apikey_streetview, lat_lon, filename="{0}_{1}".format(filestem,panel_ind), savepath=savepath, picsize="640x640", heading=tmp_heading, pitch=tmp_pitch, fi=".jpg", fov=fov, get_metadata=False, verbose=True, outdoor=True, radius=5)

def assemble_grid_of_images(filestem, savepath, outfilestem, grid_dim, crop_dim="640x640+0+0"):
	panel_inds = np.reshape(np.arange(np.prod(grid_dim)), grid_dim, 1).transpose()
	grid_filenames = [["{0}/{1}_{2}.jpg -crop {3}".format(savepath,filestem,pind,crop_dim) for pind in pindrow] for pindrow in panel_inds]
	command_string = "convert " + "   ".join([" \( " + " ".join(row+["+append"]) + " \) " for row in grid_filenames]) + " -append {0}.jpg".format(outfilestem)
	# print(command_string)
	subprocess.call(command_string, shell=True)

# Line up files in order to make a video using ffmpeg.
# ffmpeg requires all images files numbered in sequence, with no gaps.
# However, some images will not have been downloaded, so we need to shift everything to tidy up gaps.
# Also, some images will be duplicates, and we can remove them.
# Also, a user may want to manually discard images because they are clearly out of step with the path (e.g., they might be view inside a building, or slightly down a cross-street.) After manually removing files, re-running this will line up the files.
def line_up_files(filestem, new_dir="./movie_lineup", command="mv", override_nums=None):
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)
	files = glob.glob("./photos/"+filestem+"*.jpg")
	file_nums = [int(filename[9+len(filestem):-4]) for filename in files]
	file_sort = [files[i] for i in np.argsort(file_nums)]
	# First, remove file_nums that represent duplicate files
	file_keepers = prune_repeated_images_from_list(file_sort)
    # for i in range(1,len(file_sort)):
    #     prev_file = file_keepers[-1]
    #     curr_file = file_sort[i]
    #     result = os.system("diff " + curr_file + " " + prev_file)
    #     if result > 0:
    #         file_keepers += [curr_file]
	# Now, shuffle the files into a packed numbering:
	for i in range(len(file_keepers)):
		old_filename = file_keepers[i]
		new_filename = "{0}/{1}{2}.jpg".format(new_dir,filestem,i)
		print("{0} {1} {2}".format(command, old_filename, new_filename))
		os.system("{0} {1} {2}".format(command, old_filename, new_filename))

# Refactor line_up_files as separate steps:
def line_up_files_with_numbers_script(filestem, numbers, new_dir):
    files = ["./photos/{0}{1}.jpg".format(filestem,num) for num in sorted(numbers)]
    file_keepers = prune_repeated_images_from_list(files)
    copy_files_to_sequence(file_keepers, "./photos/{0}/{1}".format(new_dir, filestem))

def copy_files_to_sequence(list_of_files, new_filestem, command='cp'):
	for i in range(len(list_of_files)):
		old_filename = list_of_files[i]
		new_filename = "{0}{1}.jpg".format(new_filestem,i)
		print("{0} {1} {2}".format(command, old_filename, new_filename))
		os.system("{0} {1} {2}".format(command, old_filename, new_filename))

def prune_repeated_images_from_list(list_of_files):
    file_keepers = [list_of_files[0]]
    for i in range(1,len(list_of_files)):
    	prev_file = file_keepers[-1]
    	curr_file = list_of_files[i]
    	result = os.system("diff " + curr_file + " " + prev_file)
    	if result > 0:
            file_keepers += [curr_file]
    return file_keepers

def make_video(base_string, rate=20, video_string=None, picsize="640x640", basepath="./photos"):
	if video_string is None:
		video_string = base_string
	subprocess.call("ffmpeg -r {0} -f image2 -s {3} -i {4}/{1}%d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p {2}.mp4 -y".format(rate, base_string, video_string, picsize, basepath), shell=True)


















