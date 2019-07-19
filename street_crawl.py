# Some useful Google API documentation:
# https://developers.google.com/maps/documentation/directions/
# https://developers.google.com/maps/documentation/roads/snap

from API_KEYS import *
import googlemaps
from datetime import datetime
import urllib, os
import numpy as np
import imreg_dft as ird
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image
import json
# plt.ion()
myloc = "/Users/jordan/Documents/repositories/street-view-animation/"
photo_folder = myloc + "photos/"

# Adapted directly from Andrew Wheeler:
# https://andrewpwheeler.wordpress.com/2015/12/28/using-python-to-grab-google-street-view-imagery/
# Usage example:
# >>> download_streetview_image((46.414382,10.012988))
def download_streetview_image(lat_lon, filename="image", savepath=photo_folder, size="600x300", heading=151.78, pitch=-0.76, fi=".jpg", fov=90, get_metadata=False, verbose=True, outdoor=True):
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
    url = base + "?size=" + size + "&location=" + lat_lon_str + "&heading=" + str(heading) + "&pitch=" + str(pitch) + "&fov=" + str(fov) + outdoor_string + "&key=" + API_KEY_STREETVIEW
    if verbose:
        print url
    if get_metadata:
        # Description of metadata API: https://developers.google.com/maps/documentation/streetview/intro#size
        response = urllib.urlopen(url)
        data = json.loads(response.read())
        return data
    else:
        urllib.urlretrieve(url, savepath+filename+fi)

def download_streetview_circle(lat_lon,filename, **kwargs):
    for heading in range(0,359,15):
        download_streetview_image(lat_lon, heading=heading, filename=filename+str(heading), **kwargs)


download_streetview_circle(central_park_s, filename="cp_pano_")

def make_video(base_string, rate=20):
    from subprocess import call
    call("ffmpeg -r 25 -f image2 -s 600x300 -i photos/cp_pano_%d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p cp_pano.mp4")
    call(["ffmpeg", "-r", str(rate), "-f", "image2", "-s", "600x300", "-i", base_string+"%d.jpg", "-vcodec libx264 -crf 25 -pix_fmt yuv420p", base_string+"_pano.mp4"])
    # ffmpeg -r 25 -f image2 -s 600x300 -i photos/central_park%d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p central_park_bike_ride.mp4
    

# Given two GPS points (lat/lon), interpolate a sequence of GPS points in a straight line
def interpolate_points(a_gps,b_gps,n_points=10,hop_size=None):
    if hop_size is not None:
        distance = haversine(a_gps, b_gps)
        n_points = np.ceil(distance*1.0/hop_size)
    x = np.linspace(a_gps[0],b_gps[0],n_points)
    y = np.linspace(a_gps[1],b_gps[1],n_points)
    return zip(x,y)
    # else:
    #     print "You forgot to provide a hop parameter! Choose between:"
    #     print "  n_points = number of points to interpolate;"
    #     print "  hop_size = maximum distance between points in meters."

# Given two GPS points, find the heading (as an angle from true north) that looks from point A to point B
def get_angle_between_points(a,b):
    if a==b:
        print "Points a and b are the same! Returning angle 0."
        return 0
    x = b[0]-a[0]
    y = b[1]-a[1]
    r = np.sqrt(np.power(x,2) + np.power(y,2))
    theta = np.arcsin(y/r) * 180 / np.pi
    # Q2:
    if (y>0) and (x<0):
        # theta is negative already, so add 180
        theta += 180
    # Q3:
    elif (y<0) and (x<0):
        # theta is negative
        theta -= 180
    # Q4:
    elif (y<0) and (x>0):
        theta += 360
    angle_from_north = 90 - theta
    return angle_from_north

def crude_estimate_bearing(a_latlon,b_latlon):
    import geopy
    import geopy.distance
    start = geopy.Point(a_latlon[1],a_latlon[0])
    goal = geopy.Point(b_latlon[1],b_latlon[0])
    d = geopy.distance.VincentyDistance(kilometers = 0.01)
    dest_opts = []
    for bearing in range(360):
        dist = geopy.distance.great_circle(goal, d.destination(point=start, bearing=bearing)).miles
        dest_opts.append(dist)
    bearing_ccw_from_east = np.argmin(dest_opts)
    bearing_ccw_from_north = bearing_ccw_from_east - 90
    bearing_cw_from_north = np.mod(-bearing_ccw_from_north,360)
    return bearing_cw_from_north

# Gist copied from https://gist.github.com/jeromer/2005586 which is in the public domain:
def calculate_initial_compass_bearing(pointA, pointB):
	import math
	"""
	Calculates the bearing between two points.
	The formulae used is the following:
		θ = atan2(sin(Δlong).cos(lat2),
				  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
	:Parameters:
	  - `pointA: The tuple representing the latitude/longitude for the
		first point. Latitude and longitude must be in decimal degrees
	  - `pointB: The tuple representing the latitude/longitude for the
		second point. Latitude and longitude must be in decimal degrees
	:Returns:
	  The bearing in degrees
	:Returns Type:
	  float
	"""
	if (type(pointA) != tuple) or (type(pointB) != tuple):
		raise TypeError("Only tuples are supported as arguments")
	lat1 = math.radians(pointA[0])
	lat2 = math.radians(pointB[0])
	diffLong = math.radians(pointB[1] - pointA[1])
	x = math.sin(diffLong) * math.cos(lat2)
	y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
			* math.cos(lat2) * math.cos(diffLong))
	initial_bearing = math.atan2(x, y)
	# Now we have the initial bearing but math.atan2 return values
	# from -180° to + 180° which is not what we want for a compass bearing
	# The solution is to normalize the initial bearing as shown below
	initial_bearing = math.degrees(initial_bearing)
	compass_bearing = (initial_bearing + 360) % 360
	return compass_bearing


from math import radians, cos, sin, asin, sqrt
def haversine(a_gps, b_gps):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lat1, lon1 = a_gps
    lat2, lon2 = b_gps
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    m = 6367000.0 * c
    return m

# Example 1: Look around in a circle!
dumps = []
hachiko = (35.6595104,139.7010075)
for heading in range(0,359,5):
    p = heading/4.0
    metadata = download_streetview_image(hachiko, filename="hachiko_" + str(heading), heading=heading, pitch=p, get_metadata=True)
    if metadata['status']=='OK':
        download_streetview_image(hachiko, filename="hachiko_" + str(heading), heading=heading, pitch=p)
    else:
        dumps += [metadata]

# Example 2: Walk along a street!
gr = googlemaps.Client(key=API_KEY_ROADS)
a_gps = (45.499931, -73.573001)
b_gps = (45.502009, -73.570964)
atob = interpolate_points(a_gps,b_gps,101)
for i,gps_point in enumerate(atob[:20]):
    heading = get_angle_between_points(gps_point, atob[i+1])
    download_streetview_image(gps_point, filename="stcats_" + str(i), heading=90-heading)

# Example 3: In Example 2, lots of those images are the same. Maybe interpolate with snap to roads is the better option?
# Nope, this only gets me about 12 unique points for this 3-block route, whereas the manual interpolation gets me about 30.
atob_wdata = gr.snap_to_roads([a_gps, b_gps], interpolate=True)
atob = [(g["location"]["latitude"], g["location"]["longitude"]) for g in atob_wdata]
for i,gps_point in enumerate(atob[:-1]):
    heading = get_angle_between_points(gps_point, atob[i+1])
    download_streetview_image(gps_point, filename="stcats_snap_" + str(i), heading=90-heading)


import time
p_home = (36.070847,140.115591)
p_ushiku = (35.982697,140.220276)
joshua_tree_a = (33.657139, -115.802200)
joshua_tree_b = (33.829343, -115.759485)
provincetown_west_end = (42.037576, -70.196374)
provincetown_east_end = (42.063885, -70.152947)
logan_airport = (42.365764, -71.010655)
central_park_s = (40.765810, -73.976188)
central_park_ne = (40.797063, -73.949645)
arizona_a = (33.610460, -113.628236)
arizona_b = (33.266353, -111.300382)
arizona_c = (32.630093, -109.015760)
ottawa = (45.423392, -75.698306)
montreal = (45.508964, -73.554354)
danforth_music_hall = (43.676533, -79.357132)
scenes_des_arbres = (45.515202, -73.535611)
# Getting directions to obtain a route
import polyline
gd = googlemaps.Client(key=API_KEY_DIRECTIONS)
directions_result = gd.directions(origin=p_home,destination=p_ushiku,mode="walking")
directions_result = gd.directions(origin=central_park_s,destination=central_park_ne,mode="bicycling")
directions_result = gd.directions(origin=provincetown_east_end,destination=provincetown_west_end,mode="driving")
directions_result = gd.directions(origin=arizona_b,destination=arizona_c,mode="driving")
directions_result = gd.directions(origin=ottawa,destination=montreal,mode="driving")

path_points = polyline.decode(directions_result[0]['overview_polyline']['points'])
look_points = [interpolate_points(pt[0],pt[1],hop_size=10) for pt in zip(path_points[:-1],path_points[1:])]
look_points = [item for sequence in look_points for item in sequence]
pt_diffs = [np.array(a)-np.array(b) for (a,b) in zip(look_points[:-1],look_points[1:])]
keepers = np.abs(np.array(pt_diffs))>0
look_points = [look_points[i] for i in range(len(keepers)) if np.any(keepers[i])]
look_points += look_points[-1:]*3
for i in range(len(look_points)-3):
for i in range(10,200):
    # for i in range(1000):
    gps_point = look_points[i]
    heading = calculate_initial_compass_bearing(gps_point, look_points[i+1])
    probe = download_streetview_image(gps_point, filename="" + str(i), heading=heading, size="640x640", get_metadata=True)
    # if probe['status']=="OK" and probe['copyright']==u'\xa9 Google, Inc.':
    if probe['status']=="OK" and 'Google' in probe['copyright']:
        download_streetview_image(gps_point, filename="ottawa_montreal" + str(i), heading=heading, size="640x640", get_metadata=False)

# if np.mod(i,150)==0:
    # time.sleep(5)

filestem = "arizona"
import subprocess
import os.system
def trim_repeats(filestem):
files = glob.glob("./photos/"+filestem+"*.jpg")
n = len(files)
prev_frame = files[0]
for curr_frame in files[1:]:
	result = os.system("diff " + curr_frame + " " + prev_frame)
	if result>0:
		os.system("mv " + curr_frame + " " + curr_frame+".txt")
	else:
		prev_frame - curr_frame


files = glob.glob("./photos/"+filestem+"*.jpg")
# for fil in files:
# 	new_fil = re.sub(".txt","",fil)
# 	os.system("mv " + fil + " " +new_fil)
n = len(files)
prev_frame = files[0]
for curr_frame in files[1:]:
	result = os.system("diff " + curr_frame + " " + prev_frame)
	if result>0: # new frame is different, reset basis
		prev_frame = curr_frame
	else:  # new frame is identical, move it to a text file
		os.system("mv " + curr_frame + " " + curr_frame+".txt")

ffmpeg -r 20 -f image2 -s 600x600 -i photos/ottawa_montreal%d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p ottawa_montreal.mp4

ffmpeg -r 20 -f image2 -s 600x600 -i photos/capecod%d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p capecod.mp4

ffmpeg -r 20 -f image2 -s 600x300 -i photos/joshua_tree%d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p joshua_tree.mp4


ffmpeg -r 25 -f image2 -s 600x300 -i photos/central_park%d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p central_park_bike_ride.mp4


# 
# Find alignment between two images in a path, and smooth between them:
# 

# Second image is a subset of the first.
# So, we resize the second image to be smaller, and we cut out the presumed center of the first image.
# (If we try to match im0 and im1 direactly, features like the road and sky will dominate and it will just say to overlay them as-is, generally.)
# These will be the same size, and we find the best alignment between them.
# In the future, this should be based on some knowledge of the true distance between the GPS points, and knowledge of the shared heading.

def pick_out_center(mat):
    x,y = mat.shape
    x4 = int(np.round(x/4.0))
    y4 = int(np.round(y/4.0))
    return mat[x4:3*x4,y4:3*y4]

def estimate_scale_and_shift(im0, im1):
    # Retrieve the middle half of im0:
    ar0 = pick_out_center(np.mean(np.array(im0),axis=2))
    # Shrinkg im1 to half its size:
    ar1 = np.mean(np.array(im1.resize(np.array(im1.size)/2)),axis=2)
    # Set strong expectation that angle will not change:
    # (Scale and horizontal and vertical transformations are what we will estimate.)
    result = ird.similarity(ar0, ar1, numiter=3, constraints={"angle":[0, .5]})
    # ,"tx":[0, 1], "ty":[0, 1]})
    scale = result["scale"]
    tvec = np.round(result["tvec"])
    # We originally scaled it by 1/2; we need to multiply this by scale.
    scale1to0 = scale/2.0
    # We also need to offset by 1/4 of the height and width of the original, plus tvec.
    shift1to0 = np.round(np.array(np.array(im0).shape[:2])/4.0) - tvec
    return scale1to0, shift1to0

# To generate an animated transition between im0 and im1, we want to generate intermediate frames according to the following diagram:
#    _______________
#   |\     im0     /| 
#   | \_ _ _ _ _ _/ |
#   |  |\  FRAME /| |
#   |    \______/   |
#   |  | |  im1 | | |
#   |    |______|   |
#   |  |/       \ | |
#   |  /_ _ _ _ _\  |
#   | /           \ |
#   |/_____________\|
# 
# im1 is the full-resolution im1, WxH for our test case.
# im0 has been blown up to fit: (WxH)/scale1to0
# We want to select the dotted box FRAME, of intermediate dimensions,
# and also with im1 alpha-overlaid on im0.

# im1 is a subset of im0
# So, we shrink im1 and search for the best alignment between it and im0.
# (In he future, the correct scale could be chosen directly from the known exact distance between GPS points.)

# Overlay im1 and a blown-up im0, using estimate scale and shift parameters:
def add_shrunk_image_inside(im0, im1, scale1to0, shift1to0, alpha=1):
    big_w, big_h = (np.array(im1.size)/scale1to0).astype(int)
    im_canvas = im0.resize((big_w, big_h))
    im_composite = im_canvas.copy()
    y_off, x_off = (shift1to0/scale1to0).astype(int)
    im_composite.paste(im1, [x_off, y_off])
    im_alpha = Image.blend(im_canvas, im_composite, alpha=alpha)
    return im_alpha

# Crop output frame from canvas:
def intermediate_zoom(im_canvas, im1, shift1to0, scale1to0, zoom=0.8):
    # Zoom goes from 0 (no zoom) to 1 (zoom in on im1 entirely)
    big_w, big_h = im_canvas.size
    small_w, small_h = im1.size
    y_off, x_off = shift1to0/scale1to0
    w = small_w + (big_w-small_w)*(1-zoom)
    h = small_h + (big_h-small_h)*(1-zoom)
    y = y_off*zoom
    x = x_off*zoom
    w,h,y,x = np.array((w,h,y,x)).astype(int)
    # Returns a copy of a rectangular region from the current image. The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
    im_crop = im_canvas.crop([x,y,x+w,y+h])
    im_crop = im_crop.resize((small_w, small_h))
    return im_crop

# Example 4: Zoom-transition between two frames
im0path = photo_folder + "carlton_60.jpg"
im1path = photo_folder + "carlton_61.jpg"
im0 = Image.open(im0path)
im1 = Image.open(im1path)
scale1to0, shift1to0 = estimate_scale_and_shift(im0, im1)
for i in range(11):
    im_alpha_canvas = add_shrunk_image_inside(im0, im1, scale1to0, shift1to0, alpha=0.1*i)
    im_crop = intermediate_zoom(im_alpha_canvas, im1, shift1to0, scale1to0, zoom=0.1*i)
    im_crop.save(photo_folder + "zoom_"+str(i)+".jpg")

# Example 5: Zoom between 10 frames
master_number = 0
for pic_number in range(50):
    print pic_number
    im0path = photo_folder + "carlton_" + str(pic_number+0) + ".jpg"
    im1path = photo_folder + "carlton_" + str(pic_number+1) + ".jpg"
    im0 = Image.open(im0path)
    im1 = Image.open(im1path)
    # scale1to0, shift1to0 = estimate_scale_and_shift(im0, im1)
    for i in range(4):
        im_alpha_canvas = add_shrunk_image_inside(im0, im1, scale1to0, shift1to0, alpha=0.25*i)
        im_crop = intermediate_zoom(im_alpha_canvas, im1, shift1to0, scale1to0, zoom=0.25*i)
        im_crop.save(photo_folder + "zoom_"+str(master_number).zfill(3)+".jpg")
        master_number+=1

# Add the last photo
im_alpha_canvas = add_shrunk_image_inside(im0, im1, scale1to0, shift1to0, alpha=1)
im_crop = intermediate_zoom(im_alpha_canvas, im1, shift1to0, scale1to0, zoom=1)
im_crop.save(photo_folder + "zoom_"+str(master_number).zfill(3)+".jpg")
# Convert the output to a movie:
subprocess.call(["ffmpeg", "-r", "20", "-f", "image2", "-s", "600x300", "-i", "photos/zoom_%03d.jpg", "-vcodec", "libx264", "-crf", "25", "-pix_fmt", "yuv420p", "test.mp4"])

# Example 6: Long, realistic example.
gr = googlemaps.Client(key=API_KEY_ROADS)
a_gps = (34.090818, -118.361490)
b_gps = (34.090923, -118.286259)
atob = interpolate_points(a_gps,b_gps,1000)
for i,gps_point in enumerate(atob[:-2]):
    heading = get_angle_between_points(gps_point, atob[i+1])
    download_streetview_image(gps_point, filename="santa_monica_"+str(i), savepath=myloc + "la/",heading=90-heading)

# Clean up list of points with calls to snap_to_roads to get rid of identical spots.
uniq_images = np.ones(len(atob))
for i in range(len(atob)-3):
    tmp1 = Image.open(myloc+"la/santa_monica_"+str(i)+".jpg")
    tmp2 = Image.open(myloc+"la/santa_monica_"+str(i+1)+".jpg")
    if tmp1==tmp2:
        uniq_images[i+1] = 0

# Load all images and create a profile 
tmp_image = Image.open(myloc+"la/santa_monica_"+str(0)+".jpg")
im_list = []
for i in range(997):
    if uniq_images[i]:
        im_list.append(Image.open(myloc+"la/santa_monica_"+str(i)+".jpg"))

ars = []
for im in im_list:
    ars.append(np.mean(np.array(im),axis=2))

# This takes a while
ar_stack = np.stack(ars,axis=2)
# Faster from here:
local_sim = np.zeros(len(ar_stack))
# Number of neighbours to the left and right:
n = 4
for i in range(len(ar_stack)):
    ind_range = np.arange(-n,n+1) + i
    ind_range[ind_range<0] += n*2+1
    ind_range[ind_range>=len(ar_stack)] -= n*2+1
    tmp_stack = np.stack([ar_stack[k] for k in ind_range if k != i],axis=2)
    tmp_template = np.median(tmp_stack,axis=2)
    tmp_template -= np.mean(tmp_template)
    correlation = tmp_template * (ar_stack[i]-np.mean(ar_stack[i]))
    local_sim[i] = np.sum(correlation>0)

    print ind_range, i
    ind_range += 
template = np.mean(ar_stack,axis=2)
template -= np.mean(template)
cors = [template * (np.mean(im,axis=2)-np.mean(im)) for im in im_list]
corvals = [np.sum(ar>0) for ar in cors]


keep_points = np.zeros(len(atob))
atob_wdata = gr.snap_to_roads([atob], interpolate=True)

for i in range(len(keep_points)-1):
a_gps = atob[i]
b_gps = atob[i+1]
print "Making snap_to_roads call #" + str(i) + "..."
atob_wdata = gr.snap_to_roads([a_gps, b_gps], interpolate=True)
    if atob_wdata[1]["placeId"] != atob_wdata[0]["placeId"]:
        kkep_points[i] = 1

atob = [(g["location"]["latitude"], g["location"]["longitude"]) for g in atob_wdata]

# Lots of these pictures will be doubles, so watch out for them!
for i in range(len(atob)):
    

master_number = 0
for pic_number in range(50):
    print pic_number
    im0path = myloc + "la/santa_monica_" + str(pic_number+0) + ".jpg"
    im1path = photo_folder + "carlton_" + str(pic_number+1) + ".jpg"
    im0 = Image.open(im0path)
    im1 = Image.open(im1path)
    # scale1to0, shift1to0 = estimate_scale_and_shift(im0, im1)
    for i in range(4):
        im_alpha_canvas = add_shrunk_image_inside(im0, im1, scale1to0, shift1to0, alpha=0.25*i)
        im_crop = intermediate_zoom(im_alpha_canvas, im1, shift1to0, scale1to0, zoom=0.25*i)
        im_crop.save(photo_folder + "zoom_"+str(master_number).zfill(3)+".jpg")
        master_number+=1





ffmpeg -r 20 -f image2 -s 600x300 -i zoom_%03d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4


ffmpeg -r 20 -f image2 -s 600x300 -i la/santa_monica_%1d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p sm_video.mp4


ffmpeg -r 20 -f image2 -s 600x300 -i photos/road_to_ushiku_%d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p ushiku.mp4




# Let's try correlation instead. Slower, but probably better.
from scipy import signal
c2 = signal.correlate2d(ar0, ar1)
np.unravel_index(np.argmax(c2),c2.shape)

dp = np.dot(ar0.transpose(),ar1)
bo0 = np.argmax(np.sum(dp,axis=0))
bo1 = np.argmax(np.sum(dp,axis=1))
result = ird.similarity(np.mean(im0,axis=2), np.mean(im2,axis=2), numiter=3)
