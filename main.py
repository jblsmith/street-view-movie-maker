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
plt.ion()
myloc = "/Users/jordan/Documents/repositories/street-view-animation/"
photo_folder = myloc + "photos/"

# Adapted directly from Andrew Wheeler:
# https://andrewpwheeler.wordpress.com/2015/12/28/using-python-to-grab-google-street-view-imagery/
# Usage example:
# >>> download_streetview_image((46.414382,10.012988))
def download_streetview_image(lat_lon, filename="image", savepath=photo_folder, size="600x300", heading=151.78, pitch=-0.76, fi=".jpg", address=None):
	base = "https://maps.googleapis.com/maps/api/streetview"
	if address is None:
		address = str(lat_lon[0]) + "," + str(lat_lon[1])
	url = base + "?size=" + size + "&location=" + address + "&heading=" + str(heading) + "&pitch=" + str(pitch) + "&key=" + API_KEY_STREETVIEW
	print url
	urllib.urlretrieve(url, savepath+filename+fi)

# Given two GPS points (lat/lon), interpolate a sequence of GPS points in a straight line
def interpolate_points(a_gps,b_gps,n_points=11):
	x = np.linspace(a_gps[0],b_gps[0],n_points)
	y = np.linspace(a_gps[1],b_gps[1],n_points)
	return zip(x,y)

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


# Example 1: Look around in a circle!
hachiko = (35.6595104,139.7010075)
for heading in range(0,359):
	download_streetview_image(hachiko, filename="hachiko_" + str(heading), heading=heading)

# Example 2: Walk along a street!
gr = googlemaps.Client(key=API_KEY_ROADS)
a_gps = (45.499931, -73.573001)
b_gps = (45.502009, -73.570964)
atob = interpolate_points(a_gps,b_gps,101)
for i,gps_point in enumerate(atob[:-2]):
	heading = get_angle_between_points(gps_point, atob[i+1])
	download_streetview_image(gps_point, filename="stcats_" + str(i), heading=90-heading)

# Example 3: In Example 2, lots of those images are the same. Maybe interpolate with snap to roads is the better option?
# Nope, this only gets me about 12 unique points for this 3-block route, whereas the manual interpolation gets me about 30.
atob_wdata = gr.snap_to_roads([a_gps, b_gps], interpolate=True)
atob = [(g["location"]["latitude"], g["location"]["longitude"]) for g in atob_wdata]
for i,gps_point in enumerate(atob[:-1]):
	heading = get_angle_between_points(gps_point, atob[i+1])
	download_streetview_image(gps_point, filename="stcats_snap_" + str(i), heading=90-heading)

