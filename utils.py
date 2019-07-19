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
		return savepath+filename+fi

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


