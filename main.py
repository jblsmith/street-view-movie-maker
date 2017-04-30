from API_KEYS import *
import googlemaps
from datetime import datetime
import urllib, os

myloc = "/Users/jordan/Documents/repositories/street-view-animation/"
photo_folder = myloc + "photos/"

# Adapted directly from Andrew Wheeler:
# https://andrewpwheeler.wordpress.com/2015/12/28/using-python-to-grab-google-street-view-imagery/
def download_streetview_image(lat_lon, filename="image", savepath=photo_folder, size="600x300", heading=151.78, pitch=-0.76, fi=".jpg", address=None):
	base = "https://maps.googleapis.com/maps/api/streetview"
	if address is None:
		address = str(lat_lon[0]) + "," + str(lat_lon[1])
	url = base + "?size=" + size + "&location=" + address + "&heading=" + str(heading) + "&pitch=" + str(pitch) + "&key=" + API_KEY_STREETVIEW
	print url
	urllib.urlretrieve(url, savepath+filename+fi)

download_streetview_image((46.414382,10.013988))
