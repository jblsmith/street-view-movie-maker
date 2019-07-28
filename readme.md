# Google Street View Movie Maker

It makes movies out of Google Street View images!

You provide point A and point B. It uses the Google Roads API to get directions from A to B, then repeatedly looks for Street View images along that path, and then converts them into a movie.

E.g., [here is a video](https://www.youtube.com/watch?v=puzhsLtn8AQ) taken along the Copacabana beach in Rio, with a little snippet of Senor Coconut's "Neon Lights (Cha Cha Cha)" added for fun.

![](copacabana.jpg?raw=true)

## Requirements

To run this code yourself takes some fussing around:

1. Make sure [FFMPEG](https://ffmpeg.org/) is installed and callable from the command line. I brew-installed it.
2. pip-install this project's python [requirements](requirements.txt). The only packages you may not have are [googlemaps](https://pypi.org/project/googlemaps/) and [polyline](https://pypi.org/project/polyline/). Note: this project is written for Python 2.
3. Lastly, and trickiest of all: the code requires API keys for Google Map's [Street View](https://developers.google.com/maps/documentation/streetview/get-api-key) and [Directions](https://developers.google.com/maps/documentation/directions/get-api-key) APIs. Note: setting up the Street View API now requires a billing account! it tends to be free for small amounts of traffic, but you have to set it up anyway.

## Project history

When using Google Maps to plan a route I haven't driven or walked before, I always though it would be nifty to be able to preview the directions as a video. Obviously, you can check out the route by looking at Street View at random points, or navigating in Street View mode itself. But these options are tedious!

Other people have had the same idea; at least two have created web services to do it:

- [Streetview Player](http://brianfolts.com/driver/)
- [Route View](http://routeview.org/VirtualRide/)

In each case, the user gives points A and B, the site fetches directions, picks a set of waypoints about 150m apart, and shows you the image at each waypoint. But the sites don't download the images and compile them into a movie; they just repeatedly reload Google Street View at each waypoint. I wanted to obtain a simulated video preview.

Also, both services are basically defunct now that using the Street View API requires billing. (I last checked the functional sites in 2017.)

## Project hurdles

Step 1 was downloading an image. I started with [the Street View API documentation](https://developers.google.com/maps/documentation/streetview/intro), but to get it working in Python, I used [this blog post](https://andrewpwheeler.wordpress.com/2015/12/28/using-python-to-grab-google-street-view-imagery/) as a reference.

Step 2 was computing the route and selecting GPS points along that route. Using the Directions API was straightforward with the [Python Client for Google Maps Services](https://github.com/googlemaps/google-maps-services-python).

Step 3 was computing the correct heading (compass direction) from A to B, which is really tricky! I ultimately found [someone else's code](https://gist.github.com/jeromer/2005586) to do this.

But there were lots of failed hacks in between. The math for computing distances and angles on spheres is very cool---the [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula), but I probably would have enjoyed learning about it more in high school. The best among them used the GeoPy package, but was ugly: with [GeoPy](https://geopy.readthedocs.io/), I couldn't compute the heading from A to B. But, given point A, a bearing, and a distance, I could compute a destination C. I could also compute the distance between any two points. So, I computed 360 potential destinations, each a degree apart and a fixed distance from A, and then found the one that was nearest to C, which gave the approximate heading.

Step 4 was concatenating the images into a movie, for which [FFMPEG](https://ffmpeg.org/) is indispensible!
