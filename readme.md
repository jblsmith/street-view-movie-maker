# Google Street View Movie Maker

It makes movies out of Google Street View images!

You provide point A and point B. It uses the Google Roads API to get directions from A to B, then repeatedly looks for Street View images along that path, and then converts them into a movie.

E.g., here's a video taken along the Copacabana beach in Rio (with a little snippet of Senor Coconut's "Neon Lights (Cha Cha Cha)" added for fun):

<iframe width="560" height="315" src="https://www.youtube.com/embed/puzhsLtn8AQ?rel=0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Requirements: 

To run this code yourself takes some fussing around:

1. Make sure [FFMPEG](https://ffmpeg.org/) is installed and callable from the command line. I brew-installed it.
2. pip-install this project's python requirements. The only packages you may not have are googlemaps and polyline. Note: the project is written for Python 2.
3. Lastly, and trickiest of all: the code requires API keys for Google Map's [Street View](https://developers.google.com/maps/documentation/streetview/get-api-key) and [Directions](https://developers.google.com/maps/documentation/directions/get-api-key) APIs. Note: setting up the Street View API now requires a billing account! it tends to be free for small amounts of traffic, but you have to set it up anyway.
