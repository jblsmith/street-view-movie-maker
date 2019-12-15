# Script for Hollerado - Don't Shake project
# 
# The idea is to trace a path from The Barfly in Montreal to
# The Danforth Music Hall in Toronto, and to synchronize
# the video with the song "Don't Shake" by Hollerado.
# 
# This script is just intended to be an illustration of
# the workflow, and a record of the scripting I did.
# It won't reproduce exactly what I did to generate the 
# video, because lots of unreplicable work was done
# (e.g., probing various points along the route to see what
# data was available, and basing future design decisions
# on that).



#
#
#       Query Google for route and set up pandas dataframe of GPS points
#
#

# Imports
from utils import *
from API_KEYS import API_KEY_DIRECTIONS, API_KEY_STREETVIEW
import pickle

# Point A and point B:
barfly = (45.517146, -73.579837)
danforth = (43.676533,-79.357132)

# Get the route
gd = googlemaps.Client(key=API_KEY_DIRECTIONS)
directions_result = gd.directions(origin=barfly, destination=danforth, mode="driving")
# Save directions (commented out to not accidentally overwrite!)
# with open("barfly_to_danforth_route.p", "wb") as f:
#     pickle.dump(directions_result, f)
# Load directions
with open("barfly_to_danforth_route.p", "rb") as f:
	directions_result = pickle.load(f)

# Decode polyline (the directions) into dense sequence of GPS points
path_points = polyline.decode(directions_result[0]['overview_polyline']['points'])
dense_points = [interpolate_points(pt[0],pt[1],hop_size=1) for pt in zip(path_points[:-1],path_points[1:])]
look_points_rough = [item for sequence in dense_points for item in sequence]
# Remove unnecessary points
look_points = clean_look_points(look_points_rough)
# Create an itinerary object and probe a 1000th of the frames:
pickle_filename = "bd_points.p"
# Save
# Don't accidentally recreate it! You only want to create this dataframe once.
# itin_bd = create_itinerary_df(look_points)
# Load
itin_bd = pd.read_pickle(pickle_filename)

# Probe a subset of the path
take_these_steps = range(0,itin_bd.shape[0],1000)   # Do a subset of every 1000th point?
take_these_steps = range(0,itin_bd.shape[0],10)     # Or every 10th point?
# Google can get angry at you if you probe too much. Maybe you're trying to copy their database! Ha.
# So, it's prudent to probe different subsets at intervals, and to save the output after you've
# probed a significant new chunk to make sure you don't lose data.
probe_itinerary_items(itin_bd, take_these_steps, API_KEY_STREETVIEW)
# Save your work:
# itin_bd.to_pickle(pickle_filename)

# Convert date field to numerical year and month columns
years = [date[:4] for date in itin_bd.date]
years_int = [int(y) if len(y) else 0 for y in years]
months = [date[5:] for date in itin_bd.date]
months_int = [int(y) if len(y) else 0 for y in months]
itin_bd["year"] = years_int
itin_bd["month"] = months_int
# Add distance in months from most common month of most common year (which turned out to be 2018-07)
months_before = (2018-itin_bd.year)*12 + (7-itin_bd.month)
months_after  = (itin_bd.year-2018)*12 + (itin_bd.month-7)
itin_bd["dist_from_2018-07"] = np.max((months_before, months_after), axis=0)



#
#
#       Information about song to be matched
#
#

# The following array was estimated using madmom from the audio file.
# That work isn't replicated here. Just the output.
beats = np.array([  0.25,   0.86,   1.52,   2.14,   2.81,   3.42,   4.08,   4.69,
         5.36,   5.96,   6.62,   7.24,   7.91,   8.52,   9.18,   9.8 ,
        10.47,  11.07,  11.73,  12.35,  13.02,  13.63,  14.29,  14.91,
        15.58,  16.19,  16.85,  17.47,  18.14,  18.74,  19.4 ,  20.01,
        20.66,  21.29,  21.94,  22.57,  23.22,  23.84,  24.5 ,  25.11,
        25.77,  26.39,  27.05,  27.68,  28.32,  28.95,  29.6 ,  30.22,
        30.88,  31.5 ,  32.15,  32.78,  33.43,  34.05,  34.71,  35.33,
        35.98,  36.61,  37.26,  37.89,  38.54,  39.16,  39.81,  40.45,
        41.09,  41.72,  42.37,  43.  ,  43.64,  44.28,  44.92,  45.55,
        46.2 ,  46.83,  47.47,  48.11,  48.75,  49.38,  50.02,  50.66,
        51.3 ,  51.94,  52.58,  53.22,  53.86,  54.49,  55.13,  55.77,
        56.41,  57.04,  57.67,  58.32,  58.96,  59.59,  60.25,  60.86,
        61.51,  62.14,  62.79,  63.42,  64.07,  64.69,  65.35,  65.97,
        66.62,  67.25,  67.9 ,  68.52,  69.18,  69.79,  70.45,  71.09,
        71.73,  72.36,  73.  ,  73.63,  74.28,  74.91,  75.56,  76.19,
        76.83,  77.46,  78.11,  78.75,  79.39,  80.02,  80.66,  81.3 ,
        81.94,  82.57,  83.21,  83.85,  84.49,  85.12,  85.77,  86.4 ,
        87.05,  87.67,  88.32,  88.95,  89.6 ,  90.23,  90.88,  91.51,
        92.15,  92.79,  93.42,  94.06,  94.71,  95.34,  95.98,  96.62,
        97.26,  97.9 ,  98.54,  99.17,  99.81, 100.45, 101.09, 101.72,
       102.37, 103.  , 103.64, 104.28, 104.92, 105.56, 106.19, 106.83,
       107.47, 108.11, 108.74, 109.39, 110.03, 110.65, 111.3 , 111.94,
       112.58, 113.21, 113.86, 114.49, 115.13, 115.76, 116.41, 117.05,
       117.68, 118.32, 118.96, 119.59, 120.24, 120.87, 121.51, 122.15,
       122.79, 123.43, 124.07, 124.7 , 125.35, 125.98, 126.62, 127.26,
       127.9 , 128.53, 129.18, 129.81, 130.45, 131.08, 131.72, 132.36,
       133.  , 133.64, 134.28, 134.91, 135.56, 136.19, 136.83, 137.46,
       138.11, 138.74, 139.39, 140.02, 140.67, 141.3 , 141.94, 142.57,
       143.22, 143.85, 144.49, 145.13, 145.77, 146.41, 147.05, 147.68,
       148.32, 148.95, 149.62, 150.23, 150.9 , 151.51, 152.17, 152.78,
       153.43, 154.07, 154.71, 155.34, 155.98, 156.62, 157.26, 157.89,
       158.54, 159.16, 159.81, 160.45, 161.09, 161.73, 162.36, 163.  ,
       163.64, 164.28, 164.92, 165.56, 166.2 , 166.83, 167.47, 168.11,
       168.75, 169.38, 170.02, 170.66, 171.3 , 171.94, 172.58, 173.21,
       173.86, 174.49, 175.12, 175.77, 176.41, 177.04, 177.68, 178.32,
       178.96, 179.6 , 180.24, 180.88, 181.52, 182.14, 182.8 , 183.42,
       184.07, 184.7 , 185.34, 185.98, 186.62, 187.25, 187.9 , 188.53,
       189.18, 189.81, 190.45, 191.08, 191.73, 192.36, 193.  , 193.63,
       194.28, 194.91, 195.56, 196.19, 196.83, 197.47, 198.11, 198.74,
       199.39, 200.02, 200.67, 201.3 , 201.94, 202.57, 203.22, 203.84,
       204.49, 205.11, 205.77, 206.39])

tmp_half_beats = beats[:-1] + (beats[1:] - beats[:-1])/2
halfbeats = np.array(sorted(list(tmp_half_beats) + list(beats)))

# Define a timeline object so that we can easily generate a movie
# that aligns to points in the music.

class timeline(object):
    def __init__(self, duration, fps=24, new_stem="default_stem", base_path="./photos"):
        self.timeline = pd.DataFrame(columns=['time','beatindex','filename'])
        self.timeline.time = np.arange(0,duration,1.0/fps)
        self.timeline.beatindex = np.zeros(self.timeline.shape[0]).astype(int)
        self.timeline = self.timeline.fillna("")
        self.fps = fps
        self.new_stem = new_stem
        self.base_path = base_path
    
    def set_beat_indices(self, beat_times_seconds):
        nearest_frames_to_beats = [np.argmin(np.abs(self.timeline.time - b)) for b in beat_times_seconds]
        cumulative_beat_index = np.zeros((self.timeline.shape[0],))
        cumulative_beat_index[nearest_frames_to_beats] = 1
        self.timeline["beatindex"] = np.cumsum(cumulative_beat_index).astype(int)
    
    def set_pic_to_beat(self, pic_filename, beat1, beat2=None):
        if beat2 is None:
            beat2 = beat1 + 1
        self.timeline["filename"][(self.timeline.beatindex>=beat1) & (self.timeline.beatindex<beat2)] = pic_filename
    
    def set_continuous_pics_from_beat(self, pic_filenames, beat1, beat2):
        start_index = self.timeline.index[self.timeline.beatindex==beat1].values[0]
        end_index = self.timeline.index[self.timeline.beatindex==beat2].values[0]
        range_len = end_index-start_index
        self.timeline['filename'][start_index:end_index] = pic_filenames[:range_len]
        return range_len
    
    def copy_images_in_timeline(self):
        for ind in self.timeline.index:
            old_filename = self.timeline.loc[ind]['filename']
            new_filename = "{0}/{1}{2}.jpg".format(self.base_path, self.new_stem, ind)
            if old_filename is not '':
                print("{0} {1} {2}".format("cp", old_filename, new_filename))
                os.system("{0} {1} {2}".format("cp", old_filename, new_filename))
    
    def script_make_video(self):
        video_filename = self.new_stem + "vid"
        sound_filename = self.new_stem + "vid_sound"
        make_video(self.new_stem, rate=self.fps, video_string=video_filename, picsize="640x640", basepath=self.base_path)
        os.system("ffmpeg -i {0}.mp4 -i \"/Users/jordan/Music/iTunes/iTunes Music/Hollerado/Born Yesterday/02 Don't Shake.wav\" -shortest {1}.mp4 -y".format(video_filename, sound_filename))
        print("Video should have been successfully made here: {0}".format(sound_filename))


def download_missing_items_for_timeline(timeline_obj, itinerary, stem="bd_1000s"):
    missing_ids = []
    for fn in timeline_obj.timeline["filename"].values:
        # Check if it exists.
        if not os.path.exists(fn):
            folder = os.path.dirname(fn)
            filename = os.path.basename(fn)
            basename, ext = os.path.splitext(filename)
            index = int(basename.split(stem)[1])
            missing_ids += [index]
    print("For this route, there are {0} images to download.".format(len(missing_ids)))
    continue_opt = raw_input('Would you like to download them all Type yes to proceed; otherwise, program halts.\n')
    if continue_opt not in ['Yes','yes']:
        return
    download_pics_from_list(itinerary, API_KEY_STREETVIEW, stem, "640x640", redownload=False, index_filter=missing_ids)
    return itinerary

# Construct a plan for the song, deciding, for each range of sub-beats (2 ticks per beat),
# whether to assign one picture for the entire beat or one picture per frame of video.
# set a "pace" to skip more pictures at a time
def define_program():
    pic_per_4_beats = []
    pic_per_2_beats = []
    pic_per_1_beat = []
    pic_per_1_frame = []
    pace = {}
    # verse 1, first half = 0, 16
    pic_per_2_beats += [0, 14]
    pic_per_4_beats += range(2,14,4)
    pic_per_2_beats += range(16,32,2)
    # verse 1, second half = 32, 64
    pic_per_1_beat  += range(32,64)
    # verse 2 first half
    pic_per_1_frame += range(64,68)
    pic_per_2_beats += range(68,72,2)
    pic_per_1_frame += range(72,76)
    pic_per_1_beat  += range(76,80)
    pic_per_1_frame += range(80,84)
    pic_per_2_beats += range(84,88,2)
    pic_per_1_frame += range(88,92)
    pic_per_1_beat  += range(92,96)
    # verse 2 second half
    pic_per_1_frame += range(96, 100)
    pic_per_2_beats += range(100, 104, 2)
    pic_per_1_frame += range(104, 108)
    pic_per_1_beat  += range(108, 112)
    pic_per_1_frame += range(112, 116)
    pic_per_2_beats += range(116, 120, 2)
    pic_per_1_frame += range(120, 128)
    pace[128] = 2
    # chorus = 128 to 192
    pic_per_1_frame += range(128,192)
    pace[192] = 1
    # interlude
    pic_per_1_frame += range(192,194)
    pic_per_4_beats += [194]
    pic_per_2_beats += [197,199]
    pic_per_1_frame += range(200,208)
    pic_per_1_frame += range(208,210)
    pic_per_4_beats += [210]
    pic_per_2_beats += [213,215]
    pic_per_1_frame += range(216,220)
    pic_per_4_beats += [220]
    # verse 3
    pic_per_1_frame += range(224,288)
    pace[288] = 2
    # chorus 2
    pic_per_1_frame += range(288,352)
    pace[352] = 1
    # interlude 2
    pic_per_1_frame += range(352,400)
    # bridge -- accelerate until whatever point is necessary to get to Toronto by the end
    pic_per_1_frame += range(400,464)
    pace[400] = 8
    # interlude 3
    # super quiet, ritard.: (x = new pic, X = hard on this 16th) [xxxxx.x.x...x.XX], off the 401 onto local Toronto highway
    pic_per_2_beats += range(464,468)
    pic_per_4_beats += [468,470,472]
    pic_per_2_beats += [476]
    pic_per_1_frame += range(478,480)
    pace[478] = 2
    # chorus 3, 1.5 times longer
    pic_per_1_frame += range(480,576)
    pace[480] = 2
    pace[572] = 1
    # outro
    pic_per_1_frame += range(576,640)
    # extra snaps
    pic_per_4_beats += [640,644]
    # At these points, leap a little further in the series of points.
    leap_onsets = [194,197,199,200,210,213,215,216,220,224,464,466,468,470,472,474,476,478]
    return pic_per_4_beats, pic_per_2_beats, pic_per_1_beat, pic_per_1_frame, leap_onsets, pace



#
#
#       Generate music video
#
#

use_this_bd = itin_bd.loc[itin_bd.status=="OK"]
pano_id_changes = use_this_bd.pano_id.values[:-1] != use_this_bd.pano_id.values[1:]
unique_ids = use_this_bd.index[np.where(pano_id_changes)[0]]
spaced_ids = list(unique_ids[range(0,100,2)]) + list(unique_ids[range(100,len(unique_ids),3)])
itinerary_ids = spaced_ids

# Set up timeline object. Give it the beats. Get plan.
tl = timeline(208, 24)
tl.set_beat_indices([0] + halfbeats[1:])
# Start at picture 0. For each click of the timeline, assign the next filenames.
eligible_pic_ind = 0
pic_per_4_beats, pic_per_2_beats, pic_per_1_beat, pic_per_1_frame, leap_onsets, pace = define_program()
current_pace = 1

# PHASE 1:
# Assign all the pictures
for beat_i in range(0,647):
    if beat_i == 480:
        eligible_pic_ind_at_480 = eligible_pic_ind
    if np.mod(beat_i,10)==0:
        print("Beat  {0}/646 ... pic_ind {1}/13564".format(beat_i, eligible_pic_ind))
    if beat_i in pace.keys():
        current_pace = pace[beat_i]
    if beat_i in leap_onsets:
        eligible_pic_ind += 50
    if beat_i in pic_per_2_beats:
        tl.set_pic_to_beat("./photos/bd_1000s{0}.jpg".format(itinerary_ids[eligible_pic_ind]), beat_i, beat_i+2)
        eligible_pic_ind += 1*current_pace
    elif beat_i in pic_per_4_beats:
        tl.set_pic_to_beat("./photos/bd_1000s{0}.jpg".format(itinerary_ids[eligible_pic_ind]), beat_i, beat_i+4)
        eligible_pic_ind += 1*current_pace
    elif beat_i in pic_per_1_beat:
        tl.set_pic_to_beat("./photos/bd_1000s{0}.jpg".format(itinerary_ids[eligible_pic_ind]), beat_i, beat_i+1)
        eligible_pic_ind += 1*current_pace
    elif beat_i in pic_per_1_frame:
        pic_filenames = ["./photos/bd_1000s{0}.jpg".format(itinerary_ids[ep_ind]) for ep_ind in range(eligible_pic_ind,len(itinerary_ids))]
        pace_pic_filenames = [pic_filenames[i] for i in range(0,len(pic_filenames),current_pace)]
        range_len = tl.set_continuous_pics_from_beat(pace_pic_filenames, beat_i, beat_i+1)
        eligible_pic_ind = eligible_pic_ind + range_len*current_pace

eligible_pic_ind_at_647 = eligible_pic_ind
# At beat 480, we want to make sure we are on track to finish at the end perfectly.
# So, we record the value of eligible_pic_ind at 480, and again at the end, and then
# re-run the above loop after resetting eligible_pic_ind forward sufficiently.
epis_needed_for_480_onward = eligible_pic_ind_at_647 - eligible_pic_ind_at_480
max_epi = len(itinerary_ids)
id_to_set_at_480 = max_epi - epis_needed_for_480_onward
# >>> eligible_pic_ind_at_480
# 8175
# >>> eligible_pic_ind_at_647
# 10108
# >>> id_to_set_at_480
# 11631

eligible_pic_ind = id_to_set_at_480
for beat_i in range(480,647):
    if np.mod(beat_i,10)==0:
        print("Beat  {0}/646 ... pic_ind {1}/13564".format(beat_i, eligible_pic_ind))
    if beat_i in pace.keys():
        current_pace = pace[beat_i]
    if beat_i in leap_onsets:
        eligible_pic_ind += 50
    if beat_i in pic_per_2_beats:
        tl.set_pic_to_beat("./photos/bd_1000s{0}.jpg".format(itinerary_ids[eligible_pic_ind]), beat_i, beat_i+2)
        eligible_pic_ind += 1*current_pace
    elif beat_i in pic_per_4_beats:
        tl.set_pic_to_beat("./photos/bd_1000s{0}.jpg".format(itinerary_ids[eligible_pic_ind]), beat_i, beat_i+4)
        eligible_pic_ind += 1*current_pace
    elif beat_i in pic_per_1_beat:
        tl.set_pic_to_beat("./photos/bd_1000s{0}.jpg".format(itinerary_ids[eligible_pic_ind]), beat_i, beat_i+1)
        eligible_pic_ind += 1*current_pace
    elif beat_i in pic_per_1_frame:
        pic_filenames = ["./photos/bd_1000s{0}.jpg".format(itinerary_ids[ep_ind]) for ep_ind in range(eligible_pic_ind,len(itinerary_ids))]
        pace_pic_filenames = [pic_filenames[i] for i in range(0,len(pic_filenames),current_pace)]
        range_len = tl.set_continuous_pics_from_beat(pace_pic_filenames, beat_i, beat_i+1)
        eligible_pic_ind = eligible_pic_ind + range_len*current_pace


# Final steps: make sure all the pics are downloaded,
# copy pictures into proper sequence,
# then make the video using ffmpeg.
itin_bd_copy = download_missing_items_for_timeline(tl, itin_bd, stem="bd_1000s")
tl.copy_images_in_timeline()
tl.script_make_video()

# Preserve output!
itin_bd.to_pickle("new_pickled_filename.p")
