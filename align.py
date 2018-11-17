import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch

import librosa
import librosa.display
from dlnco.DLNCO import dlnco
from parser.txt_parser import sv_score_parser
from operator import itemgetter

n_fft = 4096
hopsize = int(n_fft/4)
merge_dlnco = True
y_t, sr_t = librosa.load("./examples/KissTheRain_2_t_short.wav")
y_s, sr_s = librosa.load("./examples/KissTheRain_2_s_short.wav")

# wav length in time
len_y_t = len(y_t)/sr_t
len_y_s = len(y_s)/sr_s

# chroma and DLNCO features
y_t_chroma = librosa.feature.chroma_stft(y=y_t, sr=sr_t, tuning=0, norm=2,
                                         hop_length=hopsize, n_fft=n_fft)
y_t_dlnco = dlnco(y_t, sr_t, n_fft)

y_s_chroma = librosa.feature.chroma_stft(y=y_s, sr=sr_s, tuning=0, norm=2,
                                         hop_length=hopsize, n_fft=n_fft)
y_s_dlnco = dlnco(y_s, sr_s, n_fft)

# plt.figure(figsize=(16, 8))
# plt.subplot(2, 1, 1)
# plt.title('Chroma Representation of $X_s$')
# librosa.display.specshow(y_s_chroma, x_axis='time',
#                          y_axis='chroma', cmap='gray_r', hop_length=hopsize)
# plt.colorbar()
# plt.subplot(2, 1, 2)
# plt.title('Chroma Representation of $X_s$')
# librosa.display.specshow(y_s_dlnco, x_axis='time',
#                          y_axis='chroma', cmap='gray_r', hop_length=hopsize)
# plt.colorbar()
# plt.tight_layout()
# plt.show()

y_t_merge = y_t_chroma + y_t_dlnco if merge_dlnco else y_t_chroma
y_s_merge = y_s_chroma + y_s_dlnco if merge_dlnco else y_s_chroma

D, wp = librosa.sequence.dtw(X=y_t_merge, Y=y_s_merge, metric='cosine') # D (153, 307)
wp_s = np.asarray(wp) * hopsize / sr_t # (330, 2) wrapping path

# print(wp_s[:, 1])
# print(wp_s[:, 0])

# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111)
# librosa.display.specshow(D, x_axis='time', y_axis='time',
#                          cmap='gray_r', hop_length=hopsize)
# imax = ax.imshow(D, cmap=plt.get_cmap('gray_r'),
#                  origin='lower', interpolation='nearest', aspect='auto')
# ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
# plt.title('Warping Path on Acc. Cost Matrix $D$')
# plt.colorbar()
# plt.tight_layout()
# plt.show()

# load score of sonic visualizer
score_t = sv_score_parser("./examples/KissTheRain_2_t_short.txt")
score_s = sv_score_parser("./examples/KissTheRain_2_s_short.txt")

# use wrapping path to align the notes
# step1: sort the overlapping note area from high to low
# step2: abs(note_t - note_s) <= 2
wp_stu = wp_s[:, 1][::-1]
wp_t = wp_s[:, 0][::-1]

list_score_aligned = []
non_aligned_score_s = []
for note_s in score_s: # search teacher's corresponding notes using student's notes
    found_s = False # bool student's note found or not
    idx_note_start = np.argmin(np.abs(wp_stu - note_s[0]))
    idx_note_end = np.argmin(np.abs(wp_stu - (note_s[0] + note_s[2])))
    time_note_start_t = wp_t[idx_note_start] # note time start corresponding in teacher's wrapping time
    time_note_end_t = wp_t[idx_note_end] # note time end
    
    list_dur_pitch_t = [] # calculate overlapping area and note difference
    for ii_note_t, note_t in enumerate(score_t):
        i0 = max(time_note_start_t, note_t[0])
        i1 = min(time_note_end_t, note_t[0] + note_t[2])
        diff_dur = i1-i0 if i0 < i1 else 0.0
        diff_pitch = np.abs(note_t[1] - note_s[1])
        list_dur_pitch_t.append([diff_dur, diff_pitch, ii_note_t])
    list_dur_pitch_t = sorted(list_dur_pitch_t, key=itemgetter(0), reverse=True)
    
    for ldp_t in list_dur_pitch_t:
        if ldp_t[0] > 0 and ldp_t[1] <=2: # find the most overlapped note and pitch diff <= 2
            list_score_aligned.append([score_t[ldp_t[2]], note_s])
            # print(len(score_t))
            score_t.pop(ldp_t[2])
            # print(len(score_t))
            found_s = True
            break

    if not found_s:
        non_aligned_score_s.append(note_s)

if len(score_t):
    for st in score_t:
        list_score_aligned.append([st, []])

if len(non_aligned_score_s):
    for ss in non_aligned_score_s:
        list_score_aligned.append([[], ss])

# plot the alignment, red aligned notes, black extra or missing notes
f, (ax1, ax2) = plt.subplots(2, 1)
for note_pair in list_score_aligned:
    if len(note_pair[0]) and len(note_pair[1]):
        face_color = 'r'
    elif len(note_pair[0]):
        face_color = 'k'
    else:
        continue
    rect = patches.Rectangle((note_pair[0][0], note_pair[0][1]-0.5), note_pair[0][2], 1.0, linewidth=1,edgecolor=face_color,facecolor=face_color)
    ax1.add_patch(rect)
    
ax1.set_ylabel('Teacher')
ax1.set_xlim(0, len_y_t)
ax1.set_ylim(0, 88)

for note_pair in list_score_aligned:
    if len(note_pair[0]) and len(note_pair[1]):
        face_color = 'r'
        con = ConnectionPatch(xyA=(note_pair[1][0], note_pair[1][1]-0.5), xyB=(note_pair[0][0], note_pair[0][1]-0.5), coordsA="data", coordsB="data",
                      axesA=ax2, axesB=ax1, color="b")
        ax2.add_artist(con)
    elif len(note_pair[1]):
        face_color = 'k'
    else:
        continue        
    rect = patches.Rectangle((note_pair[1][0], note_pair[1][1]-0.5), note_pair[1][2], 1.0, linewidth=1,edgecolor=face_color,facecolor=face_color)
    ax2.add_patch(rect)

ax2.set_ylabel('Student')
ax2.set_xlim(0, len_y_s)
ax2.set_ylim(0, 88)
ax2.set_xlabel('time (s)')

# plt.tight_layout()
plt.show()