## -------------------------- Modules ------------------------------ ##
import numpy as np
import matplotlib
import os
import h5py
import math
import numpy as np
from natsort import natsorted
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image, ImageDraw
import pandas as pd
import seaborn as sn
import glob
import scipy.stats as stats
import statistics
import matplotlib.patches as mpatches
from numba import jit
from scipy import signal
from scipy.io import savemat, loadmat
from scipy.stats import gamma
## ------------------------ Types of Results ------------------------ ##
TimeCourseCBV = False
TimeCourseCBVMovie = False
TimeCourseRFMovie = False
OnlyDisplayandNotsave = False;
averageCBV = True # mean CBV responses
averageCBVMovie = averageCBV and True
ROI_seed = True # calculate CBV responses averaged within seed region of interests selected if True,
                # otherwise calculate them within significantly positively and negatively correlated pixels
CorrelationMap1 = True # Pearson's correlation map
CaptureFrame = False # visualize individual CBV frames

## ---------------------- Data path --------------------- ##
path = 'D:\\Journal\\fUS-Disp-Motor-BehaviorPaper\\Data\\Figure3\\acquisition\\'
save_path = 'D:\\Journal\\fUS-Disp-Motor-BehaviorPaper\\Processed\\Figure3\\'
AcqInfo = 'D:\\Journal\\fUS-Disp-Motor-BehaviorPaper\\Data\\Figure3\\AcqInfo.mat'

path = 'D:\\Journal\\fUS-Disp-Motor-BehaviorPaper\\Data\\Figure5\\LThFUS\\acquisition\\'
save_path = 'D:\\Journal\\fUS-Disp-Motor-BehaviorPaper\\Processed\\Figure5\\LThFUS\\'
AcqInfo = 'D:\\Journal\\fUS-Disp-Motor-BehaviorPaper\\Data\\Figure5\\LThFUS\\AcqInfo.mat'

os.makedirs(save_path, exist_ok=True)
# CBV
wn = 4
CBVlim = 15
vmax = 0.4
medfilt_size = 5
# crop window
left_xlim=5.5
right_xlim=3
# correlation parameters
corr_threshold = 0.2
corr_lag = 0
# correlation visualization parameters
Corrlim = 0.4
Corrlim2 = 0.35
# functional framerate about 1 s
f_framerate=1;

cbv_window = int(30*f_framerate); #duration + post_stim
bsl_window = int(10*f_framerate); #pre_stim
bsl_fig_window = int(5*f_framerate); # time frame we wanna show in the figure

## -------------------------- Functions ----------------------------- ##
def moving_average(x, w):
    return np.convolve(x,np.ones(w),'valid') / w

@jit(nopython = True)
def calc_r(s,A,v,w):
    r = 0

    r1 = r; r2 = r; r3 = r
    for i in range(len(A)):
        r1 += (s[v,w,i]-np.mean(s[v,w,:]))*(A[i]-np.mean(A))
        r2 += (s[v,w,i]-np.mean(s[v,w,:]))**2
        r3 += (A[i]-np.mean(A))**2
    r = r1/(np.sqrt(r2)*np.sqrt(r3))
    return r

def hrf(times):
    peak_values = gamma.pdf(times, 3)
    undershoot_values = gamma.pdf(times, 3)
    values = peak_values - 0.9 * undershoot_values
    return values /np.max(values) * 0.6

## -------------------------- Colormaps ----------------------------- ##
cmap = plt.cm.RdBu
newRdBu = cmap(np.arange(cmap.N))
newRdBu[:,-1] = np.abs(np.linspace(-1, 1, cmap.N))**2
newRdBu = matplotlib.colors.ListedColormap(newRdBu);

cmap = plt.cm.gray
newgray = cmap(np.arange(cmap.N))
newgray[:,-1] = np.abs(np.linspace(-1, 1, cmap.N))**2
newgray = matplotlib.colors.ListedColormap(newgray);

## -------------------------- Processing ---------------------------- ##                
for u, v in h5py.File(AcqInfo, mode='r').items():
    exec("%s = v" % u)
imgsize = [int(CUDArecon['imZsize'][0][0]), int(CUDArecon['imXsize'][0][0])]
baseline = int(P['stim']['baseline'][0][0]*f_framerate)
cooldown = int(P['stim']['cooldown'][0][0]*f_framerate)
duration = int(P['stim']['duration'][0][0]*f_framerate)
total_stim= int(P['numstims'][0][0])
print("num stims:", total_stim, ", baseline:", baseline, ", stim:", duration, ", cooldown:", cooldown);

path_specific=save_path +'\\' 'lag' + str(corr_lag);
os.makedirs(path_specific, exist_ok=True);

frames = os.listdir(path)
nframes = len(frames)-(wn-1)

stim_frames = np.zeros(len(frames));

# setting regressor for correlation analyses
cooldown= cooldown+5;
duration= duration-5;
for i in range(0,total_stim):
    stim_frames[baseline+(i)*(duration+cooldown)+corr_lag+1:baseline+(i)*(duration+cooldown)+duration+corr_lag]=0;
    stim_frames[baseline+(i)*(duration+cooldown)+corr_lag+1:baseline+(i)*(duration+cooldown)+duration+corr_lag]=1;

baseline = int(P['stim']['baseline'][0][0]*f_framerate)
cooldown = int(P['stim']['cooldown'][0][0]*f_framerate)
duration = int(P['stim']['duration'][0][0]*f_framerate)

stim_frames_mw = moving_average(stim_frames,wn)
stim_frames_mw[stim_frames_mw>0]=1

RF = np.zeros((imgsize[0],imgsize[1],nframes));
non_norm_RF = np.zeros((imgsize[0],imgsize[1],nframes));
idx = 0;
s = np.zeros((imgsize[0],imgsize[1],len(frames)));

frames = os.listdir(path)

imgsize = [int(CUDArecon['imZsize'][0][0]), int(CUDArecon['imXsize'][0][0])]
im_extent = [CUDArecon['imXrange'][0][0],CUDArecon['imXrange'][-1][0],CUDArecon['imZrange'][-1][0],CUDArecon['imZrange'][0][0]]
im_extent = [x*Trans['wl'][0][0]*1e3 for x in im_extent]

lower_ylim = im_extent[2];
upper_ylim = im_extent[3];

text_x = -left_xlim+2.5;
text_y = upper_ylim+0.3;
brain_figratio = abs((left_xlim+right_xlim)/abs(upper_ylim-lower_ylim));
brain_figsize = [13,round(13/(brain_figratio+0.4),1)]
rcvdata = np.zeros((imgsize[0],imgsize[1],len(frames)))
files = natsorted(frames)
for i,f in enumerate(files):
    filepath = os.path.join(path,f)
    for u, v in h5py.File(filepath, mode = 'r').items():
        rcvdata[:,:,i] =np.transpose(np.array(v))
        print(i, 'th frame imported')
rcvdata = rcvdata/np.max(rcvdata);

for z in range(imgsize[0]):
    for x in range(imgsize[1]):
        RF[z,x,:] = moving_average(rcvdata[z,x,:],wn)

## -------------------------------------------------------------------- ##
dCBV = np.zeros((imgsize[0],imgsize[1],nframes))
baseline_rf = np.zeros((imgsize[0],imgsize[1]))
baseline_rf[:,:] = np.mean(RF[:,:,:baseline-wn+1],axis=2)

for frame in range(nframes):
    dCBV[:,:,frame] = signal.medfilt2d((RF[:,:,frame]-baseline_rf[:,:])/baseline_rf[:,:]*100,medfilt_size)

scale1 = {'vmin':0, 'vmax': vmax, 'cmap':newgray, 'aspect':'auto'}
scale2 = {'vmin':0, 'vmax': vmax, 'cmap':newgray, 'extent':im_extent, 'aspect':'auto'}
scale2_hot = {'vmin':0, 'vmax': vmax, 'cmap':'afmhot', 'extent':im_extent, 'aspect':'auto'}
scale_contour = {'vmin':corr_threshold, 'vmax': Corrlim2, 'cmap':'hot', 'extent':im_extent, 'aspect':'auto'}
scale3 = {'vmin':-CBVlim, 'vmax': CBVlim, 'alpha':1,'cmap':newRdBu,'extent':im_extent, 'aspect':'auto'}

if (TimeCourseCBV):
    all_masks = []
    num_regions = 2
    for region in range(num_regions):
        fig, ax1 = plt.subplots(1,1,figsize=(10,5))
        im1 = ax1.imshow((np.mean(RF[:,:,:baseline-wn+1,0],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1,0],axis=2))),**scale1)
        ROI = plt.ginput(n = -1, timeout = 0, show_clicks = True)
        plt.close()
        region = [(np.round(x[0]),np.round(x[1])) for x in ROI]
        img = Image.new('L', (imgsize[1],imgsize[0]), 0)
        ImageDraw.Draw(img).polygon(region, outline = 1, fill = 1)
        mask = np.array(img)
        all_masks.append(mask)

    any_x=[x/framerate for x in range(0,nframes)]
    fig, ax1 = plt.subplots(1,1,figsize=(10,5))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1,0],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1,0],axis=2)),**scale2)
    im1 = ax1.imshow(all_masks[0] + all_masks[1],alpha = 0.5,**scale2)
    ax1.set_title('ROI selection')
    if OnlyDisplayandNotsave:
        plt.show();
    else:
        img_path = path_specific + '\\ROI_TimetraceCBV.png';
        fig.savefig(img_path)
    dCBV_trace = []
    dCBV_trace_err = []
    dCBV_mean = np.zeros((nframes,1))
    dCBV_stddev = np.zeros((nframes,1))
    a = np.zeros((nframes,total_acq))
    for mask in all_masks:
        for i in range(total_acq):
            for frame in range(nframes):
                a[frame,i] = np.mean(dCBV[:,:,frame,i]*mask)
        dCBV_mean = np.mean(a,axis=-1);
        dCBV_stddev = np.std(a,axis=-1)
        dCBV_trace.append(dCBV_mean)
        dCBV_trace_err.append(dCBV_stddev)

    fig,[ax1, ax2]=plt.subplots(2,1,figsize=(14,6),sharex=True)
    im = ax1.plot(any_x, dCBV_trace[0]*100);
    ax1.fill_between(any_x,np.subtract(dCBV_trace[0]*100,dCBV_trace_err[0]*100).squeeze(),np.add(dCBV_trace[0]*100,dCBV_trace_err[0]*100).squeeze(), alpha=0.1)
    ymin, ymax = ax1.get_ylim();
    for i in range(0,total_stim):
        stim_window = mpatches.Rectangle(((baseline+i*(duration+cooldown)-wn+1)/framerate,ymin),duration/framerate,ymax-ymin, fill = True, linewidth = None, color = "red", alpha=0.2);
        ax1.add_patch(stim_window);
    ax1.set_title("CBV - ROI in Left Hemisphere (Imagewise)")

    im = ax2.plot(any_x, dCBV_trace[1]*100);
    ax2.fill_between(any_x,np.subtract(dCBV_trace[1]*100,dCBV_trace_err[1]*100).squeeze(),np.add(dCBV_trace[1]*100,dCBV_trace_err[1]*100).squeeze(), alpha=0.1)
    ymin, ymax = ax2.get_ylim();
    for i in range(0,total_stim):
        stim_window = mpatches.Rectangle(((baseline+i*(duration+cooldown)-wn+1)/framerate,ymin),duration/framerate,ymax-ymin, fill = True, linewidth = None, color = "red", alpha=0.2);
        ax2.add_patch(stim_window);
    ax2.set_title("CBV - ROI in Right Hemisphere (Imagewise)")
    plt.suptitle("$\Delta$CBV/CBV [%]")
    plt.xlabel("second [s]")
    if OnlyDisplayandNotsave:
        plt.show();
    else:
        img_path = path_specific +'\\Timecourse_CBVtrace.png';
        fig.savefig(img_path)
        img_path = path_specific +'\\Timecourse_CBVtrace.svg';
        fig.savefig(img_path)
    
if (TimeCourseCBVMovie):
    def init():
        im1.set_data(np.mean(RF[:,:,:baseline-wn+1,0],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1,0],axis=2)))
        im2.set_data(np.mean(dCBV[:,:,0,:],axis=-1))
        return im1, im2

    def animate(i):
        im1.set_data(np.mean(RF[:,:,:baseline-wn+1,0],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1,0],axis=2)))
        im2.set_data(np.mean(dCBV[:,:,i,:],axis=-1))
        if stim_frames_mw[i]:
            t.set_text('ON - time = {:.2f} s'.format(i/framerate))
            t.set_color('yellow');
        else:
            t.set_text('OFF - time = {:.2f} s'.format(i/framerate))
            t.set_color('white');
        return im1, im2, t

    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1,0],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1,0],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV[:,:,0,:],axis=-1),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    t = ax1.text(text_x,text_y,'OFF - time = 0 s',color='white',fontweight='bold',fontsize=24)
    ax1.set_xlabel('mm',fontsize=20)
    ax1.set_ylabel('mm',fontsize=20)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)

    ani = animation.FuncAnimation(fig,animate,interval = 100,frames=range(0,nframes),init_func = init, repeat=True)

    if OnlyDisplayandNotsave:
        plt.show()
    else:
        movie_path = path_specific + '\\Timetrace_CBV.mp4';
        ani.save(movie_path);

if (TimeCourseRFMovie):
    def init():
        im1.set_data(np.mean(RF[:,:,0,:],axis=-1))
        return im1

    def animate(i):
        im1.set_data(np.mean(RF[:,:,i,:],axis=-1))
        if stim_frames_mw[i]:
            t.set_text('ON - time = {:.2f} s'.format(i/framerate))
            t.set_color('yellow');
        else:
            t.set_text('OFF - time = {:.2f} s'.format(i/framerate))
            t.set_color('white');
        return im1, t
        
    fig, ax1 = plt.subplots(1,1,figsize=(15,7.5))
    im1 = ax1.imshow(np.mean(RF[:,:,0,:],axis=-1),**scale2_hot)
    t = ax1.text(text_x,text_y,'OFF - time = 0 s',color='white',fontweight='bold',fontsize=24)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_xlabel('mm',fontsize=20)
    ax1.set_ylabel('mm',fontsize=20)
    cbar = fig.colorbar(im1, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('PDI', rotation = 270, labelpad = 10, fontsize=20)

    ani = animation.FuncAnimation(fig,animate,interval = 100,frames=range(0,nframes),init_func = init, repeat=True)
    if OnlyDisplayandNotsave:
        plt.show()
    else:
        movie_path = path_specific +'\\RF.mp4';
        ani.save(movie_path);

if (averageCBV):
    dCBV_cropped_bsl=np.zeros((imgsize[0],imgsize[1],bsl_window+cbv_window+1,total_stim))
    bsl_rf = np.zeros((imgsize[0],imgsize[1],total_stim))
    for j in range(total_stim):
        bsl_rf[:,:,j] = np.mean(RF[:,:,baseline-bsl_window+j*(cooldown+duration)-wn+1:baseline+j*(cooldown+duration)-wn+1],axis=2)
    for i in range(total_stim):
        for frame in range(bsl_window+cbv_window+1):
            dCBV_cropped_bsl[:,:,frame,i] = signal.medfilt2d((RF[:,:,baseline-bsl_window+i*(cooldown+duration)-wn+1+frame]-bsl_rf[:,:,i])/bsl_rf[:,:,i]*100,medfilt_size)  

if (averageCBVMovie):
    def init():
        data1 = np.mean(RF[:,:,:baseline-wn+1], axis=2) / np.max(np.mean(RF[:,:,:baseline-wn+1], axis=2))
        im1.set_data(data1)

        data2 = np.mean(dCBV_cropped_bsl[:,:,0,:], axis=-1)
        im2.set_data(data2)
        return im1, im2

    def animate(i):
        data1 = np.mean(RF[:,:,:baseline-wn+1], axis=2) / np.max(np.mean(RF[:,:,:baseline-wn+1], axis=2))
        im1.set_data(data1)
        
        data2 = np.mean(dCBV_cropped_bsl[:,:,i,:], axis=-1)
        im2.set_data(data2)
        
        if i >= bsl_window and i < bsl_window + duration:
            t.set_text('FUS ON - time = {:.2f} s'.format((i-bsl_window+1)/f_framerate))
            t.set_color('yellow')
        else:
            t.set_text('FUS OFF - time = {:.2f} s'.format((i-bsl_window+1)/f_framerate))
            t.set_color('white')
        return im1, im2, t

    scale3 = {'vmin':-CBVlim, 'vmax': CBVlim, 'alpha':1,'cmap':newRdBu,'extent':im_extent, 'aspect':'auto'}

    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV_cropped_bsl[:,:,0,:],axis=-2),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_ylim((lower_ylim,upper_ylim))
    t = ax1.text(text_x,text_y,'FUS OFF - time = {:.2f} s'.format(-bsl_window),color='white',fontweight='bold',fontsize=24)
    ax1.set_xlabel('mm',fontsize=20)
    ax1.set_ylabel('mm',fontsize=20)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    ani = animation.FuncAnimation(fig,animate,interval = 200,frames=range(0,bsl_window+cbv_window+1-5),init_func = init, repeat=True)
    if OnlyDisplayandNotsave:
        plt.show()
    else:
        movie_path = path_specific +'\\CBV' + '.mp4';
        ani.save(movie_path);

if(CaptureFrame):
    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV_cropped_bsl[:,:,0,:],axis=-1),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_xlabel('mm',fontsize=10)
    ax1.set_ylabel('mm',fontsize=10)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)
    im2.set_data(np.mean(dCBV_cropped_bsl[:,:,10,:],axis=-1))
    plt.show()

    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV_cropped_bsl[:,:,0,:],axis=-1),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_xlabel('mm',fontsize=20)
    ax1.set_ylabel('mm',fontsize=20)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)
    im2.set_data(np.mean(dCBV_cropped_bsl[:,:,12,:],axis=-1))
    plt.show()

    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV_cropped_bsl[:,:,0,:],axis=-1),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_xlabel('mm',fontsize=20)
    ax1.set_ylabel('mm',fontsize=20)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)
    im2.set_data(np.mean(dCBV_cropped_bsl[:,:,13,:],axis=-1))
    plt.show()

    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV_cropped_bsl[:,:,0,:],axis=-1),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_xlabel('mm',fontsize=20)
    ax1.set_ylabel('mm',fontsize=20)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)
    im2.set_data(np.mean(dCBV_cropped_bsl[:,:,14,:],axis=-1))
    plt.show()

    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV_cropped_bsl[:,:,0,:],axis=-1),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_xlabel('mm',fontsize=20)
    ax1.set_ylabel('mm',fontsize=20)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)
    im2.set_data(np.mean(dCBV_cropped_bsl[:,:,16,],axis=-1))
    plt.show()

    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV_cropped_bsl[:,:,0,:],axis=-1),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_xlabel('mm',fontsize=20)
    ax1.set_ylabel('mm',fontsize=20)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)
    im2.set_data(np.mean(dCBV_cropped_bsl[:,:,32,:],axis=-1))
    plt.show()

if (CorrelationMap1):
    a = np.zeros((imgsize[0], imgsize[1], nframes))
    r = np.zeros((imgsize[0], imgsize[1]));
    for frame in range(nframes):
        a[:,:,frame] = RF[:,:,frame]
    for z in range(0,imgsize[0]):
        for x in range(0,imgsize[1]):
            r[z,x] = calc_r(a[:,:,:],stim_frames_mw[:],z,x);
    r0=np.copy(r);
    r0[r0<corr_threshold] = 0;
    r[:,:]=signal.medfilt2d(r[:,:],medfilt_size);
    r0[:,:]=signal.medfilt2d(r0[:,:],medfilt_size);
    
    fig, ax8 = plt.subplots(1, 1, figsize=(brain_figsize[0],brain_figsize[1]));
    ax8.set_xlim((-left_xlim,right_xlim))
    ax8.set_ylim((lower_ylim,upper_ylim))
    ax8.set_xlabel('mm',fontsize=20);
    ax8.set_ylabel('mm',fontsize=20);
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    
    ax8.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im8 = ax8.imshow(r[:,:], cmap = 'jet', vmin=-Corrlim, vmax=Corrlim, alpha=0.7, extent = im_extent, aspect = 'auto');
    cbar = fig.colorbar(im8, ax = ax8, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel(' ', rotation = 270, labelpad = 10, fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    if OnlyDisplayandNotsave:
        plt.show();
    else:
        img_path = path_specific +'\\correlation.png';
        fig.savefig(img_path)

    fig, ax8 = plt.subplots(1, 1, figsize=(brain_figsize[0],brain_figsize[1]));
    ax8.set_xlim((-left_xlim,right_xlim))
    ax8.set_ylim((lower_ylim,upper_ylim))
    ax8.set_xlabel('mm',fontsize=20);
    ax8.set_ylabel('mm',fontsize=20);
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    alpha_arr=(r0>0).astype(float);
    ax8.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im8 = ax8.imshow(r0[:,:], cmap = 'magma', vmin=corr_threshold, vmax=Corrlim2, alpha=alpha_arr, extent = im_extent, aspect = 'auto');
    cbar = fig.colorbar(im8, ax = ax8, fraction = 0.05, pad = 0.01)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel(' ', rotation = 270, labelpad = 10, fontsize=16)
    if OnlyDisplayandNotsave:
        plt.show();
    else:
        img_path = path_specific +'\\correlation_threshold_' + str(corr_threshold) + '.png';
        fig.savefig(img_path)

    fig, ax9 = plt.subplots(1, 1, figsize=(brain_figsize[0],brain_figsize[1]));
    ax9.set_xlim((-left_xlim,right_xlim))
    ax9.set_ylim((lower_ylim,upper_ylim))
    ax9.set_xlabel('mm',fontsize=20);
    ax9.set_ylabel('mm',fontsize=20);
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    alpha_arr=(r0>0).astype(float);
    ax9.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im8 = ax9.imshow(r0[:,:], cmap = 'hot', vmin=corr_threshold, vmax=Corrlim2, alpha=alpha_arr, extent = im_extent, aspect = 'auto');
    cbar = fig.colorbar(im8, ax = ax9, fraction = 0.05, pad = 0.01)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel(' ', rotation = 270, labelpad = 10, fontsize=16)
    if OnlyDisplayandNotsave:
        plt.show();
    else:
        img_path = path_specific + '\\sig_corr_' + str(corr_threshold) + '.png';
        fig.savefig(img_path)

    # seed based ROI
    if ROI_seed:
        seed_mask_A = np.zeros((r.shape[0],r.shape[1]))
        seed_mask_B = np.zeros((r.shape[0],r.shape[1]))
        seed_mask_A_contour = np.zeros((r.shape[0],r.shape[1]))
        seed_mask_B_contour = np.zeros((r.shape[0],r.shape[1]))
        fig, ax8 = plt.subplots(1, 1, figsize=(brain_figsize[0],brain_figsize[1]));
        fig.suptitle('Correlation Map', fontsize=20);
        ax8.set_xlabel('mm',fontsize=20);
        ax8.set_ylabel('mm',fontsize=20);
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        alpha_arr=(r0>0).astype(float);
        im8 = ax8.imshow(r0[:,:], cmap = 'hot', vmin=corr_threshold, vmax=Corrlim, alpha=alpha_arr, aspect = 'auto');
        cbar = fig.colorbar(im8);
        cbar.ax.tick_params(labelsize=20)
        seed_wn = 3
        AA = plt.ginput(n = 2, timeout = 0, show_clicks = True)
        plt.close()
        A = [int(x) for x in AA[0]]
        B = [int(x) for x in AA[1]]
        
        seed_mask_A[A[1]-seed_wn:A[1]+seed_wn,A[0]-seed_wn:A[0]+seed_wn]=1
        seed_mask_B[B[1]-seed_wn:B[1]+seed_wn,B[0]-seed_wn:B[0]+seed_wn]=1
        
        seed_mask_A_contour = np.copy(seed_mask_A);
        seed_mask_B_contour = np.copy(seed_mask_B);
        for k in range(1,seed_wn*2-1):
            seed_mask_A_contour[A[1]-seed_wn+k,A[0]-seed_wn+1:A[0]+seed_wn-1]=0
            seed_mask_B_contour[B[1]-seed_wn+k,B[0]-seed_wn+1:B[0]+seed_wn-1]=0
        seed_mask_A_contour=0.1*seed_mask_A_contour;
        seed_mask_B_contour=0.1*seed_mask_B_contour;
        # plot contour or ROI
        fig, ax8 = plt.subplots(1, 1, figsize=(brain_figsize[0],brain_figsize[1]));
        ax8.set_xlim((-left_xlim,right_xlim))
        ax8.set_ylim((lower_ylim,upper_ylim))
        fig.suptitle('Correlation Map', fontsize=20);
        ax8.set_xlabel('mm',fontsize=20);
        ax8.set_ylabel('mm',fontsize=20);
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        alpha_arr=(r0>0).astype(float);
        ax8.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
        im8 = ax8.imshow(r0[:,:], cmap = 'hot', vmin=corr_threshold, vmax=Corrlim2, alpha=alpha_arr, extent = im_extent, aspect = 'auto');
        im8 = ax8.imshow(seed_mask_A_contour+seed_mask_B_contour,alpha = seed_mask_A_contour*10+seed_mask_B_contour*3,**scale_contour)
        cbar = fig.colorbar(im8, ax = ax8, fraction = 0.05, pad = 0.01)
        cbar.ax.tick_params(labelsize=20)
        cbar.ax.set_ylabel(' ', rotation = 270, labelpad = 10, fontsize=16)
        cbar.ax.tick_params(labelsize=20)
        plt.show()
        img_path = path_specific + '\\sig_corr_ROIs' + '.png';
        fig.savefig(img_path)

        ROI_mat = {"seed_mask_A": seed_mask_A, "seed_mask_B": seed_mask_B, "seed_mask_A_contour": seed_mask_A_contour, "seed_mask_B_contour": seed_mask_B_contour}
        filename_roi =path_specific + "\\ROI"+".mat";
        savemat(filename_roi,ROI_mat)

    else: # 
        ROI_path =path_specific + "\\ROI"  +".mat";                                                                                                                        
        loadstr = loadmat(ROI_path);
        seed_mask_A = loadstr['seed_mask_A']; 
        seed_mask_B = loadstr['seed_mask_B'];
        
    ## CBV at activated region
    dCBV_act = np.zeros((nframes,1))
    RF_act_pos = np.zeros((nframes,1))
    RF_act_neg = np.zeros((nframes,1))
    a = np.zeros((imgsize[0], imgsize[1], nframes))
    b = np.zeros((imgsize[0], imgsize[1], nframes))
    print("# of positively correlated pixels: ", np.sum(r>corr_threshold));
    print("# of negatively correlated pixels: ", np.sum(r<-corr_threshold));

    for i in range(nframes):
        a[:,:,i] = dCBV[:,:,i]
        #b[:,:,i] = np.mean(non_norm_RF[:,:,i,:],axis=-1)
        dCBV_act[i] = np.mean(a[r>0,i]);
        if ROI_seed:
            RF_act_pos[i] = np.mean(a[seed_mask_A>0,i]);
            RF_act_neg[i] = np.mean(a[seed_mask_B>0,i]);
        else:
            RF_act_pos[i] = np.mean(a[r>corr_threshold,i]);
            RF_act_neg[i] = np.mean(a[r<-corr_threshold,i]);

    ## dCBV at positively correlated pixels
    f,ax1=plt.subplots(1,1,figsize=(14,6),sharex=True)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    im = ax1.plot(stim_frames_mw);
    im = ax1.plot(RF_act_pos);
    ymin, ymax = ax1.get_ylim();
    for i in range(0,total_stim):
        stim_window = mpatches.Rectangle(((baseline+i*(duration+cooldown)-wn+1)-1/f_framerate,ymin),duration/f_framerate,ymax-ymin, fill = True, linewidth = None, color = "red", alpha=0.2);
        ax1.add_patch(stim_window);
    title_str=""
    plt.suptitle("Positively correlated pixels or seed ROI A", fontsize=20)
    plt.ylabel("CBV changes [%]", fontsize=20)
    plt.xlabel("Time [s]", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.autoscale(enable=True, axis='x', tight=True)
    if OnlyDisplayandNotsave:
        plt.show();
    else:
        img_path = path_specific +'\\positive-corr_or_seedmaskA_CBVtrace' + '.svg';
        f.savefig(img_path)

    ## dCBV at negatively correlated pixels
    f,ax1=plt.subplots(1,1,figsize=(14,6),sharex=True)
    im = ax1.plot(stim_frames_mw);
    im = ax1.plot(RF_act_neg);
    ymin, ymax = ax1.get_ylim();
    for i in range(0,total_stim):
        stim_window = mpatches.Rectangle(((baseline+i*(duration+cooldown)-wn+1)-1/f_framerate,ymin),duration/f_framerate,ymax-ymin, fill = True, linewidth = None, color = "red", alpha=0.2);
        ax1.add_patch(stim_window);
    plt.suptitle("Negatively correlated pixels or seed ROI B", fontsize=20)
    plt.ylabel("CBV changes [%]", fontsize=20)
    plt.xlabel("Time [s]", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.autoscale(enable=True, axis='x', tight=True)
    any_x=[x/f_framerate for x in range(0,nframes)]

    if OnlyDisplayandNotsave:
        plt.show();
    else:
        img_path = path_specific +'\\negative-corr_or_seedmaskB_CBVtrace' +  '.svg';
        f.savefig(img_path)

    r_mat = {"corr": r, "pos_act_pixel_num": np.sum(r>corr_threshold), "neg_act_pixel_num": np.sum(r<-corr_threshold)}
    RF_mat = {"RF":np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2))}
    filename_corr =path_specific +"\\corr" + ".mat";
    filename_rf = path_specific +"\\RF" + ".mat";
    savemat(filename_corr,r_mat)
    savemat(filename_rf,RF_mat)

    # Average CBV at activated region
    any_x = range(-bsl_fig_window+1-1,cbv_window+2)
    dCBV_act_avg = np.zeros((bsl_window+cbv_window+1,total_stim))
    dCBV_act_avg_mean = np.zeros((bsl_window+cbv_window+1,1)).squeeze()
    dCBV_act_avg_std = np.zeros((bsl_window+cbv_window+1,1)).squeeze()
    dCBV_act_avg_neg = np.zeros((bsl_window+cbv_window+1,total_stim))
    dCBV_act_avg_mean_neg = np.zeros((bsl_window+cbv_window+1,1)).squeeze()
    dCBV_act_avg_std_neg = np.zeros((bsl_window+cbv_window+1,1)).squeeze()
    for j in range(total_stim):
        for i in range(bsl_window+cbv_window+1):
            if ROI_seed:
                dCBV_act_avg[i,j] = np.mean(dCBV_cropped_bsl[seed_mask_A>0,i,j]);
                dCBV_act_avg_neg[i,j] = np.mean(dCBV_cropped_bsl[seed_mask_B>0,i,j]);                   
            else:
                dCBV_act_avg[i,j] = np.mean(dCBV_cropped_bsl[r>corr_threshold,i,j]);
                dCBV_act_avg_neg[i,j] = np.mean(dCBV_cropped_bsl[r<-corr_threshold,i,j]);
    
    for i in range(bsl_window+cbv_window+1):
        dCBV_act_avg_mean[i] = np.mean(dCBV_act_avg[i,:],axis=-1);
        dCBV_act_avg_std[i] = np.std(dCBV_act_avg[i,:],axis=-1)/np.sqrt(total_stim);
        dCBV_act_avg_mean_neg[i] = np.mean(dCBV_act_avg_neg[i,:],axis=-1);
        dCBV_act_avg_std_neg[i] = np.std(dCBV_act_avg_neg[i,:],axis=-1)/np.sqrt(total_stim);
    dCBV_act_avg_mean = dCBV_act_avg_mean[bsl_window-bsl_fig_window-1:]
    dCBV_act_avg_std = dCBV_act_avg_std[bsl_window-bsl_fig_window-1:]
    dCBV_act_avg_mean_neg = dCBV_act_avg_mean_neg[bsl_window-bsl_fig_window-1:]
    dCBV_act_avg_std_neg = dCBV_act_avg_std_neg[bsl_window-bsl_fig_window-1:]

    f,ax1=plt.subplots(1,1,figsize=(9,7),sharex=True)
    im = ax1.plot(any_x, dCBV_act_avg_mean, linewidth=4);
    ax1.fill_between(any_x,np.subtract(dCBV_act_avg_mean,dCBV_act_avg_std).squeeze(),np.add(dCBV_act_avg_mean,dCBV_act_avg_std).squeeze(), alpha=0.25)
    ymin, ymax = ax1.get_ylim();
    stim_window = mpatches.Rectangle((0/f_framerate,ymin),duration/f_framerate,ymax-ymin, fill = True, linewidth = None, color = "red", alpha=0.1);
    ax1.add_patch(stim_window);
    plt.suptitle("Positively correlated pixels", fontsize=20)
    plt.ylabel("Mean CBV changes [%]", fontsize=20)
    plt.xlabel("Time [s]", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.autoscale(enable=True, axis='x', tight=True)
    if OnlyDisplayandNotsave:
        plt.show();
    else:
        img_path = path_specific +'\\positive-corr_or_seedmaskA_meanCBV' +  '.svg';
        f.savefig(img_path)

    f,ax1=plt.subplots(1,1,figsize=(9,7),sharex=True)
    im = ax1.plot(any_x,dCBV_act_avg_mean_neg, linewidth=4);
    ax1.fill_between(any_x,np.subtract(dCBV_act_avg_mean_neg,dCBV_act_avg_std_neg).squeeze(),np.add(dCBV_act_avg_mean_neg,dCBV_act_avg_std_neg).squeeze(), alpha=0.25)
    ymin, ymax = ax1.get_ylim();
    stim_window = mpatches.Rectangle((0/f_framerate,ymin),duration/f_framerate,ymax-ymin, fill = True, linewidth = None, color = "red", alpha=0.1);
    ax1.add_patch(stim_window);
    plt.suptitle("Negatively correlated pixels", fontsize=20)
    plt.ylabel("Mean CBV changes [%]", fontsize=20)
    plt.xlabel("Time [s]", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.autoscale(enable=True, axis='x', tight=True)
    
    if OnlyDisplayandNotsave:
        plt.show();
    else:
        img_path = path_specific + '\\negative-corr_or_seedmaskB_meanCBV' +  '.svg';
        f.savefig(img_path)

    pos_cbv_trace_mat = {"pos_CBVtrace": RF_act_pos, "neg_CBVtrace": RF_act_neg, "pos_meanCBV": dCBV_act_avg_mean, "pos_meanCBV_std": dCBV_act_avg_std, "neg_meanCBV": dCBV_act_avg_mean_neg,"neg_meanCBV_std": dCBV_act_avg_std_neg}
    filename_cbv =path_specific + "\\CBV" + ".mat";
    savemat(filename_cbv,pos_cbv_trace_mat)