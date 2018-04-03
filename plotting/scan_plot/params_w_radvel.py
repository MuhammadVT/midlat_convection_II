#import matplotlib
#matplotlib.use("Agg")

from matplotlib.dates import DateFormatter 
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from davitpy import gme
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

def params_panel_plot(stime, etime, 
                      ylim_V=[350,700], ylim_Pdyn=[0,20],
                      ylim_IMF=[-30, 30], ylim_theta=[-180, 180],
                      ylim_symh=[-50, 50], ylim_ae=[0, 1000],
                      ylim_kp=[0, 9],
                      marker='.', linestyle='--',
                      markersize=2, vline=False,
                      panel_num=7, fig_size=None, hspace=None):

    fig, axes = draw_axes(fig_size=fig_size, panel_num=panel_num,
                          hspace=hspace)
    ax_V, ax_Pdyn, ax_IMF, ax_theta = axes[:4]
    
    # plot ace solar wind data
    plot_ace(stime, etime, ax_V=ax_V, ax_Pdyn=ax_Pdyn, ax_IMF=ax_IMF, ax_theta=ax_theta,
             ylim_V=ylim_V, ylim_Pdyn=ylim_Pdyn, ylim_IMF=ylim_IMF,ylim_theta=ylim_theta,
             marker=marker, linestyle=linestyle, markersize=markersize, drop_na=True)
             #marker=marker, linestyle=linestyle, markersize=markersize, drop_na=False)

    # plot symh
    ax_symh = axes[4]
    plot_symh(stime, etime, ax_symh, ylim_symh=ylim_symh,
            marker=marker, linestyle=linestyle, markersize=markersize, zero_line=True)

    # plot AE
    ax_ae = axes[5]
    plot_ae(stime, etime, ax_ae, ylim_ae=ylim_ae,
        marker=marker, linestyle=linestyle, markersize=markersize)

    # plot Kp
    ax_kp = axes[6]
    plot_kp(stime, etime, ax_kp, ylim_kp=ylim_kp,
        marker=marker, linestyle=linestyle, markersize=markersize)

#    # plot RISR vel 
#    ax_vel = axes[6]
#    plot_vel(stime, etime, ax_vel, ylim_vel=[-2000, 3000],
#        marker=marker, linestyle=linestyle, markersize=markersize)
#
#    # plot fitted vel error
#    ax_vel_err = axes[7]
#    plot_vel_err(stime, etime, ax_vel_err, ylim_vel=[0, 100], ylabel_fontsize=9,
#        marker=marker, linestyle=linestyle, markersize=markersize, padding_method='ffill', zero_line=True)


    # adjusting tick parameters
    for ll in range(panel_num): 
        axes[ll].yaxis.set_tick_params(labelsize=11)
        #axes[ll].xaxis.grid(True, which='major')

        # lining up the ylabels and setting ylabel fontsize
        labelx = -0.10  # axes coords
        axes[ll].yaxis.set_label_coords(labelx, 0.5)
        axes[ll].yaxis.label.set_size(11)

        # add vlines at the points of interest
        if vline:
            dt1 = dt.datetime(2014,9,12,15,26)
            dt2 = dt.datetime(2014,9,12,15,55)
            dt3 = dt.datetime(2014,9,12,16,57)
            #dt4 = dt.datetime(2014,9,12,17,51)
            dtt = [dt1, dt2, dt3]

            for ij, tt in enumerate(dtt):
                axes[ll].axvline(tt, color='r', linewidth=0.50, linestyle='--')

                #axes[ll].axvline(dt2, color='r', linewidth=0.50, linestyle='--')
                #axes[ll].axvline(dt3, color='r', linewidth=0.50, linestyle='--')
                ##axes[ll].axvline(dt4, color='r', linewidth=0.50, linestyle='--')

    # add roman numerals
    if vline:
        vline_loc = [(0.182, 1.0), (0.243, 1.0), (0.372, 1.0)]
        rom_nums = [r"\rom{1}", r"\rom{2}", r"\rom{3}"]
        # Turn on LaTeX formatting for text    
        plt.rcParams['text.usetex']=True

        # Place the command in the text.latex.preamble using rcParams
        plt.rcParams['text.latex.preamble']=r'\makeatletter \newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@} \makeatother'
        for ij, tt in enumerate(dtt):
            axes[0].annotate(rom_nums[ij], xy=vline_loc[ij], xycoords="axes fraction",
                             ha="center", va="bottom")

    # format the datetime xlabels
    if (etime-stime).days >= 2:   
        axes[-1].xaxis.set_major_formatter(DateFormatter('%m/%d'))
        locs = axes[-1].xaxis.get_majorticklocs()
        locs = locs[::2]
        locs = np.append(locs, locs[-1]+1)
        axes[-1].xaxis.set_ticks(locs)
    if (etime-stime).days == 0:   
        axes[-1].xaxis.set_major_formatter(DateFormatter('%H:%M'))

    # adding extra xtick labels
    if vline:
        from matplotlib import dates
        poss = np.append(axes[-1].xaxis.get_majorticklocs(), axes[-1].xaxis.get_minorticklocs())
        poss = np.append(poss, dates.date2num([dt1, dt2]))
        poss = np.delete(poss, 2)
        axes[-1].xaxis.set_ticks(poss)

    # remove xtick labels of the subplots except for the last one
    for i in range(panel_num):
        if i < panel_num-1:
            axes[i].tick_params(axis='x',which='both',labelbottom='off') 

    # rotate xtick labels
    plt.setp(axes[-1].get_xticklabels(), rotation=30)

    # set axis label and title
    axes[-1].set_xlabel('Time UT')
    axes[-1].xaxis.set_tick_params(labelsize=11)
    if (etime-stime).days > 0:
        axes[0].set_title('  ' + 'Date: ' +\
                stime.strftime("%m/%d/%Y") + ' - ' + (etime-dt.timedelta(days=1)).strftime("%m/%d/%Y")\
                + '    ACE SW data')
    else:
        axes[0].set_title(stime.strftime("%m/%d/%Y"))

    return fig

def draw_axes(fig_size=None, panel_num=6, hspace=None):
    
    fig, axes = plt.subplots(nrows=panel_num, figsize=fig_size, sharex=True)
    if hspace is not None:
        fig.subplots_adjust(hspace=hspace)

    return fig, axes

def plot_ace(stime, etime, ax_V=None, ax_Pdyn=None, ax_IMF=None,ax_theta=None,
        ylim_V=[0,800],ylim_Pdyn=[0,60], ylim_IMF=[-30, 30],ylim_theta=[-180, 180],
        marker='.', linestyle='--', markersize=2, ylabel_fontsize=9, zero_line=True,
        padding_method=None, drop_na=False):

    import funcs

    # read ace data that is minute-averaged
    df_ace = funcs.ace_read(stime, etime) 
    df_ace.loc[:, 'Bt'] = np.sqrt((df_ace.By ** 2 + df_ace.Bz ** 2))
    # clock angle
    df_ace.loc[:, 'theta_Bt'] = np.degrees(np.arctan2(df_ace.By, df_ace.Bz))
    # dynamic pressure
    mp = 1.6726219 * 1e-27  # kg
    df_ace.loc[:, 'Pdyn'] = (df_ace.Np * mp * 1e6) * (df_ace.Vx * 1e3)**2 * 1e+9
    if drop_na:
        df_ace.dropna(how='any', inplace=True)
    if padding_method is not None:
        df_ace.fillna(method=padding_method, inplace=True)

    # plot solar wind Vx
    df_ace.loc[df_ace.Vx < -1e4, 'Vx'] = np.nan 
    ax_V.plot_date(df_ace.index.to_pydatetime(), -df_ace.Vx, color='k',
            marker=marker, linestyle=linestyle, markersize=markersize)
    ax_V.set_ylabel('V [km/s]', fontsize=ylabel_fontsize)
    ax_V.set_ylim([ylim_V[0], ylim_V[1]])
    ax_V.locator_params(axis='y', nbins=4)

#    # add text to indicate sheath region
#    dt1 = dt.datetime(2014,9,12,15,26)
#    dt2 = dt.datetime(2014,9,12,16,10)
#    dt3 = dt.datetime(2014,9,12,16,57)
#    ax_V.annotate('', xy=(dt1, 400), xycoords='data', xytext=(dt3, 400),
#            arrowprops={'arrowstyle': '<->'})
#    ax_V.annotate('Sheath', xy=(dt2, 400), xycoords='data', xytext=(0, 5), ha='center',
#            textcoords='offset points')

    # plot Pdyn [nPa]
    df_ace.loc[df_ace.Pdyn < 0, 'Pdyn'] = np.nan
    ax_Pdyn.plot_date(df_ace.index.to_pydatetime(), df_ace.Pdyn, color='k',
            marker=marker, linestyle=linestyle, markersize=markersize)
    ax_Pdyn.set_ylabel('Pdyn [nPa]', fontsize=ylabel_fontsize)
    ax_Pdyn.set_ylim([ylim_Pdyn[0], ylim_Pdyn[1]])
    ax_Pdyn.locator_params(axis='y', nbins=4)

    # plot solar wind IMF
    ax_IMF.plot_date(df_ace.index.to_pydatetime(), df_ace.Bx, color='k',
            marker=marker, linestyle=linestyle, markersize=markersize, linewidth=0.3)
    ax_IMF.plot_date(df_ace.index.to_pydatetime(), df_ace.By, color='g',
            marker=marker, linestyle=linestyle, markersize=markersize)
    ax_IMF.plot_date(df_ace.index.to_pydatetime(), df_ace.Bz, color='r',
            marker=marker, linestyle=linestyle, markersize=markersize)
    lns = ax_IMF.get_lines()
    ax_IMF.legend(lns,['Bx', 'By', 'Bz'], frameon=False, bbox_to_anchor=(0.98, 0.5),
            loc='center left', fontsize='medium')
    #ax_IMF.legend(lns,['Bx', 'By', 'Bz'], frameon=False, fontsize='small', mode='expand')
    ax_IMF.set_ylabel('IMF [nT]', fontsize=ylabel_fontsize)
    ax_IMF.set_ylim([ylim_IMF[0], ylim_IMF[1]])
    ax_IMF.locator_params(axis='y', nbins=4)
    if zero_line:
        ax_IMF.axhline(y=0, color='r', linewidth=0.30, linestyle='--')

    # plot solar wind clock angle
    ax_theta.plot_date(df_ace.index.to_pydatetime(), df_ace.theta_Bt, color='k',
            marker=marker, linestyle=linestyle, markersize=markersize)
    ax_theta.set_ylabel(r'$\theta$ [deg]', fontsize=ylabel_fontsize)
    ax_theta.set_ylim([ylim_theta[0], ylim_theta[1]])
    ax_theta.locator_params(axis='y', nbins=4)
    # add extra ytick labels
    #poss1 = np.append(ax_theta.yaxis.get_majorticklocs(), ax_theta.yaxis.get_minorticklocs())
    poss1 = ax_theta.yaxis.get_majorticklocs()
    poss1 = np.append(poss1, [-180, -90, 90, 180])
    poss1 = np.delete(poss1, [0, 1, 3, 4])
    ax_theta.yaxis.set_ticks(poss1)
    # draw horizontal lines
    if zero_line:
        ax_theta.axhline(y=0, color='r', linewidth=0.30, linestyle='--')
    ninety_line = True
    if ninety_line:
        ax_theta.axhline(y=90, color='r', linewidth=0.30, linestyle='--')
        ax_theta.axhline(y=-90, color='r', linewidth=0.30, linestyle='--')

#    # add Bt component
#    ax_theta_twinx = ax_theta.twinx()
#    ax_theta_twinx.plot_date(df_ace.index.to_pydatetime(), df_ace.Bt, color='b',
#            marker=marker, linestyle=linestyle, markersize=markersize)
#    ax_theta_twinx.locator_params(axis='y', nbins=4)
#    ax_theta_twinx.set_ylim([ylim_IMF[0], ylim_IMF[1]])
#    ax_theta_twinx.set_ylabel('IMF $|$Bt$|$ [nT]', color='b')
#    ax_theta_twinx.yaxis.label.set_size(11)
#    ax_theta_twinx.yaxis.set_tick_params(labelsize=11, labelcolor='b')

#    # Plot a rectangle to shade a certain period
#    # convert to matplotlib date representation
#    sdt = dt.datetime(2014,9,12,16,57)
#    edt = dt.datetime(2014,9,12,19,00)
#    spnt = mdates.date2num(sdt)
#    epnt = mdates.date2num(edt)
#    width = epnt - spnt 
#
#    # Plot rectangle
#    recth = [700, 20, 60, 360]
#    rect_bottom = [350, 0, -30, -180]
#    for ii, axx in enumerate([ax_V, ax_Pdyn, ax_IMF ,ax_theta]):
#        rect = Rectangle((spnt, rect_bottom[ii]), width, recth[ii], color='gray', alpha=0.2)
#        axx.add_patch(rect) 

    return


def plot_symh(stime, etime, ax_symh, ylim_symh=[-100, 100], ylabel_fontsize=9,
        marker='.', linestyle='--', markersize=2, zero_line=True):

    import symasy

    # read SYMH data
    sym_list = symasy.readSymAsyWeb(sTime=stime,eTime=etime)
    #sym_list = gme.ind.symasy.readSymAsyWeb(sTime=stime,eTime=etime)
    #sym_list = gme.ind.symasy.readSymAsy(sTime=stime,eTime=etime)
    symh = []
    symh_time = []
    for k in range(len(sym_list)):
        symh.append(sym_list[k].symh)
        symh_time.append(sym_list[k].time)
    
    # plot symh
    indx = [symh_time.index(x) for x in symh_time if (x>= stime) and (x<=etime)]
    symh_time = [symh_time[i] for i in indx]
    symh = [symh[i] for i in indx]

    ax_symh.plot_date(symh_time, symh, 'k', marker=marker, linestyle=linestyle, markersize=markersize)
    ax_symh.set_ylabel('SYM-H', fontsize=9)
    ax_symh.set_ylim([ylim_symh[0], ylim_symh[1]])
    ax_symh.locator_params(axis='y', nbins=4)
    if zero_line:
        ax_symh.axhline(y=0, color='r', linewidth=0.30, linestyle='--')

#    # Plot a rectangle to shade a certain period
#    # convert to matplotlib date representation
#    sdt = dt.datetime(2014,9,12,17,51)
#    edt = dt.datetime(2014,9,12,20,00)
#    spnt = mdates.date2num(sdt)
#    epnt = mdates.date2num(edt)
#    width = epnt - spnt 
#
#    # Plot rectangle
#    recth = 100
#    rect_bottom = -50
#    rect = Rectangle((spnt, rect_bottom), width, recth, color='gray', alpha=0.2)
#    ax_symh.add_patch(rect) 

    return


def plot_ae(stime, etime, ax_ae, ylim_ae=[0, 500], ylabel_fontsize=9,
        marker='.', linestyle='--', markersize=2):
    import ae
    # read AE data
    AE_list = ae.readAeWeb(sTime=stime,eTime=etime,res=1)
    #AE_list = gme.ind.readAeWeb(sTime=stime,eTime=etime,res=1)
    #AE_list = gme.ind.readAe(sTime=stime,eTime=etime,res=1)
    AE = []
    AE_time = []
    for m in range(len(AE_list)):
        AE.append(AE_list[m].ae)
        AE_time.append(AE_list[m].time)

    # plot AE 
    indx = [AE_time.index(x) for x in AE_time if (x>= stime) and (x<=etime)]
    AE_time = [AE_time[i] for i in indx]
    AE = [AE[i] for i in indx]

    ax_ae.plot_date(AE_time, AE, 'k', marker=marker,
                    linestyle=linestyle, markersize=markersize)
    ax_ae.set_ylabel('AE', fontsize=9)
    ax_ae.set_ylim([ylim_ae[0], ylim_ae[1]])
    ax_ae.locator_params(axis='y', nbins=4)

#    # Plot a rectangle to shade a certain period
#    # convert to matplotlib date representation
#    sdt = dt.datetime(2014,9,12,17,51)
#    edt = dt.datetime(2014,9,12,20,00)
#    spnt = mdates.date2num(sdt)
#    epnt = mdates.date2num(edt)
#    width = epnt - spnt 
#
#    # Plot rectangle
#    recth = 1000
#    rect_bottom = 0
#    rect = Rectangle((spnt, rect_bottom), width, recth, color='gray', alpha=0.2)
#    ax_ae.add_patch(rect) 

    return

def plot_aualae(stime, etime, ylim_au=[0, 500],
                ylim_al=[-500, 0], ylim_ae=[0, 500], ylabel_fontsize=9,
                marker='.', linestyle='--', markersize=2):

    import ae

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    # read AE data
    AE_list = ae.readAeWeb(sTime=stime,eTime=etime,res=1)
    #AE_list = gme.ind.readAeWeb(sTime=stime,eTime=etime,res=1)
    #AE_list = gme.ind.readAe(sTime=stime,eTime=etime,res=1)
    AE = []
    AU = []
    AL = []
    AE_time = []
    for m in range(len(AE_list)):
        AU.append(AE_list[m].au)
        AL.append(AE_list[m].al)
        AE.append(AE_list[m].ae)
        AE_time.append(AE_list[m].time)

    # plot AU, AL, AE
    indx = [AE_time.index(x) for x in AE_time if (x>= stime) and (x<=etime)]
    AE_time = [AE_time[i] for i in indx]
    AE = [AE[i] for i in indx]
    
    ylabels = ["AU", "AL", "AE"]
    ylims = [ylim_au, ylim_al, ylim_ae]
    colors = ["k", "g", "r"]
    for i, var in enumerate([AU, AL, AE]):
        ax = axes[i]
        ax.plot_date(AE_time, var, colors[i], marker=marker,
                     linestyle=linestyle, markersize=markersize)
        ax.set_ylabel(ylabels[i], fontsize=ylabel_fontsize)
        ax.set_ylim([ylims[i][0], ylims[i][1]])
        ax.locator_params(axis='y', nbins=4)

    # format the datetime xlabels
    if (etime-stime).days >= 2:   
        axes[-1].xaxis.set_major_formatter(DateFormatter('%m/%d'))
        locs = axes[-1].xaxis.get_majorticklocs()
        locs = locs[::2]
        locs = np.append(locs, locs[-1]+1)
        axes[-1].xaxis.set_ticks(locs)
    if (etime-stime).days == 0:   
        axes[-1].xaxis.set_major_formatter(DateFormatter('%H:%M'))

    # rotate xtick labels
    plt.setp(axes[-1].get_xticklabels(), rotation=30)

    # set axis label and title
    axes[-1].set_xlabel('Time UT')
    axes[-1].xaxis.set_tick_params(labelsize=11)
    if (etime-stime).days > 0:
        axes[0].set_title('  ' + 'Date: ' +\
                stime.strftime("%m/%d/%Y") + ' - ' + (etime-dt.timedelta(days=1)).strftime("%m/%d/%Y")\
                + '    AU, AL, AE Indices')
    else:
        axes[0].set_title(stime.strftime("%m/%d/%Y"))



    return fig


def plot_kp(stime, etime, ax_kp, ylim_kp=[0, 9], ylabel_fontsize=9,
        marker='.', linestyle='--', markersize=2):

    import datetime as dt
    from matplotlib.ticker import MultipleLocator

    # Kp data
    #Kp_list = gme.ind.readKp(sTime=stime,eTime=etime)
    stm = dt.datetime(stime.year, stime.month, stime.day)
    etm = dt.datetime(etime.year, etime.month, etime.day+1)
    Kp_list = gme.ind.readKpFtp(sTime=stm, eTime=etm)
    Kp = []
    Kp_time = []
    for n in range((etm-stm).days+1):
        kp_tmp = Kp_list[n].kp
        time_tmp = Kp_list[n].time
        for l in range(len(kp_tmp)):
            if len(kp_tmp[l])== 2:
                if kp_tmp[l][1] == '+':
                    Kp.append(int(kp_tmp[l][0])+0.3)
                elif kp_tmp[l][1] == '-':
                    Kp.append(int(kp_tmp[l][0])-0.3)
            else:
                Kp.append(int(kp_tmp[l][0]))
            Kp_time.append(time_tmp + dt.timedelta(hours=3*l))

    # plot Kp 
    indx = [Kp_time.index(x) for x in Kp_time if (x>= stime) and (x<=etime)]
    Kp_time = [Kp_time[i] for i in indx]
    Kp = [Kp[i] for i in indx]
    #ax_kp.plot_date(Kp_time, Kp, 'k', marker=marker, linestyle=linestyle, markersize=markersize)
    ax_kp.stem(Kp_time, Kp, marketcolor="k", linefmt='k--', marker=marker, markersize=markersize)
    ax_kp.set_ylabel('Kp', fontsize=9)
    ax_kp.set_ylim([ylim_kp[0], ylim_kp[1]])
    ax_kp.locator_params(axis='y')
    ax_kp.yaxis.set_major_locator(MultipleLocator(3))

    # Plot hline at kp=2.3
    ax_kp.axhline(y=2.3, color='r', linewidth=0.30, linestyle='--')

def plot_vel(stime, etime, ax_vel, ylim_vel=[-1000, 1000], ylabel_fontsize=9,
        marker='.', linestyle='--', markersize=2, padding_method='ffill', zero_line=True):
    #df_vel = pd.read_csv('./losvelNS_09122014.csv', index_col=0, parse_dates=True)
    #df_vel = pd.read_csv('./mergevel_bm57_09122014.csv', index_col=0, parse_dates=True)
    df_vel = pd.read_csv('./mergevel_bm345789_09122014.csv', index_col=0, parse_dates=True)
    df_vel = df_vel.loc[stime:etime, :]
    df_vel.fillna(method='ffill', inplace=True)

    # plot velocity data
    ax_vel.plot_date(df_vel.index.to_pydatetime(), df_vel.vels_NS, color='k',
            marker=marker, linestyle=linestyle, markersize=markersize)
    ax_vel.plot_date(df_vel.index.to_pydatetime(), -df_vel.vels_EW, color='b',
            marker=marker, linestyle=linestyle, markersize=markersize)
    ax_vel.set_ylabel('Vel. [m/s]', fontsize=9)
    ax_vel.set_ylim([ylim_vel[0], ylim_vel[1]])
    ax_vel.locator_params(axis='y', nbins=4)
    lns = ax_vel.get_lines()
    ax_vel.legend(lns,['$V_{s}$', '$V_{w}$'], frameon=False, bbox_to_anchor=(0.98, 0.5),
            loc='center left', fontsize='medium')
    if zero_line:
        ax_vel.axhline(y=0, color='r', linewidth=0.30, linestyle='--')

    # Plot a rectangle to shade a certain period
    # convert to matplotlib date representation
    sdt = dt.datetime(2014,9,12,17,51)
    edt = dt.datetime(2014,9,12,20,00)
    spnt = mdates.date2num(sdt)
    epnt = mdates.date2num(edt)
    width = epnt - spnt 

    # Plot rectangle
    recth = 5000
    rect_bottom = -2000
    rect = Rectangle((spnt, rect_bottom), width, recth, color='gray', alpha=0.2)
    ax_vel.add_patch(rect) 

def plot_vel_err(stime, etime, ax_vel_err, ylim_vel=[0, 100], ylabel_fontsize=9,
        marker='.', linestyle='--', markersize=2, padding_method='ffill', zero_line=True):
    #df_vel = pd.read_csv('./losvelNS_09122014.csv', index_col=0, parse_dates=True)
    #df_vel = pd.read_csv('./mergevel_bm57_09122014.csv', index_col=0, parse_dates=True)
    df_vel = pd.read_csv('./fitvelerr.csv', index_col=0, parse_dates=True)
    df_vel = df_vel.loc[stime:etime, :]
    df_vel.fillna(method='ffill', inplace=True)

    # plot velocity error data
    ax_vel_err.plot_date(df_vel.index.to_pydatetime(), df_vel.velfiterr, color='k',
            marker=marker, linestyle=linestyle, markersize=markersize)
    ax_vel_err.set_ylabel('Vel. err [m/s]', fontsize=9)
    ax_vel_err.set_ylim([ylim_vel[0], ylim_vel[1]])
    ax_vel_err.locator_params(axis='y', nbins=4)
    lns = ax_vel_err.get_lines()
    ax_vel_err.legend(lns,['$V_{err}$'], frameon=False, bbox_to_anchor=(0.98, 0.5),
            loc='center left', fontsize='medium')
    if zero_line:
        ax_vel_err.axhline(y=50, color='r', linewidth=0.30, linestyle='--')

    # Plot a rectangle to shade a certain period
    # convert to matplotlib date representation
    sdt = dt.datetime(2014,9,12,17,51)
    edt = dt.datetime(2014,9,12,20,00)
    spnt = mdates.date2num(sdt)
    epnt = mdates.date2num(edt)
    width = epnt - spnt 

    # Plot rectangle
    recth = 5000
    rect_bottom = -2000
    rect = Rectangle((spnt, rect_bottom), width, recth, color='gray', alpha=0.2)
    ax_vel_err.add_patch(rect) 




# testing
if __name__ == "__main__":

    import datetime as dt
    import os
    import matplotlib.pyplot as plt

    #stime = dt.datetime(2013,11,14,3) 
    #etime = dt.datetime(2013,11,14,9) 

    #stime = dt.datetime(2013,2,4,3) 
    #etime = dt.datetime(2013,2,4,9) 

    stime = dt.datetime(2011,5,29,0) 
    etime = dt.datetime(2011,5,29,9) 

    panel_num = 7; fig_size=(8,10); hspace=None

    fig = params_panel_plot(stime, etime,
                            ylim_V=[300,800], ylim_Pdyn=[0,20],
                            ylim_IMF=[-10, 10], ylim_theta=[-180, 180],
                            ylim_symh=[-100, 50], ylim_ae=[0, 1500],
                            ylim_kp=[0, 9],
                            marker='', linestyle='-',
                            markersize=1, vline=False, panel_num=panel_num,
                            fig_size=fig_size, hspace=0.15)

#################################################################
# For testing individual functions
#    fig, ax = plt.subplots()
#    plot_symh(stime, etime, ax, ylim_symh=[-50, 50],
#              marker='', linestyle='-', markersize=1,
#              zero_line=True)

#    plot_ae(stime, etime, ax, ylim_ae=[0, 100],
#            marker='', linestyle='-', markersize=1)


#    plot_kp(stime, etime, ax, ylim_kp=[0, 9],
#            marker='', linestyle='-', markersize=1)
#################################################################

    # Create a folder for a date
    fig_dir = "../plots/scan_plot/" + stime.strftime("%Y%m%d") + "/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Save the figure
    txt = "indicies_"
    fig_name = txt +\
               stime.strftime("%Y%m%d.%H%M") + "_to_" +\
               etime.strftime("%Y%m%d.%H%M")
    fig.savefig( fig_dir + fig_name +\
                ".png", dpi=200, bbox_inches="tight")

    plt.close(fig)

#################################################################
    # Plot the AU/AL/AE indices
    ylim_au=[0, 1000]
    ylim_al=[-ylim_au[1], 0]
    #ylim_ae=[0, ylim_au[1] - ylim_al[0]]
    ylim_ae=[0, 1500]
    fig = plot_aualae(stime, etime, ylim_au=ylim_au,
                    ylim_al=ylim_al, ylim_ae=ylim_ae, ylabel_fontsize=9,
                    marker='.', linestyle='--', markersize=2)
    txt = "au_al_ae_"
    fig_name = txt +\
               stime.strftime("%Y%m%d.%H%M") + "_to_" +\
               etime.strftime("%Y%m%d.%H%M")
    fig.savefig( fig_dir + fig_name +\
                ".png", dpi=200, bbox_inches="tight")

    plt.close(fig)

    #plt.show()


