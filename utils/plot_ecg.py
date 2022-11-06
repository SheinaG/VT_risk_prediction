import sys

sys.path.append("/home/shanybiton/repos/afib-prediction/preproccesing/")
sys.path.append("/home/shanybiton/repos/afib-prediction/")
sys.path.append("/home/shanybiton/repos/afib-prediction/parser")
sys.path.append("/home/shanybiton/repos/afib-prediction/parser/parsing")
sys.path.append("/home/shanybiton/repos/afib-prediction/parser/utils")
import numpy as np
import matplotlib.pyplot as plt
# from preprocessing.Feature_extractor import bandpass_filter
# from tnmg_parser import TNMGDB_Parser
# import consts as cts
# import pandas as pd
from matplotlib.ticker import AutoMinorLocator
# import ecg_plot
from math import ceil


# from ptbxl_parser import PTBXL_Parser

# def reorganize_data(data, lead_index, orig_index, actual_fs, orig_fs):
#     from scipy import signal
#     ecg = np.zeros(shape=(4096, 12))
#     for i_lead, lead in enumerate(lead_index):
#         ecg_lead = data[:, orig_index[lead]]
#         ecg_lead = signal.resample(ecg_lead, int(len(ecg_lead) * actual_fs / orig_fs))
#         ecg_lead = bandpass_filter(data=ecg_lead, id=i_lead, lead=lead, lowcut=0.67, highcut=100, signal_freq=400,
#                                    filter_order=75, notch_freq=50, debug=False)
#         padding = 4096-len(ecg_lead)
#         offset = int(round(padding/2))
#         ecg[offset:len(ecg_lead) + offset, i_lead] = ecg_lead
#         # ecg[:, i_lead] = ecg_lead
#     return ecg
#
#
# def _ax_plot(ax, x, y, secs=10, lwidth=0.5, amplitude_ecg=1.8, time_ticks=0.2):
#     ax.set_xticks(np.arange(0, 11, time_ticks))
#     ax.set_yticks(np.arange(-ceil(amplitude_ecg), ceil(amplitude_ecg), 1.0))
#
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
#     ax.xaxis.set_ticks_position('none')
#     ax.yaxis.set_ticks_position('none')
#
#     ax.minorticks_on()
#
#     # ax.xaxis.set_minor_locator(AutoMinorLocator(5))
#     ax.set_ylim(-amplitude_ecg, amplitude_ecg)
#     ax.set_xlim(0, secs)
#     ax.grid(b=False, which='major', linestyle='-', linewidth='0.5', color='red')
#     ax.grid(b=False, which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
#
#     ax.plot(x, y, linewidth=lwidth, color='k')
#
# def plot_ecg_fig(df, patient_id, exams_id, namedfile,
#                  path_to_save='/home/shanybiton/AIMLabProjects/afib-prediction/outputs/examples/Grant/id_'):
#     path_to_csv = '/MLdata/AIMLab/databases/tnmg/ecg-traces/ecg-traces/preprocessed/traces.hdf5'
#     f = h5py.File(path_to_csv, 'r')
#     # Get ids
#     traces_ids = np.array(f['id_exam'])
#     x = f['signal']
#
#     signals_num = 12
#     lead_index = ('DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
#
#     for k in patient_id:
#         path = path_to_save + str(k)
#
#         try:
#             os.mkdir(path)
#         except OSError:
#             print("Creation of the directory %s failed" % path)
#         exams_k = df[df['id_patient'] == k]
#         # patients_id_exams[m] in i for id_k['id_exam']  in patients_id_exams
#         contained = [x in exams_k['id_exam']._values for x in exams_id]
#         exam_to_export = (np.asarray(exams_id))[contained]
#         exams_id_to_export = [(np.where(traces_ids == exam_to_export[0]))[0][0]]
#         # , (np.where(traces_ids == exam_to_export[1]))[0][0]]
#         # for m in exams_id_to_export:
#         data = x[exams_id_to_export[0], :, :]
#         f = plt.figure()
#         f.subplots_adjust(hspace=1, wspace=0.5)
#
#         axes = []
#         ax = plt.subplot(signals_num / 6, 1, 1)
#
#         # ylims = (-1.5, 1.5)
#         start_time = 0  # beginning position in seconds
#         time = 10.24  # data length in seconds
#         sample_length = int(time * fs)  # data length in sample units
#         end_time = start_time + time
#         t = np.arange(start_time, end_time, 1 / fs)
#         vl1 = np.arange(start_time, end_time, 0.04)
#         vl2 = np.arange(start_time, end_time, 0.2)
#
#         for j in np.arange(0, 12, 2):
#             f = plt.figure()
#             f.subplots_adjust(hspace=1, wspace=0.5)
#             for i in np.arange(0 + j, 2 + j):
#                 MaxAmp = round(np.max(data[:, i]) + 0.5)
#                 MinAmp = round(np.min(data[:, i]) - 0.5)
#                 ax = plt.subplot(signals_num / 6, 1, i + 1 - j)
#                 axes.append(ax)
#
#                 plt.ylabel('voltage (mV)')
#
#                 # draw pink grid
#                 hl1 = np.arange(MinAmp, MaxAmp, 0.1)
#                 hl2 = np.arange(MinAmp, MaxAmp, 0.5)
#
#                 ax.vlines(vl1, MinAmp, MaxAmp, colors='r', linestyles='-', alpha=0.2, linewidth=0.4)
#                 ax.hlines(hl1, start_time, end_time, colors='r', linestyles='-', alpha=0.2, linewidth=0.4)
#                 ax.vlines(vl2, MinAmp, MaxAmp, colors='r', linestyles='-', alpha=0.6, linewidth=0.4)
#                 ax.hlines(hl2, start_time, end_time, colors='r', linestyles='-', alpha=0.6, linewidth=0.4)
#
#                 # read data for specified signal
#                 # equal to record.read(i, ...
#                 # draw signal
#                 ax.plot(t, data[:, i], linewidth=1, color='k', alpha=1.0)
#                 ax.tick_params(axis="y", labelsize=8)
#                 ax.set_title(str(lead_index[i]), color='k')
#                 # ax.text(0.55,0.02, ('patient id: %d, exam id: %d' %(patient['id_exam'],patient['id_patient'])), transform=ax.transAxes)
#                 ax.annotate(('patient id: %d, exam id: %d' % (k, exam_to_export[0])), xy=(1, 0),
#                             xycoords='axes fraction',
#                             fontsize=8,
#                             horizontalalignment='right', verticalalignment='bottom')
#                 plt.ylim(MinAmp, MaxAmp)
#                 plt.xlim(start_time, end_time)
#
#             plt.xlabel('time (s)')
#             xticklabels = [a.get_xticklabels() for a in axes[:-1]]
#             plt.setp(xticklabels, visible=False)
#             plt.tight_layout()
#
#             plt.show()
#
#             f.savefig(path + '/' + str(exam_to_export[0]) + '_' + str(j / 2) + '.png', dpi=800)
#         m = [(x / 2) for x in np.arange(0, 12, 2)]
#
#         im_name = [path + '/' + str(exam_to_export[0]) + '_' + str(n) + '.png' for n in m]
#         images1 = [Image.open(x) for x in im_name[:3]]
#         images2 = [Image.open(x) for x in im_name[3:]]
#
#         widths, heights = zip(*(i.size for i in images1))
#
#         gap = 300
#         max_width = max(widths)
#         total_height = sum(heights) - (len(images1) - 1) * gap
#
#         new_im1 = Image.new('RGB', (max_width, total_height))
#         new_im2 = Image.new('RGB', (max_width, total_height))
#         final_image = Image.new('RGB', (2 * max_width, total_height))
#         y_offset = 0
#         x_offset = max_width
#         for im1, im2 in zip(images1, images2):
#             new_im1.paste(im1, (0, y_offset))
#             new_im2.paste(im2, (0, y_offset))
#             y_offset += im1.size[1] - gap
#
#         final_image.paste(new_im1, (0, 0))
#         final_image.paste(new_im2, (x_offset, 0))
#         final_image.save(path + '/' + str(exam_to_export[0]) + '.png')


def plot_ecg_fig_MD(
        ecg,
        lead_index,
        sample_rate=500,
        title='ECG 12',
        lead_order=None,
        style=None,
        columns=2,
        row_height=6,
        show_lead_name=True,
        show_grid=True,
        show_separate_line=True,
        peaks=False,
        peak_dict=None,
        detector='epltd0',
):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of eclg, wil be shown on
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order
        columns    : display columns, defaults to 2
        style      : display style, defaults to None, can be 'bw' which means black white
        row_height :   how many grid should a lead signal have,
        show_lead_name : show lead name
        show_grid      : show grid
        show_separate_line  : show separate line
    """

    if not lead_order:
        lead_order = list(range(0, len(ecg)))
    secs = len(ecg[0]) / sample_rate
    leads = len(lead_order)
    rows = int(ceil(leads / columns))
    # display_factor = 2.5
    display_factor = 1
    line_width = 0.5
    fig, ax = plt.subplots(figsize=(secs * columns * display_factor, rows * row_height / 5 * display_factor))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(
        hspace=0,
        wspace=0,
        left=0,  # the left side of the subplots of the figure
        right=1,  # the right side of the subplots of the figure
        bottom=0,  # the bottom of the subplots of the figure
        top=1
    )

    fig.suptitle(title)

    x_min = 0
    x_max = columns * secs
    y_min = row_height / 4 - (rows / 2) * row_height
    y_max = row_height / 4

    if (style == 'bw'):
        color_major = (0.4, 0.4, 0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line = (0, 0, 0)
    else:
        color_major = (1, 0, 0)
        color_minor = (1, 0.7, 0.7)
        color_line = (0, 0, 0.7)

    if (show_grid):
        ax.set_xticks(np.arange(x_min, x_max, 0.2))
        ax.set_yticks(np.arange(y_min, y_max, 0.5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.minorticks_on()

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < leads):
                y_offset = -(row_height / 2) * ceil(i % rows)
                # if (y_offset < -5):
                #     y_offset = y_offset + 0.25

                x_offset = 0
                if (c > 0):
                    x_offset = secs * c
                    if (show_separate_line):
                        ax.plot([x_offset, x_offset],
                                [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3],
                                linewidth=line_width * display_factor, color=color_line, zorder=2, )

                t_lead = lead_order[c * rows + i]

                step = 1.0 / sample_rate
                if (show_lead_name):
                    ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9 * display_factor, zorder=2)
                ax.plot(
                    np.arange(0, len(ecg[t_lead]) * step, step) + x_offset,
                    ecg[t_lead] + y_offset,
                    linewidth=line_width * display_factor,
                    color=color_line, zorder=4
                )
                if peaks:
                    yvals = ecg[t_lead] + y_offset
                    tvals = np.arange(0, len(ecg[t_lead]) * step, step) + x_offset
                    peakvals = peak_dict[detector][str(t_lead)]
                    ax.scatter(tvals[peakvals], yvals[peakvals], zorder=3)
    plt.show()
    # path_to_csv = '/MLdata/AIMLab/databases/tnmg/ecg-traces/ecg-traces/preprocessed/traces.hdf5'
    # f = h5py.File(path_to_csv, 'r')
    # # Get ids
    # traces_ids = np.array(f['id_exam'])
    # x = f['signal']
    #
    # signals_num = 12
    # rows = int(round(signals_num/columns))
    # row_height = 6
    # for idx, k in enumerate(exams_id):
    #     print("ploting MD ecg recording for recording id: " + str(k))
    #     path = path_to_save
    #     exams_id_to_export = [(np.where(traces_ids == k))[0][0]]
    #     data = x[exams_id_to_export[0], :, :]
    #     axes = []
    #     ecg_len = len(data[:, 0]) / fs
    #     f = plt.figure(figsize=(ecg_len*columns, rows * row_height / 5))
    #     gs = mpl.gridspec.GridSpec(3, 1)
    #     gs.update(wspace=0, hspace=0)
    #     # mpl.rcParams['axes.linewidth'] = 0.1  # set the value globally
    #     # f.subplots_adjust(hspace=0, wspace=0)
    #     idx_loc = 0
    #     for i in np.arange(rows):
    #         ldl = lead_index[i::rows]
    #         line = data[:, i::rows]  # get only leads {I, aVR, V1, V4}
    #         start = (line[:, 0] != 0).argmax(axis=0)
    #         end = len(line) - (line[::-1, 1] != 0).argmax(axis=0)
    #         if filt:
    #             filtered = np.zeros(shape=(line.shape))
    #             for d in range(4):
    #                 ecg_lead = line[:, d]
    #                 ecg_filtered = bandpass_filter(ecg_lead, k, d+idx_loc, 0.67, 100, fs, 75, debug=False)
    #                 # ecg_filtered = ecg_filtered - bandpass_filter(ecg_filtered, 48, 52, fs, 3)
    #                 # ecg_filtered = ecg_filtered - bandpass_filter(ecg_filtered, 58, 62, fs, 3)
    #                 filtered[:,d] = ecg_filtered
    #             line=filtered
    #         line = (line[start:end, :]).flatten('F')
    #         time = (end - start) / fs  # data length in seconds
    #         start_time = 0  # beginning position in seconds
    #         end_time = start_time + time
    #         end_time_of_plot = 4 * end_time
    #         MaxAmp = round(np.max(line) + 0.5)
    #         MinAmp = round(np.min(line) - 0.5)
    #         ax = plt.subplot(gs[i])
    #         axes.append(ax)
    #         # plt.ylabel('Amplitude (mV)')
    #         t = np.arange(start_time, end_time_of_plot, 1 / fs)
    #
    #         # draw pink grid
    #         vl1 = np.arange(start_time, end_time_of_plot, 0.04)
    #         vl2 = np.arange(start_time, end_time_of_plot, 0.2)
    #
    #         hl1 = np.arange(MinAmp, MaxAmp, 0.1)
    #         hl2 = np.arange(MinAmp, MaxAmp, 0.5)
    #
    #         ax.vlines(vl1, MinAmp, MaxAmp, colors='r', linestyles='-', alpha=0.2, linewidth=0.4)
    #         ax.hlines(hl1, start_time, end_time_of_plot, colors='r', linestyles='-', alpha=0.2, linewidth=0.4)
    #         ax.vlines(vl2, MinAmp, MaxAmp, colors='r', linestyles='-', alpha=0.6, linewidth=0.4)
    #         ax.hlines(hl2, start_time, end_time_of_plot, colors='r', linestyles='-', alpha=0.6, linewidth=0.4)
    #
    #         # draw lead location
    #         ld = np.arange(0, end_time_of_plot, end_time)
    #         ld[0] += 0.3
    #         ax.vlines(ld, line.mean() - 0.5, line.mean() + 0.5, colors='b', linestyles='-', alpha=1, linewidth=1.2)
    #         # read data for specified signal
    #         # equal to record.read(i, ...
    #         # draw signal
    #         ax.plot(t.astype(float)[:len(line)], line, linewidth=0.5, color='k', alpha=1.0)
    #         plt.setp(ax.spines.values(), color='r', alpha=0.6, linewidth=0.4)
    #         # ax.set(frame_on=False)
    #         # ax.tick_params(axis="y", labelsize=8)
    #         # ax.set_title(idx, color='k')
    #         # plt.text(0.55, 0.02, ('exam id: %d' % k), transform=ax.transAxes)
    #         if i == 2:
    #             ax.annotate(('exam id: %d' % (k, )), xy=(1, 0),
    #                         xycoords='axes fraction',
    #                         color='b', fontsize=16,
    #                         horizontalalignment='right', verticalalignment='bottom')
    #         for mm, loc in enumerate(ld):
    #             ax.annotate(ldl[mm], xy=(loc, line.mean() - 0.8), textcoords="offset points",
    #                         color='b', fontsize=18, xytext=(0.2, -2))
    #         plt.ylim(MinAmp, MaxAmp)
    #         plt.xlim(start_time, end_time_of_plot)
    #
    #         # plt.xlabel('time (s)')
    #         # xticklabels = [a.get_xticklabels() for a in axes[:-1]]
    #         plt.xticks([])
    #         plt.yticks([])
    #         idx_loc = idx_loc+4
    #     plt.suptitle(idx, color='k', fontsize=20)
    #     plt.tight_layout()
    #
    fig.savefig('/home/sheina/VT/vt_fig', dpi=400, transparent=True)
    plt.close()
    return


# def plot_ecg_as_Antonio(exams_id, namedfile, path_to_save, fs=400, filt=False, format="png"):
#     path_to_csv = '/MLdata/AIMLab/databases/tnmg/ecg-traces/ecg-traces/preprocessed/traces.hdf5'
#     f = h5py.File(path_to_csv, 'r')
#     # Get ids
#     traces_ids = np.array(f['id_exam'])
#     x = f['signal']
#
#     signals_num = 3
#     lead_index = {1: 'DII', 6: 'V1', 11: 'V6'}
#     # f = plt.figure(figsize=(7, 6))
#     j=0
#     axes = []
#     gs = mpl.gridspec.GridSpec(3, 3, width_ratios=[3, 1, 1])
#     gs.update(wspace=0.03, hspace=0.3)
#
#
#     f=plt.figure(figsize=(11,2))
#     plt.subplots_adjust(
#         hspace = 0.3,
#         wspace = 0.03,
#         # left   = 0.04,  # the left side of the subplots of the figure
#         # right  = 0.98,  # the right side of the subplots of the figure
#         # bottom = 0.2,   # the bottom of the subplots of the figure
#         # top    = 0.88
#         )
#
#     for idx, k in exams_id.iterrows():
#         print("ploting MD ecg recording for recording id: " + str(k))
#         path = path_to_save
#         exams_id_to_export = [(np.where(traces_ids == k.id_exam))[0][0]]
#         data = x[exams_id_to_export[0], :, :]
#         # mpl.rcParams['axes.linewidth'] = 0.1  # set the value globally
#         # f.subplots_adjust(hspace=0, wspace=0)
#         for i in lead_index.keys():
#             ldl = lead_index[i]
#             line = data[:, i]  # get only leads {I, aVR, V1, V4}
#             if i==1:
#                 length_ecg = 5 # sec
#             else:
#                 length_ecg=2.5
#             start = round(len(line)/4)
#             end = round(start + length_ecg*fs)
#             if filt:
#                 ecg_filtered = bandpass_filter(line, k, i, 0.67, 100, fs, 75, debug=False)
#                 line=ecg_filtered
#             line = (line[start:end]).flatten('F')
#             time = (end - start) / fs  # data length in seconds
#             start_time = 0  # beginning position in seconds
#             end_time = start_time + time
#             end_time_of_plot = 4 * end_time
#             MaxAmp = round(np.max(line) + 0.5)
#             MinAmp = round(np.min(line) - 0.5)
#             seconds = len(line) / fs
#
#             ax = plt.subplot(gs[j])
#             # plt.rcParams['lines.linewidth'] = 5
#             step = 1.0 / fs
#             # ecg_plot.plot_1(line, 400, ecg_amp=4, fig_width=4)
#             _ax_plot(ax, np.arange(0, len(line) * step, step), line, seconds, 0.5, 4, 0.2)
#
#             axes.append(ax)
#             # plt.ylabel('Amplitude (mV)')
#             # t = np.arange(start_time, end_time, 1 / fs)
#             # vl1 = np.arange(start_time, end_time, 0.04)
#             # vl2 = np.arange(start_time, end_time, 0.2)
#
#             # draw pink grid
#             # hl1 = np.arange(MinAmp, MaxAmp, 0.1)
#             # hl2 = np.arange(MinAmp, MaxAmp, 0.5)
#             #
#             # ax.vlines(vl1, MinAmp, MaxAmp, colors='r', linestyles='-', alpha=0.2, linewidth=0.4)
#             # ax.hlines(hl1, start_time, end_time, colors='r', linestyles='-', alpha=0.2, linewidth=0.4)
#             # ax.vlines(vl2, MinAmp, MaxAmp, colors='r', linestyles='-', alpha=0.6, linewidth=0.4)
#             # ax.hlines(hl2, start_time, end_time, colors='r', linestyles='-', alpha=0.6, linewidth=0.4)
#
#             # read data for specified signal
#             # equal to record.read(i, ...
#             # draw signal
#             # ax.plot(t.astype(float)[:len(line)], line, linewidth=0.5, color='k', alpha=1.0)
#             #
#             # plt.setp(ax.spines.values(), color='k', alpha=1, linewidth=1)
#             # ax.set(frame_on=False)
#             # ax.tick_params(axis="y", labelsize=8)
#             # ax.set_title(idx, color='k')
#             # plt.text(0.55, 0.02, ('exam id: %d' % k), transform=ax.transAxes)
#             # plt.ylim(MinAmp, MaxAmp)
#             # plt.xlim(start_time, end_time)
#
#             # plt.xlabel('time (s)')
#             # xticklabels = [a.get_xticklabels() for a in axes[:-1]]
#             # plt.xticks([])
#             # plt.yticks([])
#             # ax.set_xticklabels([])
#             # ax.set_yticklabels([])
#             j = j+1
#     plt.tight_layout()
#
#     f.savefig(path + '/' + namedfile + '.' + str(format), format=format, dpi=400, transparent=True)
#     plt.close()

if __name__ == '__main__':
    a = 5
    # db = TNMGDB_Parser()
    # exams = pd.read_csv("/MLdata/AIMLab/ShanySheina/dataset/"+ db.name + "/exams_info.csv")
    # # plot_ecg_fig_MD(df=[db.traces_ids[0]], patient_id=[db.traces_ids[0]], exams_id=[db.traces_ids[0]],
    # #                 namedfile =str(db.traces_ids[0]), lead_index = db.lead_index,
    # #                 path_to_save="/home/shanybiton/repos/afib-prediction/",
    # #                 format="png")
    # lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    # peak_dir = cts.ANN_DIR / db.name / "Other"
    # f = h5py.File(db.path_to_hdf5, 'r')
    # traces = f['signal']
    # ids_C = f['id_exam']
    #
    # # ecg = reorganize_data(data, cts.lead_index, db.lead_dict, db.actual_fs, db.orig_fs)
    # # ecg_plot.plot(ecg.T, sample_rate=400, title='ECG 12')
    # # for id in ids_in[:20]:
    # id = 3082058
    # loc = np.where(ids_C[()] == id)[0][0]
    # data = db.get_trace(id)
    # peaks_filename = str(id) + '_peaks_dict.npy'
    # peaks = np.load(peak_dir / peaks_filename, allow_pickle=True).item()
    #
    # plot_ecg_fig_MD(traces[loc].T, sample_rate=400, title=' ', lead_index=lead_index, peaks=True, peak_dict=peaks,
    #                 detector='epltd0')
    # plt.savefig("/home/shanybiton/repos/afib-prediction/AIMLab_report"/db.name/"12lead_examples/resampled_" + str(
    #     id) + ".pdf", dpi=400)
    # plot_ecg_fig_MD(data.T, sample_rate=500, title=' ', lead_index=db.lead_index, peaks=False, )
    # plt.savefig(
    #     "/home/shanybiton/repos/afib-prediction/AIMLab_report"/db.name/"12lead_examples/orig_" + str(id) + ".pdf",
    #     dpi=400)
