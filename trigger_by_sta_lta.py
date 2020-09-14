import argparse
import pandas as pd
import glob
import os
import math
import shutil
import obspy
import fnmatch
from obspy.core import Stream, read, UTCDateTime
from obspy.clients.arclink import Client
from obspy.signal.trigger import recursive_sta_lta, classic_sta_lta, trigger_onset, ar_pick


def preprocess_stream(stream):
    stream = stream.detrend('constant')
    stream = stream.filter('bandpass', freqmin=0.5, freqmax=20)
    return stream


def main(args):
    times_csv = {"start_time": [], "end_time": [],
                 "utc_start_timestamp": [], "utc_end_timestamp": [], "stname": []}
    stream_path = args.stream_path
    stream_files = [file for file in os.listdir(stream_path) if
                    fnmatch.fnmatch(file, 'XX.MXI.2008207000000.mseed')]

    for file in stream_files:
        stream_path1 = os.path.join(stream_path, file)
        print("+ Loading Stream {}".format(file))
        st1 = read(stream_path1)
        st1 = preprocess_stream(st1)
        tr1 = st1[0]
        tr2 = st1[1]
        tr3 = st1[2]
        threechannels = st1
        msg = "%s %s %s" % (tr1.stats.station, str(
            tr1.stats.starttime), str(tr1.stats.endtime))
        print(msg)
        delta1 = UTCDateTime(tr1.stats.starttime)
        delta2 = UTCDateTime(tr1.stats.endtime)
        t1 = math.ceil(delta1.timestamp)
        t2 = delta2.timestamp
        print(t1, t2)
        if args.save_mseed:
            mseed_dir = os.path.join(args.output, "mseed")
            if os.path.exists(mseed_dir):
                shutil.rmtree(mseed_dir)
            os.makedirs(mseed_dir)
            output_mseed = os.path.join(
                mseed_dir, tr1.stats.station + "start.mseed")
            st1.slice(tr1.stats.starttime, tr1.stats.starttime +
                      args.window_size).write(output_mseed, format="mseed")
        for t3 in range(int(t1), int(t2), args.window_step):
            t = UTCDateTime(t3)
            #print("Cut a slice at time:",t,tr1.stats.station,tr1.stats.sac.stlo, tr1.stats.sac.stla,str(file))
            lsplit3 = '{:0>2s}'.format(str(t.hour))
            lsplit4 = '{:0>2s}'.format(str(t.minute))
            lsplit5 = '{:0>2s}'.format(str(t.second))
            lsplit1 = '{:0>2s}'.format(str(t.year))
            lsplit2 = '{:0>2s}'.format(str(t.month))
            lsplit6 = '{:0>2s}'.format(str(t.day))
            t4 = t + args.window_size
            st1_ = tr1.slice(t, t4)
            st2_ = tr2.slice(t, t4)
            st3_ = tr3.slice(t, t4)
            df = st1_.stats.sampling_rate
            # Characteristic function and trigger onsets
            #cft = recursive_sta_lta(st1_.data, int(2 * df), int(20. * df))
            try:
                cft = classic_sta_lta(st1_.data, int(0.5 * df), int(60. * df))
            #cft1 = recursive_sta_lta(st2_.data, int(2 * df), int(20. * df))
                cft1 = classic_sta_lta(st2_.data, int(0.5 * df), int(60. * df))
            #cft2 = recursive_sta_lta(st3_.data, int(2 * df), int(20. * df))
                cft2 = classic_sta_lta(st3_.data, int(0.5 * df), int(60. * df))
            except:
                continue
            #cft = classic_sta_lta(st_.data, int(2.5 * df), int(10. * df))
            on_of = trigger_onset(cft, 5, 1.2)
            on_of1 = trigger_onset(cft1, 5, 1.2)
            on_of2 = trigger_onset(cft2, 5, 1.2)
            if len(on_of) or len(on_of1) or len(on_of2):
                print(t, t4, tr1.stats.station)
                print(on_of, on_of1, on_of2)
                filename1 = tr1.stats.station + lsplit1 + \
                    lsplit2 + lsplit6 + lsplit3 + lsplit4 + lsplit5
                times_csv["start_time"].append(t)
                times_csv["utc_start_timestamp"].append(t.timestamp)
                times_csv["end_time"].append(t4)
                times_csv["utc_end_timestamp"].append(t4.timestamp)
                times_csv["stname"].append(filename1)

               # mseed_files = filename1+lsplit3+lsplit4+lsplit5 + '.mseed'
               # mseed_path = os.path.join(args.output, mseed_files)
               # threechannels.write(mseed_path, format="mseed")
                mseed_files = filename1 + '.mseed'
                print(mseed_files)
                if len(on_of):
                    min_on_of = min(on_of[:, 0])
                    max_on_of = max(on_of[:, 1])
                else:
                    min_on_of = args.window_size*df
                    max_on_of = 0
                if len(on_of1):
                    min_on_of1 = min(on_of1[:, 0])
                    max_on_of1 = max(on_of1[:, 1])
                else:
                    min_on_of1 = args.window_size*df
                    max_on_of1 = 0
                if len(on_of2):
                    min_on_of2 = min(on_of2[:, 0])
                    max_on_of2 = max(on_of2[:, 1])
                else:
                    min_on_of2 = args.window_size*df
                    max_on_of2 = 0
                minon_of = min(min_on_of, min_on_of1, min_on_of2)
                maxon_of = max(max_on_of, max_on_of1, max_on_of2)
                minon = int(minon_of/100)
                maxon = int(maxon_of/100)
                print(minon_of, maxon_of)

                if args.save_mseed:
                    mseed_dir = os.path.join(args.output, "mseed")
                    output_mseed = os.path.join(mseed_dir, mseed_files)
                    threechannels.slice(
                        t+minon, t+maxon).write(output_mseed, format="mseed")
                if args.plot:
                    viz_dir = os.path.join(args.output, "viz")
                    if not os.path.exists(viz_dir):
                        os.makedirs(viz_dir)
                    threechannels.slice(
                        t+minon, t+maxon).plot(outfile=os.path.join(viz_dir, mseed_files.split(".mseed")[0]+'.png'))
        threechannels.clear()
        st1.clear()

    df = pd.DataFrame.from_dict(times_csv)
    print(df.shape[0])
    output_catalog = os.path.join(args.output, 'detection.csv')
    df.to_csv(output_catalog)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_path", type=str, default="./data",
                        help="path to mseed to analyze")
    parser.add_argument("--output", type=str, default="./output",
                        help="dir of predicted events")
    parser.add_argument("--save_mseed", action="store_true",
                        help="dir of source mseed file")
    parser.add_argument("--plot", action="store_true",
                        help="pass flag to plot detected events in output")
    parser.add_argument("--window_size", type=int, default=120,
                        help="size of the window to analyze")
    parser.add_argument("--window_step", type=int, default=120,
                        help="size of the window to analyze")
    args = parser.parse_args()

    main(args)
