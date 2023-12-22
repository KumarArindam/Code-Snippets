import os
from moviepy.editor import VideoFileClip
import auditok
import soundfile
import re
import torch
import datetime
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import accelerate
import cv2
import time
import imutils
import shutil
import glob
import argparse
import easyocr
import subprocess
from PIL import Image
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
from pathlib import Path


class SubtitleGenerator():
    def __init__(self, model_path, video_dir, audio_dir, subtitle_dir, segment_dir):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.subtitle_dir = subtitle_dir
        self.segment_dir = segment_dir
        self.files = os.listdir(video_dir)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_path = model_path
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model=self.model,
                    tokenizer=self.processor.tokenizer,
                    feature_extractor=self.processor.feature_extractor,
                    max_new_tokens=3000,
                    torch_dtype=self.torch_dtype,
                    device=self.device,
                )
       
    @staticmethod
    def extract_audio(input_path, output_path):
        audio_file = output_path
        command = ["ffmpeg", "-i", input_path, "-ac", "1", "-ar", "16000","-vn", "-f", "wav", audio_file]
        subprocess.run(command)
        
    @staticmethod
    def clean_text(text):
        clean_text = re.sub(r'  ', ' ', text)
        clean_text = re.sub(r'\bi\s', 'I ', clean_text)
        clean_text = re.sub(r'\si$', ' I', clean_text)
        clean_text = re.sub(r'i\'', 'I\'', clean_text)
        return clean_text
    
    @staticmethod
    def get_srt_line(inferred_text, line_count, limits):
        sep = ','   
        d = str(datetime.timedelta(seconds=float(limits[0])))
        if '.' in list(d):
            from_dur = '0' + str(d.split(".")[0]) + sep + str(d.split(".")[-1][:2])
        else:
            from_dur = '0' + str(d) + sep + '00'

        d = str(datetime.timedelta(seconds=float(limits[1])))
        if '.' in list(d):
            to_dur = '0' + str(d.split(".")[0]) + sep + str(d.split(".")[-1][:2])
        else:
            to_dur = '0' + str(d) + sep + '00'
        return f'{str(line_count)}\n{from_dur} --> {to_dur}\n{inferred_text}\n\n'
    
    @staticmethod
    def timestamp_formatter(time_stamp_str):
        sep = ','   
        d = str(datetime.timedelta(seconds=float(time_stamp_str)))
        if '.' in list(d):
            dur = '0' + str(d.split(".")[0]) + sep + str(d.split(".")[-1][:2])
        else:
            dur = '0' + str(d) + sep + '00'
        return dur

    def extract_all_audio(self):
        for i in self.files:
            self.extract_audio(self.video_dir + '/' + i, self.audio_dir + '/' + i.split(".")[0] + '.wav')
            
    def segment_audio(self, audio_name):
        os.mkdir(self.segment_dir + '/' + audio_name[:-4])
        audio_regions = auditok.split(self.audio_dir + '/'+ audio_name,
            min_dur=1,       # minimum duration of a valid audio in seconds
            max_dur=8,       # maximum duration of an audio segment
            max_silence=0.8, # maximum duration of tolerated continuous silence within an event
            energy_threshold=55, # threshold of detection
            sampling_rate=16000
          )
        for i, r in enumerate(audio_regions):
            filename = r.save(self.segment_dir + '/' + audio_name[:-4] + '/' + audio_name[:-4]+f'_{r.meta.start:08.3f}-{r.meta.end:08.3f}.wav')
            
            
    def get_subs(self, single_seg_directory, output_file):
        segments = sorted([f for f in Path(single_seg_directory).glob(f'*.wav')])
        line_count = 0
        transcript_dataframe = pd.DataFrame()

        with open(output_file, 'w', encoding="utf-8") as out_file:
            for audio_file in segments:
                # Run OpenAI Whisper inference on each segemented audio file.
                speech, rate = soundfile.read(audio_file) 
                output_json = self.pipe(speech)
                inferred_text = output_json['text']

                if len(inferred_text) > 0:
                    inferred_text = self.clean_text(inferred_text)
                else:
                    inferred_text = ''

                limits = audio_file.name[:-4].split("_")[-1].split("-")
                limits = [float(limit) for limit in limits]
                transcript_dataframe = pd.concat([transcript_dataframe, 
                            pd.DataFrame([{'Video':single_seg_directory.split('/')[-1], "Text":inferred_text,
                              "start_time": self.timestamp_formatter(limits[0]),
                              "end_time": self.timestamp_formatter(limits[1])}])])
                out_file.write(self.get_srt_line(inferred_text, line_count, limits))
                out_file.flush()
                line_count += 1
            return transcript_dataframe
        
    def main(self):
        self.extract_all_audio()
        audio_names = os.listdir(self.audio_dir)
        for i in audio_names:
            self.segment_audio(i)
            single_seg_dir = self.segment_dir + '/' + i[:-4]
            output_file = self.subtitle_dir + '/' +  i[:-4] + '.srt'
            transcript_dataframe = self.get_subs(single_seg_dir, output_file)
            transcript_dataframe.to_excel(self.subtitle_dir + '/' +  i[:-4] + '.xlsx', index=False)


model_path = "../models/distil-large-v2"
video_dir = "../Data/Video"
audio_dir = "../Data/Audio"
subtitle_dir="../Data/Subtitle"
segment_dir= "../Data/Audio_Segments"

Subtitle_gen = SubtitleGenerator(model_path, video_dir, audio_dir, subtitle_dir, segment_dir)
Subtitle_gen.main()

class Video_text_extractor():
    def __init__(self, video_dir, screenshots_dir, ocr_dir, FRAME_RATE = 3, VAR_THRESHOLD = 16, DETECT_SHADOWS = False, MIN_PERCENT = 0.1, MAX_PERCENT = 3):
        self.video_dir = video_dir
        self.screenshots_dir = screenshots_dir
        self.ocr_dir = ocr_dir
        self.files = os.listdir(video_dir)
        self.FRAME_RATE = FRAME_RATE                   # no.of frames per second that needs to be processed, fewer the count faster the speed
        self.WARMUP = FRAME_RATE              # initial number of frames to be skipped
        self.FGBG_HISTORY = FRAME_RATE * 15   # no.of frames in background object
        self.VAR_THRESHOLD = VAR_THRESHOLD               # Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model.
        self.DETECT_SHADOWS = DETECT_SHADOWS            # If true, the algorithm will detect shadows and mark them.
        self.MIN_PERCENT = MIN_PERCENT                # min % of diff between foreground and background to detect if motion has stopped
        self.MAX_PERCENT = MAX_PERCENT                  # max % of diff between foreground and background to detect if frame is still in motion
        
    def get_frames(self, video_path):
        '''A function to return the frames from a video located at video_path
        this function skips frames as defined in FRAME_RATE'''
        # open a pointer to the video file initialize the width and height of the frame
        vs = cv2.VideoCapture(video_path)
        if not vs.isOpened():
            raise Exception(f'unable to open file {video_path}')
        total_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_time = 0
        frame_count = 0
        print("total_frames: ", total_frames)
        print("FRAME_RATE", self.FRAME_RATE)
        # loop over the frames of the video
        while True:
            # grab a frame from the video
            vs.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)    # move frame to a timestamp
            frame_time += 1/self.FRAME_RATE
            (_, frame) = vs.read()
            # if the frame is None, then we have reached the end of the video file
            if frame is None:
                break
            frame_count += 1
            yield frame_count, frame_time, frame
        vs.release()
        
    def initialize_output_folder(self, video_path):
        '''Clean the output folder if already exists'''
        output_folder_screenshot_path = f"{self.screenshots_dir}/{video_path.rsplit('/')[-1].split('.')[0]}"
        if os.path.exists(output_folder_screenshot_path):
            shutil.rmtree(output_folder_screenshot_path)
        os.makedirs(output_folder_screenshot_path, exist_ok=True)
        print('initialized output folder', output_folder_screenshot_path)
        return output_folder_screenshot_path
    
    def detect_unique_screenshots(self, video_path, output_folder_screenshot_path):
        # Initialize fgbg a Background object with Parameters
        # history = The number of frames history that effects the background subtractor
        # varThreshold = Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model. This parameter does not affect the background update.
        # detectShadows = If true, the algorithm will detect shadows and mark them. It decreases the speed a bit, so if you do not need this feature, set the parameter to false.
        fgbg = cv2.createBackgroundSubtractorMOG2(history = self.FGBG_HISTORY, varThreshold = self.VAR_THRESHOLD, detectShadows = self.DETECT_SHADOWS)
        captured = False 
        start_time = time.time()
        (W, H) = (None, None)
        screenshoots_count = 0
        for frame_count, frame_time, frame in self.get_frames(video_path):
            orig = frame.copy() # clone the original frame (so we can save it later), 
            frame = imutils.resize(frame, width=600) # resize the frame
            mask = fgbg.apply(frame) # apply the background subtractor
            # apply a series of erosions and dilations to eliminate noise
            # eroded_mask = cv2.erode(mask, None, iterations=2)
            # mask = cv2.dilate(mask, None, iterations=2)
            # if the width and height are empty, grab the spatial dimensions
            if W is None or H is None:
                (H, W) = mask.shape[:2]
            # compute the percentage of the mask that is "foreground"
            p_diff = (cv2.countNonZero(mask) / float(W * H)) * 100
            # if p_diff less than N% then motion has stopped, thus capture the frame
            if p_diff < self.MIN_PERCENT and not captured and frame_count > self.WARMUP:
                captured = True
                filename = f"{screenshoots_count:03}_{round(frame_time/60, 2)}.png"
                path = os.path.join(output_folder_screenshot_path, filename)
                print("saving {}".format(path))
                cv2.imwrite(path, orig)
                screenshoots_count += 1
            # otherwise, either the scene is changing or we're still in warmup
            # mode so let's wait until the scene has settled or we're finished
            # building the background model
            elif captured and p_diff >= self.MAX_PERCENT:
                captured = False
        print(f'{screenshoots_count} screenshots Captured!')
        print(f'Time taken {time.time()-start_time}s')
        return 
            
    @staticmethod
    def extract_text_from_screenshot(image_path):
        reader = easyocr.Reader(['en']) # Replace 'en' with the language code if needed
        result = reader.readtext(image_path)
        text_info = []
        for detection in result:
            text = detection[1]
            box = detection[0]
            text_info.append({
                'text': text,
                'height': box[3][1] - box[0][1],
                'left_x': box[0][0],
                'top_y': box[0][1],
                'right_x':box[1][0],
                'bottom_y': box[2][1],
            })
        return pd.DataFrame(text_info)
    
    @staticmethod
    def add_linenum(text_dataframe):
        linenum = 0
        text_with_linenum = pd.DataFrame()
        while len(text_dataframe)>0:
            y_thresh = text_dataframe.iloc[0,:]['bottom_y']
            text_with_linenum_iter = text_dataframe[text_dataframe['top_y']<y_thresh]
            text_with_linenum_iter.loc[:,'linenum'] = linenum
            text_with_linenum = pd.concat([text_with_linenum, text_with_linenum_iter])
            text_dataframe = text_dataframe[text_dataframe['top_y']>=y_thresh]
            text_dataframe.reset_index(inplace=True, drop = True)
            linenum = linenum + 1
        text_with_linenum.reset_index(inplace=True, drop = True)
        return text_with_linenum
    
    @staticmethod
    def clean_linewise(text_data_per_line):
        text_data_per_line.reset_index(inplace=True, drop=True)
        multiplier = 1.1
        final_data = pd.DataFrame()
        while len(text_data_per_line)>0:
            text_iterator = text_data_per_line.iloc[0, :]
            merge_thresh = text_iterator['right_x'] + multiplier * text_iterator['height']
            text_data_per_line = text_data_per_line.iloc[1:,:]
            merge_data =  text_data_per_line[text_data_per_line['left_x'] < merge_thresh]
            if len(merge_data) > 0:
                text_data_per_line = text_data_per_line[text_data_per_line['left_x'] >= merge_thresh]
                for index, row in merge_data.iterrows():
                    text_iterator['text'] = text_iterator['text'] + ' ' +row['text']
                    text_iterator['right_x'] = row['right_x']
                text_data_per_line = pd.concat([pd.DataFrame(text_iterator).transpose(), text_data_per_line])
                text_data_per_line.reset_index(inplace = True, drop = True)
            else:
                final_data = pd.concat([final_data, pd.DataFrame(text_iterator).transpose()])
        final_data.reset_index(inplace=True, drop=True)
        return final_data
    
    def horizontal_merge_clean(self, text_dataframe):
        linenum_unq = text_dataframe['linenum'].unique()
        clean_full_data = pd.DataFrame()
        for i in linenum_unq:
            text_data_per_line = text_dataframe[text_dataframe['linenum'] == i]
            clean_text_data_per_line = self.clean_linewise(text_data_per_line)
            clean_full_data = pd.concat([clean_full_data, clean_text_data_per_line])
        clean_full_data.reset_index(inplace = True, drop=True)
        return clean_full_data
    
    @staticmethod
    def cluster_and_compile(text_dataframe, features, max_cluster=5):
        best_score = -1
        best_feature = None
        best_cluster = 2

        for feature in features:
            X = text_dataframe[[feature]]
            if X.nunique()[0]==1:
                labels = np.zeros(len(X)).astype(int)
                break
            if len(X)==2:
                labels = np.zeros(len(X)).astype(int)
                break
            for n_clusters in range(2,min(len(X),5)):
                clustering = AgglomerativeClustering(n_clusters = n_clusters)
                labels = clustering.fit_predict(X)
                silhouette_avg = silhouette_score(X, labels)
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_feature = feature
                    best_cluster = n_clusters

        text_dataframe['labels'] = labels
        text_dataframe['labels'] = text_dataframe['labels'].astype(str)
        cluster_info = text_dataframe.groupby('linenum', as_index=False).agg({'labels': ', '.join})
        cluster_info['cluster_pattern_group'] = np.cumsum(~np.concatenate(([False], np.array(cluster_info['labels'])[1:] == np.array(cluster_info['labels'])[:-1])))
        cluster_info.drop('labels', axis =1, inplace = True)
        text_dataframe = text_dataframe.merge(cluster_info, 'inner', on = 'linenum')
        Processed_text = text_dataframe.groupby(['cluster_pattern_group', 'labels'], as_index=False).agg({'text': '\n'.join}) 
        Processed_text = '\n\n'.join(Processed_text['text'])
        heading = text_dataframe['text'][np.argmax(text_dataframe['height'])]
        return Processed_text, heading
    
    @staticmethod
    def timestamp_formatter(time_stamp_str):
        sep = ','   
        d = str(datetime.timedelta(seconds=float(time_stamp_str)))
        if '.' in list(d):
            dur = '0' + str(d.split(".")[0]) + sep + str(d.split(".")[-1][:2])
        else:
            dur = '0' + str(d) + sep + '00'
        return dur
    
    def compile_text_per_video(self, screenshots):
        files = [f for f in os.listdir(screenshots) if f.endswith('.png')]
        text_file = pd.DataFrame()
        for i in files:
            print(i)
            text_dataframe = self.extract_text_from_screenshot(screenshots + '/' + i)
            if len(text_dataframe) == 0:
                continue
            text_dataframe = self.add_linenum(text_dataframe)
            text_dataframe = self.horizontal_merge_clean(text_dataframe)
            text_dataframe['centroid_x'] = text_dataframe['left_x'] + text_dataframe['right_x']
            features = ['centroid_x', 'left_x']
            Processed_text, heading = self.cluster_and_compile(text_dataframe, features, max_cluster=5)
            i_timestamp = i.replace('.png', '')
            i_timestamp = i_timestamp.split('_')[1]
            i_timestamp = str(float(i_timestamp)*60)
            i_timestamp = self.timestamp_formatter(i_timestamp)
            video = screenshots.split('/')[-1]
            slide_text = pd.DataFrame({'Video':[video], 'start_time':[i_timestamp], 'slide_text':[Processed_text], 'header':[heading]})
            text_file = pd.concat([text_file, slide_text])
        return text_file
    
    def main(self):
        for i in self.files:
            output_folder_screenshot_path = self.initialize_output_folder(self.video_dir + '/' + i)
            self.detect_unique_screenshots(self.video_dir + '/' + i, output_folder_screenshot_path)
            text_file = self.compile_text_per_video(output_folder_screenshot_path)
            text_file.to_excel(self.ocr_dir + '/' + i[:-4] + '.xlsx', index = False)


video_dir= "../Data/Video"
screenshots_dir= "../Data/Slides"
ocr_dir="../Data/OCR"

Video_proessor = Video_text_extractor(video_dir, screenshots_dir, ocr_dir)
Video_proessor.main()

#### Final

model_path = "../models/distil-large-v2"
video_dir = "../Data/Video"
audio_dir = "../Data/Audio"
subtitle_dir ="../Data/Subtitle"
segment_dir = "../Data/Audio_Segments"
screenshots_dir = "../Data/Slides"
ocr_dir ="../Data/OCR"


def Process_video(Transcription_model_path, video_dir, audio_dir, subtitle_dir, segment_dir, screenshots_dir, ocr_dir)
    Subtitle_gen = SubtitleGenerator(Transcription_model_path, video_dir, audio_dir, subtitle_dir, segment_dir)
    Subtitle_gen.main()
    Video_proessor = Video_text_extractor(video_dir, screenshots_dir, ocr_dir)
    Video_proessor.main()


def merge_files(subtitle_dir, ocr_dir, Results_file):
    subtitle_files = list(pd.Series(os.listdir(subtitle_dir))[pd.Series(os.listdir(subtitle_dir)).str.contains('xlsx')]) 
    ocr_files = list(pd.Series(os.listdir(ocr_dir))[pd.Series(os.listdir(ocr_dir)).str.contains('xlsx')]) 
    files = list(set(subtitle_file).intersection(set(ocr_files)))
    final_dataframe = pd.DataFrame()
    for i in files:
        sub_file = pd.read_excel(subtitle_dir + '/' + i)
        ocr_file = pd.read_excel(ocr_dir + '/' + i)
        ocr_file = ocr_file.sort_values('start_time')
        ocr_file.reset_index(inplace=True, drop = True)
        ocr_file['end_time'] = list(ocr_file['start_time'].iloc[1:].append(pd.Series(list(sub_file['end_time'])[-1])))
        ocr_file['ts_start_time'] = pd.to_datetime(ocr_file['start_time'], format = '%H:%M:%S,%f')
        ocr_file['ts_end_time'] = pd.to_datetime(ocr_file['end_time'], format = '%H:%M:%S,%f')

        sub_file['ts_start_time'] = pd.to_datetime(sub_file['start_time'], format = '%H:%M:%S,%f')
        sub_file['ts_end_time'] = pd.to_datetime(sub_file['end_time'], format = '%H:%M:%S,%f')
        ocr_file['Sub_text'] = pd.Series()
        for j in range(len(ocr_file)):
            start_t = ocr_file.loc[j]['ts_start_time']
            end_t = ocr_file.loc[j]['ts_end_time']
            relevant_data = sub_file[(sub_file['ts_end_time'] >= start_t) & (sub_file['ts_start_time'] <= end_t)]
            ocr_file['Sub_text'][j]=' '.join(relevant_data['Text'])
        final_dataframe = pd.concat([final_dataframe, ocr_file])
    final_dataframe.to_excel(Results_file , index=False)

merge_files(subtitle_dir, ocr_dir, '../Data/Finaldata.xlsx')

## Clean Data
import pandas as pd
from vllm import LLM, SamplingParams

merged_df = pd.read_excel('../Data/Finaldata.xlsx')
sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=3000)
llm = LLM(model="../models/openchat_3.5")


