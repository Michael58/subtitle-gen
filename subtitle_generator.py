import time
import traceback
import os
import shutil
from datetime import timedelta
import re
import argparse

import ffmpeg
from google import genai
from pydub import AudioSegment
from dotenv import load_dotenv
from config import Config


def get_language_specific_instructions(language):
    """
    Generate language-specific instructions based on the target language.
    
    Different languages have different grammatical features that need special handling:
    - Some languages have grammatical gender (Slavic, Romance, Germanic languages)
    - Some have complex case systems 
    - Some have honorifics or formal/informal distinctions
    
    This function provides appropriate instructions based on the language.
    """
    # Languages with grammatical gender that affects word forms based on speaker gender
    languages_with_gender_morphing = [
        "slovak", "czech", "polish", "ukrainian", "serbian", 
        "croatian", "bulgarian", "slovenian", "german", "french", 
        "italian", "spanish", "portuguese", "romanian"
    ]
    
    # Basic instruction for all languages
    instruction = f"Convert the subtitles into {language}"
    
    # Add special handling for languages with grammatical gender
    if language.lower() in languages_with_gender_morphing:
        instruction += f" with proper male and female word morphing"
    
    return instruction

def get_prompt_first_segment(language):
    """Generate the first prompt with the appropriate language instructions."""
    language_instruction = get_language_specific_instructions(language)
    
    return f"""
        Make subtitles for the uploaded file in the srt format.
        {language_instruction}.
        
        Add empty lines between each text.
        Make sure that the timing is correct.
        
        Before first subtitle line, write **START**
        Always output in SRT format.
        Think about the content first, and translate it with the proper context
        that will make sense in {language} language.
        Make the text easily readable for subtitles, no very long texts, but rather split
        into meaningful parts. But, not too short either.
        
        Timestamps are in format hh:mm:ss,mmm --> hh:mm:ss,mmm
        Do you understand the format?
        
        Sample:
        **START**
        1
        00:00:00,008 --> 00:00:06,000
        [Sample subtitle in {language} would appear here]
        
        2
        00:00:06,000 --> 00:00:08,000
        [Another subtitle in {language}]
        
        3
        00:00:08,388 --> 00:00:11,228
        [A third subtitle in {language}]
        """

def get_prompt_later_segment(language):
    """Generate the prompt for later segments with the appropriate language instructions."""
    language_instruction = get_language_specific_instructions(language)
    
    return f"""
        Create SRT subtitles for the uploaded audio file in {language} language, {language_instruction.split(',')[1] if ',' in language_instruction else ''}.
        
        Ensure the following:
        - Format: SRT.
        - Timestamps: Accurate and in HH:mm:ss,mmm --> HH:mm:ss,mmm format.
        - Text:  Easily readable subtitles, split into meaningful parts, not too long or too short.
        - Empty lines: Add empty lines between each subtitle block (number, timestamp, text).
        - Start marker: Begin the subtitle output with the marker **START** on a new line.
        - Numbering: Continue the subtitle sequence numbering from where the provided subtitles end (do not repeat the last provided subtitle).  Do *not* restart numbering from 1.
        
        You are provided with a snippet of the *previous* subtitles to maintain context and ensure correct numbering.  **Generate subtitles for the *remaining audio* in the file, starting from where the provided subtitles finish.**
        
        Here is the snippet of previous subtitles to continue from:
        """

class SubtitleProcessor:
    def __init__(self, config):
        
        self.config = config
        
        # Create directories if they don't exist
        os.makedirs(self.config.input_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.temp_dir, exist_ok=True)
        
        # Initialize Gemini API
        self.client = genai.Client(api_key=self.config.api_key)

    def get_video_files(self):
        """
        Get all video files from input directory
        """
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
        return [f for f in os.listdir(self.config.input_dir)
                if os.path.splitext(f)[1].lower() in video_extensions]

    def convert_to_mp3(self, video_path):
        input_path = os.path.join(self.config.input_dir, video_path)
        output_path = os.path.join(self.config.temp_dir, os.path.splitext(video_path)[0] + '.mp3')
        
        try:
            # Extract audio using ffmpeg
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(stream, output_path, acodec='libmp3lame', ac=2, ar='44100')
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
            return output_path
        except ffmpeg.Error as e:
            print(f"Error extracting audio: {e.stderr.decode()}")
            raise

    def split_audio(self, mp3_path):
        """
        Split MP3 into segments with overlapping intervals.
        """
        audio = AudioSegment.from_mp3(mp3_path)
        duration = len(audio)
        segment_ms = self.config.segment_length * 1000
        overlap_ms = self.config.overlap * 1000
        segments = []
        
        start = 0
        while start < duration:
            end = min(start + segment_ms, duration)
            segment = audio[start:end]
            segment_path = f"{mp3_path[:-4]}_{start//1000}-{end//1000}.mp3"
            segment.export(segment_path, format="mp3")
            segments.append(segment_path)
            start = end - overlap_ms if end < duration else duration
            
        return segments
    
    def parse_subs(self, subtitles_data, segment_start):
        """
        Parse AI generated string subtitles into a list of dicts. 
        """
        try:
        
            if '**START**' in subtitles_data:
                subtitles_data = subtitles_data.split('**START**')[-1].strip()
            
            lines = subtitles_data.split('\n')
            
            number_line = False
            timestamp_line = False
            text_line = False
            number_line_str = None
            
            timing = None
            text = None
            
            subs = []
            
            for line in lines:
                
                if re.match('^\\d+$', line):
                    number_line = True
                    number_line_str = line
                    timestamp_line = False
                    text_line = False
                    
                elif number_line and re.match('^\\d+.*-->.*\\d+$', line):
                    number_line = False
                    timestamp_line = True
                    timing = line
                    
                elif timestamp_line and line:
                    number_line = False
                    timestamp_line = False
                    text_line = True
                    text = line
                
                if text_line:
                    rel_start, rel_end = self._parse_timestamp_line(timing)
                    
                    if not text.strip():
                        continue
                    
                    subs.append({
                        'id': number_line_str,
                        'start': rel_start + float(segment_start),
                        'end': rel_end + float(segment_start),
                        'text': text,
                        'timestamp': timing,
                    })
                    
                    number_line = False
                    timestamp_line = False
                    text_line = False
            
            return subs
        except Exception as e:
            traceback.print_exc()
            raise e
        
    def check_empty_subtitles(self, subs, segment_start, segment_end):
        """
        Raise Exception if there is interval without any subtitles,
        which might indicate issue with generating subtitles via LLM.
        
        Split subtitles into segments that are {empty_segment_check} seconds long,
        and raise Exception if any such segment is without subtitles.
        """
        
        if segment_end - segment_start < 180:
            return
        
        segment_times = {}
        time_sec = segment_start
        
        while time_sec < segment_end:
            segment_time = int(time_sec / self.config.empty_segment_check)
            
            # if less than 2 minutes are left to analyze, do not check empty subtitles
            if (segment_end - time_sec) < 120:
                break
            
            segment_times[segment_time] = False
            time_sec += self.config.empty_segment_check
        
        for sub in subs:
            segment_time = int(sub['start'] / self.config.empty_segment_check)
            segment_times[segment_time] = True
            
        if [s_t for s_t in [s for s in segment_times.keys()][1:] if not segment_times[s_t]]:
            raise Exception(f'Empty segment found {segment_times[s_t]} --> {segment_times[s_t] + self.config.empty_segment_check}')
        
    def check_subs_interval(self, subs, segment_start, segment_end):
        """
        Raise Exception if:
        - subtitles are outside segment start/end time
        - single subtitle text is too long and indicates a problem with timing
        """
        for sub in subs:
            
            sub_interval = sub['end'] - sub['start']
            
            if sub_interval > 45:
                raise Exception('Sub text too long')
            
            if sub['start'] < float(segment_start) or sub['end'] > float(segment_end) + 10:
                raise Exception('Subtitles are outside audio interval range')

    def check_cycling_subtitles(self, subs):
        """
        Raise Exception if same subtitle text is repeated too many times.
        Sometimes LLM generates same text over and over instead of generating
        correct subtitles. 
        """
        text_map = {}
        
        for sub in subs:
            
            if sub['text'] not in text_map:
                text_map[sub['text']] = 0
                
            text_map[sub['text']] += 1
            
        if [txt for txt in text_map.keys() if text_map[txt] >= 20]:
            raise Exception('Some sub text is repeating too many times')

    def check_text_length(self, subs):
        """
        Same as with cycling subtitles, sometimes LLM go crazy and make up
        some very long subtitle text.
        """
        for sub in subs:
            
            if len(sub['text']) >= 300:
                raise Exception('Subtitle text is too long')

    def create_subtitles(self, audio_path, subs=None, tries=10, segment_start=0, segment_end=None):
        """
        Get subtitles using Gemini API.
        """
        try:
            # upload the audio file directly using the path
            # media_file = genai.upload_file(audio_path)
            media_file = self.client.files.upload(file=audio_path)
            
            # generate subtitles with timestamps and speaker recognition
            if not subs:
            
                first_segment_prompt = get_prompt_first_segment(self.config.language)
                response = self.client.models.generate_content(
                    model=self.config.model.value,
                    contents=[first_segment_prompt, media_file]
                )
                
            else:
                
                # if we have subs from previous segment, supply the subs
                # so LLM can see the last generated subs and continue seamlessly from there
                
                last_subs = ''
                
                for sub in subs[:]:
                    last_subs += f'{sub["id"]}\n{sub["timestamp"]}\n{sub["text"]}\n\n'
                    
                print('continue sub sequence number ' + str(subs[0]['id']))
                
                later_segment_prompt = get_prompt_later_segment(self.config.language)
                response = self.client.models.generate_content(
                    model=self.config.model.value,
                    contents=[later_segment_prompt + last_subs, media_file]
                )
            
            if '**START**' not in response.text:
                # if LLM output did not follow our guidelines and did not start with
                # proper string sequence, then throw an error
                raise Exception('Subtitles missing')
                
            parsed_subs = self.parse_subs(response.text, segment_start)
            
            if segment_end - segment_start > 200 and segment_end - segment_start <= 240 and len(parsed_subs) <= 5:
                raise Exception('Insufficient subtitle density: Expected more than 5 subtitles for 200-240s segment')
            
            if segment_end - segment_start >= 300 and len(parsed_subs) <= 10:
                raise Exception('Insufficient subtitle density: Expected more than 10 subtitles for 300s segment')
            
            if segment_end - segment_start >= 400 and len(parsed_subs) <= 15:
                raise Exception('Insufficient subtitle density: Expected more than 15 subtitles for 400s segment')
            
            if segment_end - segment_start > 500 and len(parsed_subs) <= 20:
                raise Exception('Insufficient subtitle density: Expected more than 20 subtitles for 500s+ segment')
            
            if subs:
                
                # new subs should increment the last available sub id 
                if int(parsed_subs[0]['id']) < int(subs[0]['id']):
                    raise Exception('Wrong subtitle ids')
                
                numbering_correct = False
                timestamp_correct = True
                
                for sub in subs:
                    
                    if sub['id'] == parsed_subs[0]['id'] or str(int(sub['id']) + 1) == parsed_subs[0]['id']:
                        numbering_correct = True
                        
                    if sub['timestamp'] == parsed_subs[0]['timestamp']:
                        timestamp_correct = False
                        
                if not numbering_correct:
                    raise Exception('Subtitle sequence numbering is wrong')
                
                if not timestamp_correct:
                    raise Exception('Subtitle timestamps are incorrect, should start from 0')
            
            # generating subs via LLMs is not always consistent, especially when merging
            # segmented subs. perform checks if generated subtitles looks good
            self.check_subs_interval(parsed_subs, segment_start, segment_end)
            self.check_empty_subtitles(parsed_subs, segment_start, segment_end)
            self.check_cycling_subtitles(parsed_subs)
            self.check_text_length(parsed_subs)
            
            return parsed_subs
            
        except Exception as e:
            print(f"Error getting subtitles from Gemini: {str(e)}")
            
            tries -= 1
            
            if tries < 0:
                raise e
            
            time.sleep(15)
            return self.create_subtitles(audio_path, subs, tries, segment_start=segment_start, segment_end=segment_end)

    def create_srt_file(self, subtitles_data, output_path):
        """
        Create SRT file from subtitle data
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            
            if isinstance(subtitles_data, str):
                f.write(subtitles_data)
                
            else:
                for i, subtitle in enumerate(subtitles_data, 1):
                    start_time = self._seconds_to_timestamp(subtitle['start'])
                    end_time = self._seconds_to_timestamp(subtitle['end'])
                    speaker_tag = f"[{subtitle['speaker'].upper()}] " if 'speaker' in subtitle else ""
                
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{speaker_tag}{subtitle['text']}\n\n")
    
    @staticmethod
    def normalize_timestamp(timestamp: str) -> str:
        timestamp = timestamp.strip()
    
        # Replace :ddd with ,ddd using regex
        timestamp = re.sub(r':(\d{3})', r',\1', timestamp)
        
        # If only minutes and seconds (with milliseconds), add hours
        if len(timestamp.split(':')) == 2:
            timestamp = "00:" + timestamp
            
        return timestamp

    @staticmethod
    def _timestamp_to_seconds(timestamp):
        """Convert SRT timestamp to seconds"""
        try:
            timestamp = SubtitleProcessor.normalize_timestamp(timestamp)
            
            h, m, s = timestamp.split(':')
            
            if '.' in s:
                s, ms = s.split('.')
            elif ',' in s:
                s, ms = s.split(',')
            else:
                s = s
                ms = '000'
                
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
        except Exception as e:
            raise e

    @staticmethod
    def _seconds_to_timestamp(seconds):
        """Convert seconds to SRT timestamp format"""
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        seconds = td.seconds % 60
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    @staticmethod
    def _parse_timestamp_line(timing_line):
        """Parse SRT timestamp line to start and end seconds"""
        try:
            start_ts, end_ts = timing_line.split(' --> ')
            return (SubtitleProcessor._timestamp_to_seconds(start_ts), 
                    SubtitleProcessor._timestamp_to_seconds(end_ts))
        except Exception as e:
            raise e
    
    @staticmethod    
    def _format_srt_timestamp(start_seconds, end_seconds):
        def seconds_to_timestamp(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = seconds % 60
            milliseconds = int((seconds % 1) * 1000)
            seconds = int(seconds)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
           
        return f"{seconds_to_timestamp(start_seconds)} --> {seconds_to_timestamp(end_seconds)}"
        
    def get_segment_subs(self, subs, segment_start, segment_end):
        """
        Return subs from specific segment, bounded with segment_start and segment_end.
        """
        segment_subs = []
        
        for sub in subs:
            
            if segment_start <= sub['start'] <= segment_end:
                
                sub_copy = sub.copy()
                sub_copy['start'] = sub_copy['start'] - segment_start
                sub_copy['end'] = sub_copy['end'] - segment_start
                sub_copy['timestamp'] = SubtitleProcessor._format_srt_timestamp(sub_copy['start'], sub_copy['end'])
                segment_subs.append(sub_copy)
                
        return segment_subs
    
    def merge_subs(self, all_subs, subs):
        """
        Merge previous segment subs with current segment subs.
        """
        # remove overlapping subs from previous segments and current segment
        if all_subs:
            all_subs = all_subs[:int(subs[0]['id'])-1]
        
        # add subs from current segment
        for sub in subs:
            
            # skip already included sub based on sub id
            if all_subs and int(sub['id']) <= int(all_subs[-1]['id']):
                continue
            
            # fix times, if last sub ends later than current sub starts
            if all_subs and all_subs[-1]['end'] > sub['start']:
                all_subs[-1]['end'] = sub['start']
            
            all_subs.append(sub)
            
        return all_subs

    def process_video(self, video_file):
        print(f"Processing {video_file}...")
        
        try:
            # create subtitles for audio file instead of video,
            # so we can process longer segments and more efficiently
            mp3_path = self.convert_to_mp3(video_file)
            
            # split audio file into smaller chunks, because
            # longer segments would need bigger input/output context lengths
            audio_segments = self.split_audio(mp3_path)
            
            # merge all created subs here
            all_subs = []
            
            for audio_segment in audio_segments:
                
                print('processing segment ' + audio_segment)
                
                segment_start = int(audio_segment.split('_')[-1].split('-')[0])
                segment_end = int(audio_segment.split('_')[-1].split('-')[1].split('.')[0])
                
                if not all_subs:
                    # if processing first segment..
                    subs = self.create_subtitles(audio_segment, segment_start=segment_start, segment_end=segment_end)
                else:
                    # if we already have subs from one or more previous segments,
                    # supply overlapping subs for current segment process
                    segment_subs = self.get_segment_subs(all_subs, segment_start, segment_end)
                    subs = self.create_subtitles(audio_segment, segment_subs, segment_start=segment_start, segment_end=segment_end)                    
                
                all_subs = self.merge_subs(all_subs, subs)
            
            # store created subtitles
            final_srt = os.path.join(self.config.output_dir, 
                                   os.path.splitext(video_file)[0] + '.srt')
            self.create_srt_file(all_subs, final_srt)
            
            # move video file to output directory
            shutil.move(os.path.join(self.config.input_dir, video_file),
                       os.path.join(self.config.output_dir, video_file))
            
            # clean up temp MP3 files
            os.remove(mp3_path)
            
            for audio_segment in audio_segments:
                os.remove(audio_segment)
            
            print(f"Finished processing {video_file}")
            
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")

    def process_all_videos(self):
        """
        Get all video files from input folder and process them.
        
        If subtitles are successfully created, video with subtitles will be
        moved into output folder.
        """
        videos = self.get_video_files()
        
        if not videos:
            print("No video files found in input directory")
            return
            
        for video in videos:
            self.process_video(video)
            
def parse_arguments():
    """Parse command line arguments with defaults displayed from Config"""
    # Create a default config to get default values
    default_config = Config()
    
    parser = argparse.ArgumentParser(
        description="Generate subtitles for videos using AI"
    )
    parser.add_argument(
        "--input-dir", 
        help=f"Directory containing input videos (default: {default_config.input_dir})"
    )
    parser.add_argument(
        "--output-dir", 
        help=f"Directory for processed files (default: {default_config.output_dir})"
    )
    parser.add_argument(
        "--segment-length", 
        type=int, 
        help=f"Length of each segment in seconds (default: {default_config.segment_length})"
    )
    parser.add_argument(
        "--overlap", 
        type=int, 
        help=f"Overlap between segments in seconds (default: {default_config.overlap})"
    )
    parser.add_argument(
        "--language", 
        help=f"Target language for subtitles (default: {default_config.language})"
    )
    parser.add_argument(
        "--api-key", 
        help="Gemini API key (overrides environment variable)"
    )
    parser.add_argument(
        "--model", 
        help=f"Gemini model to use (default: {default_config.model})"
    )
    parser.add_argument(
        "--single-file", 
        help="Process a single video file instead of a directory"
    )
    parser.add_argument(
        "--empty-segment-check",
        type=int,
        help=f"Max seconds without any subtitles (default: {default_config.empty_segment_check})"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments and create config
    args = parse_arguments()
    config = Config.from_args(args)
    
    # Change language or any other parameter here
    config.language = 'slovak'
    
    # Create and run the processor with the config
    processor = SubtitleProcessor(config)
    
    if config.single_file:
        processor.process_video(config.single_file)
    else:
        processor.process_all_videos()
    