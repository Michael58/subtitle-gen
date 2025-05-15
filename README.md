# AI Video Subtitle Generator

A Python tool that automatically generates subtitles for video files using Google's Gemini AI. The tool extracts audio from videos, processes it in manageable chunks, and creates accurate SRT subtitles with proper timing.

> **Note:** This project was created to generate subtitles for a favorite TV show. With extensive testing and refinement, it now achieves approximately 90-95% accuracy in translation and timing. The current default settings for segment length and overlap have been extensively tested and provide the best results with current models.

## Features

- Supports multiple languages with proper grammatical handling
- Processes videos in segments to handle files of any length
- Automatically continues subtitle numbering across segments
- Includes several validation checks to ensure quality subtitles
- Configurable via command line arguments or environment variables

## Requirements

- Python 3.7+
- FFmpeg installed and available in PATH
- Google Gemini API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ai-subtitle-generator.git
   cd ai-subtitle-generator
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Gemini API key:
   ```
   export GEMINI_KEY=your_api_key_here
   ```
   Or create a `.env` file with `GEMINI_KEY=your_api_key_here`

## Usage

### Basic Usage

Place video files in the `no_subtitles` directory and run:

```
python subtitle_generator.py
```

Generated SRT files and processed videos will be saved to the `generated_subtitles` directory.

### Setting the Language

To generate subtitles in a specific language:

```
python subtitle_generator.py --language english
```

### Processing a Single File

```
python subtitle_generator.py --single-file my_video.mp4
```

### Advanced Options

```
python subtitle_generator.py --segment-length 300 --overlap 60 --model gemini-2.5-flash-preview-04-17
```

> **Important:** The default segment length (600 seconds) and overlap (120 seconds) values have been carefully optimized through extensive testing. Changing these values may result in poorer subtitle quality. As models improve, longer segment lengths may become viable.

## Command Line Options

- `--input-dir`: Directory containing input videos (default: "no_subtitles")
- `--output-dir`: Directory for processed files (default: "generated_subtitles") 
- `--segment-length`: Length of each segment in seconds (default: 600)
- `--overlap`: Overlap between segments in seconds (default: 120)
- `--language`: Target language for subtitles (default: "slovak")
- `--api-key`: Gemini API key (overrides environment variable)
- `--model`: Gemini model to use (default: gemini-2.5-flash-preview-04-17)
- `--single-file`: Process a single video file instead of a directory
- `--empty-segment-check`: Max seconds without any subtitles (default: 120)

## How It Works

1. Video files are converted to MP3 format using FFmpeg
2. Audio is split into overlapping segments for efficient processing
3. Each segment is sent to the Gemini API with appropriate prompts
4. Subtitles are processed and merged into a single SRT file
5. Multiple validation checks ensure subtitle quality and consistency

## Customization and Improvement

The subtitle generation quality can be enhanced by modifying the prompts in the `get_prompt_first_segment()` and `get_prompt_later_segment()` functions. You can adjust these to:

- Change the tone of subtitles
- Improve language-specific translations
- Add personality to character dialogues
- Enhance narrative flow

As LLM models continue to improve, this tool will likely produce even better results with less need for segmentation and validation checks.
