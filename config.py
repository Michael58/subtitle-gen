import enum
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
import os

class ModelOption(enum.Enum):
    GEMINI_FLASH = "gemini-2.0-flash-thinking-exp"
    GEMINI_PRO = "gemini-pro"
    GEMINI_25_FLASH = "gemini-2.5-flash-preview-04-17"
    # Add more models as needed


@dataclass
class Config:
    
    def __init__(
        self,
        input_dir: str = "no_subtitles",    # Directory containing input videos to process
        output_dir: str = "generated_subtitles",  # Directory where processed videos and subtitles will be saved
        segment_length: int = 600,          # Length of each audio segment in seconds (10 minutes)
        overlap: int = 120,                 # Overlap between segments in seconds (2 minutes)
        language: str = 'slovak',           # Target language for subtitles
        api_key: Optional[str] = None,      # Gemini API key (if not set, loaded from environment)
        model: ModelOption = ModelOption.GEMINI_25_FLASH,  # Gemini model to use for subtitle generation
        single_file: Optional[str] = None,  # Process a single video file instead of all files in input_dir
        empty_segment_check: int = 120,     # Maximum seconds without subtitles before raising an error
        temp_dir: str = "temp"              # Directory where to keep temporary files           
    ):
        # Set all configuration parameters
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.segment_length = segment_length
        self.overlap = overlap
        self.language = language
        self.model = model
        self.single_file = single_file
        self.empty_segment_check = empty_segment_check
        self.temp_dir = temp_dir
        
        # Handle API key - first check the provided key, then environment
        self.api_key = api_key
        if self.api_key is None:
            # Load environment variables
            load_dotenv()
            self.api_key = os.environ.get('GEMINI_KEY')
            
            if not self.api_key:
                raise Exception("GEMINI_KEY not found in environment variables")

    @classmethod
    def from_args(cls, args):
        """Create a Config instance from parsed command line arguments"""
        config = cls()
        
        # Update config with command line arguments if they're provided
        if args.input_dir is not None:
            config.input_dir = args.input_dir
            
        if args.output_dir is not None:
            config.output_dir = args.output_dir
            
        if args.segment_length is not None:
            config.segment_length = args.segment_length
            
        if args.overlap is not None:
            config.overlap = args.overlap
            
        if args.language is not None:
            try:
                config.language = args.language
            except ValueError:
                print(f"Warning: Unknown language '{args.language}'. Using default: {config.language.value}")
                
        if args.api_key is not None:
            config.api_key = args.api_key
            
        if args.model is not None:
            try:
                config.model = ModelOption(args.model)
            except ValueError:
                print(f"Warning: Unknown model '{args.model}'. Using default: {config.model.value}")
                
        if args.single_file is not None:
            config.single_file = args.single_file
            
        if args.empty_segment_check is not None:
            config.empty_segment_check = int(args.empty_segment_check)
            
        return config
    