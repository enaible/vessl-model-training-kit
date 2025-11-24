"""Main generator for IFEval drill dataset creation."""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import argparse
import aiohttp
import yaml
import os
import re
from utils import resolve_path
from openai import OpenAI
from tqdm.asyncio import tqdm

@dataclass
class Config:
    """Configuration for IFEval drill generation."""
    # Model settings
    model_name: str
    model_path: str
    api_client: str
    api_key: str
    
    # Generation parameters
    temperature: float
    max_tokens: int
    
    # Data paths
    source_data_path: str
    output_dir: str
    
    # Dataset configuration
    data_size: int
    categories: List[str]
    prompt_path: str
    
    # Processing settings
    chunk_size: int
    batch_size: int
    max_retries: int
    retry_delay: int
    
    # Quality control
    quality_mode: bool
    validation_strict: bool
    min_turn_length: int
    min_reference_length: int
    quality_keywords_required: int
    
    # Cache settings
    enable_cache: bool
    cache_dir: str
    cache_cleanup_days: int
    cache_stats_interval: int
    
    # Batch generation settings
    enable_batch_mode: bool
    max_batch_size: int
    min_batch_size: int
    batch_fallback_single: bool
    
    # Logging
    log_level: str
    log_file: str
    save_frequency: int


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file with environment variable substitution."""
    config_path = resolve_path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace environment variables
    pattern = r'\$\{([^}]+)\}'
    
    def replace_env_var(match):
        env_expr = match.group(1)
        if ':-' in env_expr:
            var_name, default_value = env_expr.split(':-', 1)
            return os.getenv(var_name, default_value)
        else:
            return os.getenv(env_expr, '')
    
    content = re.sub(pattern, replace_env_var, content)
    config_dict = yaml.safe_load(content)
    
    return Config(**config_dict)


def load_existing_data(output_path: str) -> List[Dict[str, Any]]:
    """Load existing data from JSONL file."""
    if not os.path.exists(output_path):
        return []
    
    data = []
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        logging.error(f"Error loading existing data: {e}")
        return []
    
    return data


def save_data_item(output_path: str, item: Dict[str, Any]) -> None:
    """Save a single data item to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


def create_output_path(output_dir: str, filename: str) -> str:
    """Create output path with directory creation."""
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def create_backup_if_exists(file_path: str) -> None:
    """Create a backup of the file if it exists."""
    if os.path.exists(file_path):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.backup_{timestamp}"
        
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"💾 Backup created: {backup_path}")

class DrillGenerator:
    """Main drill generation class."""
    
    def __init__(self, config: Config, dataset: str, language: str):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.api_client = config.api_client
        self.dataset = dataset
        self.language = language
        
        # Load instruction mapping (instruction_id -> class_name)
        mapping_path = Path(__file__).parent.parent / "utils" / "instruction_mapping.json"
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            self.instruction_id_to_class = mapping_data["instruction_id_to_class"]
        
        # Load task descriptions (class_name -> description)
        task_dict_path = Path(__file__).parent.parent / "utils" / "task_dict.json"
        with open(task_dict_path, 'r', encoding='utf-8') as f:
            self.task_dict = json.load(f)
        
        # Load prompt template
        prompt_path = resolve_path(config.prompt_path)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
        
        # OpenAI client setup
        self.client = None
        self.url = None
        self.api_key = None
        self.headers = None
        
        if self.api_client == "openai":
            self.url = 'https://api.openai.com/v1/chat/completions'
            self.api_key = os.environ.get("OPENAI_API_KEY")
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self.client = None
        elif self.api_client == "azure":
            self.url = "https://kt-aipt-2025-useast1-resource.cognitiveservices.azure.com/openai/v1/chat/completions"
            endpoint = self.url
            self.headers = {
                "Authorization": os.environ.get("AZURE_OPENAI_API_KEY"),
                "Content-Type": "application/json",
            }
            self.client = OpenAI(base_url=f"{endpoint}", api_key=os.environ.get("AZURE_OPENAI_API_KEY"))
        elif self.api_client == "vessl":
            self.url = "http://211.188.81.250:32759/v1/chat/completions"
            self.headers = {
                "Content-Type": "application/json"
            }
            self.client = None
        
        self.semaphore = asyncio.Semaphore(config.batch_size)
        
        self.logger.info(f"🎯 Generator initialized for {dataset} in {language}")
    
    def get_instruction_class_name(self, instruction_id: str) -> Optional[str]:
        """Get the class name from instruction_id using the mapping."""
        return self.instruction_id_to_class.get(instruction_id)
    
    def build_instructions(self, instruction_id_list: List[str]) -> str:
        """Build instruction text from instruction_id_list."""
        instructions = []
        
        for instruction_id in instruction_id_list:
            class_name = self.get_instruction_class_name(instruction_id)
            if class_name and class_name in self.task_dict:
                description = self.task_dict[class_name]
                instructions.append(description)
            else:
                self.logger.warning(f"⚠️  Unknown instruction: {instruction_id} -> {class_name}")
        
        return "\n".join(instructions)
    
    def build_prompt(self, source_item: Dict[str, Any]) -> str:
        """Build the full prompt from source_item."""
        # Extract instruction IDs and build criteria
        instruction_id_list = source_item.get("instruction_id_list", [])
        instructions = self.build_instructions(instruction_id_list)
        
        # Get the question/prompt
        prompt = source_item.get("prompt", "")
        
        # Render the template
        full_prompt = self.prompt_template.format(
            instructions=instructions,
            prompt=prompt
        )
        
        return full_prompt
    
    async def generate_single_drill(self, source_item: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """Generate a single drill from source item."""
        async with self.semaphore:
            try:
                # Build prompt
                full_prompt = self.build_prompt(source_item)
                
                # Prepare API request
                messages = [{"role": "user", "content": full_prompt}]
                
                # Call API
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "messages": messages,
                        "temperature": getattr(self.config, 'temperature', 0.7),
                        "max_tokens": getattr(self.config, 'max_tokens', 2048),
                    }
                    
                    async with session.post(self.url, headers=self.headers, json=payload) as response:
                        if response.status != 200:
                            self.logger.error(f"❌ API error {response.status}: {await response.text()}")
                            return None
                        
                        result = await response.json()
                        
                        # Extract response
                        if 'choices' in result and len(result['choices']) > 0:
                            generated_text = result['choices'][0]['message']['content']
                        else:
                            self.logger.error(f"❌ No response in API result")
                            return None
                
                # Try to extract JSON from the response
                try:
                    # First try direct parsing
                    response_json = json.loads(generated_text)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"⚠️  JSON parse error: {e}")
                    # Try to extract JSON from <JSON_OUTPUT> tags and use json.loads with strict=False
                    json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        try:
                            # Try parsing with more lenient settings
                            response_json = json.loads(json_str, strict=False)
                        except json.JSONDecodeError:
                            # Last resort: try to manually extract question and response
                            try:
                                # Find the content between quotes after "question": and before the next comma followed by "response"
                                question_start = json_str.find('"question"')
                                if question_start == -1:
                                    self.logger.error(f"❌ Could not find 'question' field")
                                    return None
                                
                                # Find the opening quote after "question":
                                q_start = json_str.find('"', json_str.find(':', question_start) + 1)
                                # Find the closing quote before ",\n    "response"
                                response_marker = json_str.find('"response"', q_start)
                                if response_marker == -1:
                                    self.logger.error(f"❌ Could not find 'response' field")
                                    return None
                                
                                # Work backwards from "response" to find the last quote
                                q_end = json_str.rfind('"', q_start + 1, response_marker)
                                question = json_str[q_start + 1:q_end]
                                
                                # Now get response
                                r_start = json_str.find('"', json_str.find(':', response_marker) + 1)
                                r_end = json_str.rfind('"')
                                response = json_str[r_start + 1:r_end]
                                
                                response_json = {
                                    "question": question,
                                    "response": response
                                }
                                self.logger.info(f"✅ Extracted JSON manually")
                            except Exception as manual_error:
                                self.logger.error(f"❌ Manual extraction failed: {manual_error}")
                                return None
                    else:
                        self.logger.error(f"❌ Could not find JSON structure in response")
                        return None
                
                # Safety check
                if not response_json or not isinstance(response_json, dict):
                    self.logger.error(f"❌ Invalid response_json: {type(response_json)}")
                    return None
                
                # Validate extracted data
                question = response_json.get('question', '').strip()
                reference = response_json.get('response', '').strip()
                
                if not question or not reference:
                    self.logger.error(f"❌ Invalid drill: empty question or response")
                    self.logger.debug(f"   Question: {bool(question)}, Response: {bool(reference)}")
                    return None
                
                # Create drill item
                drill_item = {
                    "source_key": source_item.get("key"),
                    "question": question,
                    "reference": reference,
                }
                
                return drill_item
                
            except Exception as e:
                self.logger.error(f"❌ Error generating drill {index}: {e}")
                return None
    
    async def generate_all_drills(self, output_path: str) -> None:
        """Generate all required drills by iterating through source data."""
        
        # Load source data
        source_data_path = resolve_path(self.config.source_data_path)
        self.logger.info(f"📂 Loading source data from {self.config.source_data_path}")
        with open(source_data_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        
        self.logger.info(f"📊 Source data loaded: {len(source_data)} items")
        
        # Check existing data
        existing_data = load_existing_data(output_path)
        existing_count = len(existing_data)
        
        self.logger.info(f"📂 Existing data: {existing_count} items")
        
        # Calculate remaining needed
        total_remaining = self.config.data_size - existing_count
        
        if total_remaining <= 0:
            self.logger.info("✅ Target data size already met!")
            return
        
        self.logger.info(f"🎯 Target: {self.config.data_size} | Remaining: {total_remaining}")
        
        # Create progress bar
        pbar = tqdm(
            total=total_remaining,
            desc="🎯 Generating Drills",
            unit="drills",
            initial=0
        )
        
        # Generate drills by cycling through source data
        generated_count = 0
        source_index = 0
        
        # Create tasks for concurrent generation
        tasks = []
        batch_size = self.config.batch_size
        
        while generated_count < total_remaining:
            # Create batch of tasks
            current_batch_tasks = []
            for _ in range(min(batch_size, total_remaining - generated_count)):
                source_item = source_data[source_index % len(source_data)]
                task_index = existing_count + generated_count
                
                task = self.generate_single_drill(source_item, task_index)
                current_batch_tasks.append(task)
                
                source_index += 1
            
            # Execute batch
            batch_results = await asyncio.gather(*current_batch_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"❌ Task failed: {result}")
                    continue
                
                if result is not None:
                    # Save drill item
                    save_data_item(output_path, result)
                    generated_count += 1
                    pbar.update(1)
        
        pbar.close()
        
        # Print final report
        self.logger.info("=" * 80)
        self.logger.info("🎉 Generation Complete!")
        self.logger.info(f"📊 Total generated: {generated_count}")
        self.logger.info(f"💾 Saved to: {output_path}")
        self.logger.info("=" * 80)

async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate IFEval drill dataset")
    parser.add_argument("--dataset", default="ifeval", help="Dataset name")
    parser.add_argument("--language", default="Thai", help="Language for generation")
    parser.add_argument("--backup", action="store_true", help="Create backup of existing data before starting")
    parser.add_argument("--force-new", action="store_true", help="Start fresh generation (ignores existing data)")
    parser.add_argument("--batch-size", type=int, help="Override batch size for concurrent requests")
    args = parser.parse_args()
    
    dataset = args.dataset.lower()
    if dataset not in ["ifeval"]:
        print(f"❌ Invalid dataset: {dataset}. Choose 'ifeval'")
        return

    # Load configuration
    config = load_config(f"config/{dataset}/setting.yaml")
    
    # Setup logging
    logger = setup_logging(config.log_level)
    
    print("=" * 80)
    print("🚀 IFEval Drill Generator")
    print("=" * 80)
    print(f"📋 Dataset: {dataset}")
    print(f"🌍 Language: {args.language}")
    print(f"📊 Target size: {config.data_size}")
    print(f"🔧 Model: {config.model_name}")
    print(f"🔗 API: {config.api_client}")
    print("-" * 80)
    
    # Override batch size if specified
    if args.batch_size:
        config.batch_size = args.batch_size
        print(f"⚙️  Override batch size: {args.batch_size}")
    
    # Create output path (fixed filename)
    output_dir = resolve_path(config.output_dir)
    output_path = create_output_path(output_dir, "drill_data.jsonl")
    
    # Handle existing data and backup options
    existing_data = load_existing_data(output_path)
    if existing_data and not args.force_new:
        print(f"📂 Found existing data: {len(existing_data)} items in {output_path}")
        
        if args.backup:
            create_backup_if_exists(output_path)
            print("💾 Backup created - continuing generation")
        
        print("🔄 Generation will resume from where it left off")
        
    elif existing_data and args.force_new:
        print(f"🆕 Force new generation - ignoring {len(existing_data)} existing items")
        create_backup_if_exists(output_path)
        # Clear the file
        with open(output_path, 'w') as f:
            pass
        
    else:
        print(f"🆕 Starting new generation to {output_path}")
    
    print("=" * 80)
    
    # Initialize generator
    generator = DrillGenerator(config, dataset, args.language)
    
    # Run generation
    await generator.generate_all_drills(output_path)
    
    print("\n✅ All done!")

if __name__ == "__main__":
    asyncio.run(main())