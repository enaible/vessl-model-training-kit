"""Main generator for MTBench drill dataset creation."""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import argparse

from openai import AzureOpenAI, OpenAI
from tqdm.asyncio import tqdm

from utils import (
    Config, load_config, setup_logging, create_question_id,
    load_existing_data, save_data_item, get_category_counts,
    calculate_remaining_per_category, create_output_path,
    create_backup_if_exists

)
from prompt_builder import PromptBuilder
from chunker import TokenChunker
from postprocess import DataValidator
from token_tracker import TokenTracker
from prompt_cache import PromptCache
from batch_generator import BatchDrillGenerator, BatchRequest
import os

class DrillGenerator:
    """Main drill generation class."""
    
    def __init__(self, config: Config, dataset: str, difficulty: str, language: str):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.api_client = config.api_client
        self.dataset = dataset
        self.difficulty = difficulty
        self.language = language
        # OpenAI client setup
        self.client = None  # Initialize to None first
        self.url = None
        self.api_key = None
        self.headers = None
        if self.api_client == "openai":
            self.url = 'https://api.openai.com/v1/chat/completions'
            self.api_key = os.environ.get("OPENAI_API_KEY")
            self.headers = headers = {
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
        
        if difficulty == "easy":
            self.prompt_builder = PromptBuilder(config.category_easy_prompt_map)
        elif difficulty == "hard":
            self.prompt_builder = PromptBuilder(config.category_hard_prompt_map)
        self.chunker = TokenChunker(config.model_name, config.chunk_size)
        self.validator = DataValidator()
        self.token_tracker = TokenTracker(config.model_name)
        self.semaphore = asyncio.Semaphore(config.batch_size)
        
        # Initialize cache if enabled
        if config.enable_cache:
            self.cache = PromptCache(config.cache_dir)
            self.cache_enabled = True
            self.logger.info(f"💾 Prompt cache enabled: {config.cache_dir}")
        else:
            self.cache = None
            self.cache_enabled = False
            self.logger.info("💾 Prompt cache disabled")
        
        # Initialize batch generator if enabled
        if config.enable_batch_mode:
            self.batch_generator = BatchDrillGenerator(
                self.dataset,
                self.language,
                self.prompt_builder,
                self.config, 
                self.token_tracker,
                self.api_client, 
                self.client,
                self.url,
                self.headers,
                self.cache if self.cache_enabled else None
            )
            self.batch_mode_enabled = True
            self.logger.info(f"🔄 Batch generation enabled: {config.min_batch_size}-{config.max_batch_size} problems per batch")
        else:
            self.batch_generator = None
            self.batch_mode_enabled = False
            self.logger.info("🔄 Batch generation disabled - using single mode")
    
    async def generate_batch_drills(
        self, 
        category: str, 
        source_items: List[Dict[str, Any]], 
        start_index: int,
        count: int,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate multiple drill problems using batch mode only (no single fallback)."""
        if not self.batch_mode_enabled or not self.batch_generator:
            self.logger.warning("Batch mode not enabled, falling back to single mode")
            return []

        # Determine batch size
        if batch_size is None:
            batch_size = self.batch_generator.calculate_batch_size(count)
        min_batch_size = getattr(self.config, 'min_batch_size', 2)

        all_results = []
        processed = 0

        while processed < count:
            remaining = count - processed
            current_batch_size = min(batch_size, remaining)

            batch_request = BatchRequest(
                category=category,
                source_items=source_items[processed:processed + current_batch_size],
                start_index=start_index + processed,
                count=current_batch_size,
                language=self.language
            )

            batch_results, failed_count = await self.batch_generator.generate_batch(batch_request)
            all_results.extend(batch_results)
            processed += current_batch_size

            # 만약 실패가 있으면 batch 크기를 줄여서 재귀적으로 시도
            if failed_count > 0:
                if current_batch_size > min_batch_size:
                    new_batch_size = max(current_batch_size // 2, min_batch_size)
                    self.logger.info(f"Batch failed, retrying with smaller batch size: {new_batch_size}")
                    failed_items = source_items[processed - current_batch_size + len(batch_results):processed]
                    all_results.extend(
                        await self.generate_batch_drills(
                            category,
                            failed_items,
                            start_index + processed - current_batch_size + len(batch_results),
                            len(failed_items),
                            batch_size=new_batch_size
                        )
                    )
                else:
                    self.logger.error(f"Batch failed at min batch size ({min_batch_size}), giving up on these items.")
            # 싱글 fallback 완전 제거(주석 처리)
            # if failed_count > 0 and self.config.batch_fallback_single:
            #     self.logger.info(f"Falling back to single mode for {failed_count} failed items")
            #     failed_start_idx = start_index + processed - current_batch_size + len(batch_results)
            #     failed_items = source_items[processed - current_batch_size + len(batch_results):processed]
            #     for i, item in enumerate(failed_items):
            #         drill = await self.generate_single_drill(item, category, failed_start_idx + i)
            #         if drill:
            #             all_results.append(drill)
        return all_results
    
    async def generate_category_batch(
        self, 
        category: str, 
        source_items: List[Dict[str, Any]], 
        start_index: int,
        output_path: str
    ) -> List[Dict[str, Any]]:
        """Generate batch of drills for a category using intelligent batching."""
        
        count = len(source_items)
        
        # Use the new batch generation capability
        results = await self.generate_batch_drills(category, source_items, start_index, count)
        
        # Save all results immediately
        for result in results:
            save_data_item(output_path, result)
        
        # Log final statistics
        success_rate = len(results) / count * 100 if count > 0 else 0
        mode = "batch" if (self.batch_mode_enabled and self.batch_generator and 
                          self.batch_generator.should_use_batch_mode(count)) else "single"
        
        self.logger.info(f"✅ {category} completed: {len(results)}/{count} drills ({success_rate:.1f}% success, {mode} mode)")
        
        return results
    
    async def generate_all_drills(self, output_path: str) -> None:
        """Generate all required drills."""
        
        # Load source data
        with open(self.config.source_data_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        
        # Check existing data
        existing_data = load_existing_data(output_path)
        existing_counts = get_category_counts(existing_data)
        
        self.logger.info(f"Existing data counts: {existing_counts}")
        
        # Calculate remaining needed per category
        remaining_per_category = calculate_remaining_per_category(
            existing_counts, self.config.data_size, self.config.categories
        )
        
        self.logger.info(f"Remaining needed per category: {remaining_per_category}")
        
        total_remaining = sum(remaining_per_category.values())
        if total_remaining == 0:
            self.logger.info("All required data already generated!")
            return
        
        # Create category chunks
        category_chunks = self.chunker.chunk_by_category(source_data, remaining_per_category)
        
        # Track progress
        total_generated = 0
        category_indices = {cat: existing_counts.get(cat, 0) for cat in self.config.categories}
        
        # Create overall progress bar
        overall_pbar = tqdm(
            total=total_remaining,
            desc="🎯 Overall Progress",
            unit="drills",
            position=0
        )
        
        # Create semaphore for concurrent chunk processing
        chunk_semaphore = asyncio.Semaphore(self.config.batch_size)  # Use config limit
        
        async def process_chunk(category: str, chunk: List[Dict[str, Any]], chunk_idx: int, 
                               start_index: int, output_path: str, overall_pbar) -> Tuple[str, List[Dict[str, Any]], int]:
            """Process a single chunk asynchronously."""
            async with chunk_semaphore:
                self.logger.info(f"Processing {category} chunk {chunk_idx + 1}")
                
                # Generate batch
                batch_results = await self.generate_category_batch(
                    category, chunk, start_index, output_path
                )
                
                result_count = len(batch_results)
                
                # Update progress bar immediately when this chunk completes
                overall_pbar.update(result_count)
                overall_pbar.set_postfix({
                    'Category': category,
                    'Cost': f'${self.token_tracker.total_usage.get_estimated_cost(self.config.model_name):.4f}'
                })
                
                return category, batch_results, result_count
        
        # Process each category
        for category, chunks in category_chunks.items():
            category_generated = 0
            
            self.logger.info(f"🔄 Processing {category}: {len(chunks)} chunks")
            
            # Create tasks for all chunks in this category
            chunk_tasks = []
            for chunk_idx, chunk in enumerate(chunks):
                start_index = category_indices[category] + category_generated
                task = process_chunk(category, chunk, chunk_idx, start_index, output_path, overall_pbar)
                chunk_tasks.append(task)
            
            # Process chunks concurrently and collect results
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            # Process results in order
            for i, result in enumerate(chunk_results):
                if isinstance(result, BaseException):
                    self.logger.error(f"❌ Chunk {i + 1} failed: {result}")
                    continue
                
                result_category, batch_results, result_count = result
                category_generated += result_count
                total_generated += result_count
                
                # Log token usage progress
                self.token_tracker.log_progress(total_generated, total_remaining)
                
                # Stop if we've generated enough for this category
                if category_generated >= remaining_per_category[category]:
                    break
        
        overall_pbar.close()
        
        # Print final report
        self.token_tracker.print_final_report()
        
        # Print cache statistics
        if self.cache_enabled:
            self.cache.print_stats()
            self.cache.save_stats()
        
        self.logger.info(f"🎉 Generation complete! Total generated: {total_generated}")

async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description=f"Generate drill dataset")
    parser.add_argument("--dataset", default="gsm8k", help="Dataset name")
    parser.add_argument("--subject", default="default", help="Subject name")
    parser.add_argument("--difficulty", default = "hard", help="Difficulty level")
    parser.add_argument("--backup", action="store_true", help="Create backup of existing data before starting")
    parser.add_argument("--force-new", action="store_true", help="Start fresh generation (ignores existing data)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear prompt cache before starting")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics and exit")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache for this run")
    parser.add_argument("--no-batch", action="store_true", help="Disable batch mode for this run")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    args = parser.parse_args()
    
    dataset = args.dataset.lower().replace(" ", "_")
    difficulty = args.difficulty.lower()
    subject = args.subject.lower().replace(" ", "_")
    if dataset not in ["arc", "gsm8k", "hellaswag", "mmlu", "truthfulqa", "winogrande"]:
        print("Invalid dataset. Choose between arc, gsm8k, hellaswag, mmlu, truthfulqa, and winogrande.", dataset)
        return

    # Load configuration
    config = load_config(f"config/{dataset}/default.yaml")
    
    difficulty = args.difficulty.lower()
    if difficulty not in ["easy", "hard"]:
        print("Invalid difficulty level. Choose between easy and hard.", difficulty)
        return

    print("config: ", config)
    print("-"*100)
    
    # Override cache setting if --no-cache is specified
    if args.no_cache:
        config.enable_cache = False
    
    # Override batch mode setting if --no-batch is specified
    if args.no_batch:
        config.enable_batch_mode = False
    
    # Override batch size if specified
    if args.batch_size:
        config.max_batch_size = args.batch_size
    
    # Handle cache-only operations
    if args.cache_stats:
        if config.enable_cache:
            cache = PromptCache(config.cache_dir)
            cache.print_stats()
        else:
            print("💾 Cache is disabled")
        return
    
    if args.clear_cache:
        if config.enable_cache:
            cache = PromptCache(config.cache_dir)
            cache.clear_cache()
            print("🧹 Cache cleared successfully")
        else:
            print("💾 Cache is disabled - nothing to clear")
        return
    
    # Create output path (fixed filename)
    output_path = create_output_path(config.output_dir, f"drill_data_{subject}.jsonl")
    
    # Handle existing data and backup options
    existing_data = load_existing_data(output_path)
    if existing_data and not args.force_new:
        print(f"📂 Found existing data: {len(existing_data)} items in {output_path}")
        
        if args.backup:
            create_backup_if_exists(output_path)
            print("💾 Backup created - continuing generation")
        
        print("🔄 Generation will resume from where it left off")
        existing_counts = get_category_counts(existing_data)
        print(f"📊 Current progress: {existing_counts}")
        
    elif existing_data and args.force_new:
        print(f"🆕 Force new generation - ignoring {len(existing_data)} existing items")
        create_backup_if_exists(output_path)
        
    else:
        print(f"🆕 Starting new generation to {output_path}")
    
    # Initialize generator
    generator = DrillGenerator(config, dataset, difficulty, "Thai")
    
    # Run generation
    await generator.generate_all_drills(output_path)

if __name__ == "__main__":
    asyncio.run(main())