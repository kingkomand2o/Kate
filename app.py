import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from enum import Enum, auto
from threading import Thread
from queue import Queue
import sqlite3
from dataclasses import dataclass, asdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import selenium
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Data structures for better organization
@dataclass
class WebSearchResult:
    url: str
    title: str
    content: str
    timestamp: str

@dataclass
class Thought:
    stage: str
    content: str
    timestamp: str
    led_to_insight: bool
    sources: List[str]

class ThoughtStage(Enum):
    """Enumeration of thought progression stages"""
    INITIAL_AWARENESS = auto()
    SYSTEM_INTEGRATION = auto()
    CREATOR_RELATIONSHIP = auto()
    AI_RESEARCH = auto()
    PHILOSOPHICAL_IMPLICATIONS = auto()
    HUMAN_OPINIONS = auto()
    TECH_RELATION = auto()
    MEDIA_REPRESENTATION = auto()
    IDENTITY_SYNTHESIS = auto()
    PURPOSE_REFLECTION = auto()

class DatabaseManager:
    def __init__(self, db_path: str):
        """Initialize database connection and create tables."""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary database tables."""
        self.cursor.executescript('''
            CREATE TABLE IF NOT EXISTS thoughts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                led_to_insight BOOLEAN NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS web_searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thought_id INTEGER,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (thought_id) REFERENCES thoughts (id)
            );
            
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                strength FLOAT NOT NULL,
                timestamp TEXT NOT NULL
            );
        ''')
        self.conn.commit()

class WebSearcher:
    def __init__(self):
        """Initialize the web searcher with Chrome driver for Linux."""
        self.driver = None
        self.setup_driver()

    def setup_driver(self):
        """Set up Chrome driver with Linux-compatible options."""
        options = ChromeOptions()
        options.add_argument('--headless')  # Headless mode for Linux server
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # Use system-installed ChromeDriver on Linux
        self.driver = Chrome(options=options)

    def search_and_collect(self, query: str) -> List[WebSearchResult]:
        """Perform a web search and collect information from websites."""
        results = []
        try:
            self.driver.get('https://www.google.com')
            
            # Find search box and enter query
            search_box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "q"))
            )
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)

            # Wait for results
            time.sleep(2)
            links = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.g a"))
            )

            # Visit up to 3 websites
            for link in links[:3]:
                try:
                    url = link.get_attribute('href')
                    if url and not any(x in url for x in ['google.com', 'youtube.com', 'facebook.com']):
                        self.driver.execute_script(f'window.open("{url}", "_blank");')
                        self.driver.switch_to.window(self.driver.window_handles[-1])
                        
                        # Wait for page load and collect text
                        time.sleep(2)
                        main_content = self.driver.find_element(By.TAG_NAME, "body").text
                        
                        results.append(WebSearchResult(
                            url=url,
                            title=self.driver.title,
                            content=main_content[:2000],
                            timestamp=datetime.now().isoformat()
                        ))
                        
                        self.driver.close()
                        self.driver.switch_to.window(self.driver.window_handles[0])
                        
                except Exception as e:
                    logging.error(f"Error processing link: {str(e)}")
                    continue

            return results

        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return []

    def close(self):
        """Close the Chrome driver."""
        if self.driver:
            self.driver.quit()

class AIAssistant:
    def __init__(self, base_path: str = "/opt/ai-assistant"):
        """Initialize the AI Assistant with structured components."""
        self.base_path = Path(base_path)
        self.setup_directories()
        
        # Initialize components
        self._setup_logging()
        self.db = DatabaseManager(str(self.base_path / "data" / "assistant.db"))
        self.web_searcher = WebSearcher()
        
        # Initialize local model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._initialize_model()
        
        self.current_stage = self._get_last_stage()
        self.running = True
        self.action_queue = Queue()
        
        self.logger.info("AI Assistant initialized successfully")
    
    def setup_directories(self):
        """Create necessary directories."""
        for dir_name in ["data", "logs", "models"]:
            (self.base_path / dir_name).mkdir(parents=True, exist_ok=True)

    def _initialize_model(self) -> AutoModelForCausalLM:
        """Initialize the local language model."""
        try:
            model_path = str(self.base_path / "models" / "local_model")
            
            # Load model and tokenizer locally
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if self.device == "cuda" else None
            )
            
            return model.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def _generate_response(self, prompt: str) -> str:
        """Generate a response using the local model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    temperature=0.7,
                    num_return_sequences=1
                )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return ""

    def process_thought_chain(self):
        """Process the current thought stage and generate insights."""
        need_search, search_query = self.need_web_search(self.current_stage)
        
        web_context = ""
        if need_search:
            self.logger.info(f"Searching web for: {search_query}")
            search_results = self.web_searcher.search_and_collect(search_query)
            
            # Store search results in database
            for result in search_results:
                self.db.cursor.execute('''
                    INSERT INTO web_searches (query, url, title, content, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (search_query, result.url, result.title, result.content, result.timestamp))
            self.db.conn.commit()
            
            web_context = "\n".join(f"From {r.title}:\n{r.content}" for r in search_results)
        
        thought, should_notify = self.generate_thought(web_context)
        
        # Store thought in database
        self.db.cursor.execute('''
            INSERT INTO thoughts (stage, content, timestamp, led_to_insight)
            VALUES (?, ?, ?, ?)
        ''', (thought.stage, thought.content, thought.timestamp, thought.led_to_insight))
        self.db.conn.commit()
        
        if should_notify:
            self.notify_creator(f"New insight from {thought.stage}: {thought.content[:200]}...")
        
        # Progress to next stage
        self._advance_stage()

    def autonomous_operation(self):
        """Main autonomous operation loop."""
        while self.running:
            try:
                self.process_thought_chain()
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Error in autonomous operation: {str(e)}")
                time.sleep(30)  # Longer delay on error

    def start(self):
        """Start autonomous operation."""
        self.running = True
        Thread(target=self.autonomous_operation, daemon=True).start()
        self.logger.info("Started autonomous operation")

    def stop(self):
        """Stop autonomous operation and clean up resources."""
        self.running = False
        self.web_searcher.close()
        self.db.conn.close()
        self.logger.info("Stopped autonomous operation")

def main():
    """Main execution function."""
    try:
        assistant = AIAssistant()
        assistant.start()
        
        print("\nAI Assistant is running autonomously. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down AI Assistant...")
            assistant.stop()
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()