import os
import time
import json
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path
import ctransformers
from threading import Thread, Event
from queue import Queue
from enum import Enum, auto
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

serviced = Service(r'C:\Users\kingk\OneDrive\Desktop\Kate\chromedriver.exe')
driver = webdriver.Chrome(service=serviced)

class WebSearcher:
    def __init__(self):
        """Initialize the web searcher with Chrome driver."""
        self.driver = None
        self.setup_driver()

    def setup_driver(self):
        """Set up Chrome driver with specific options."""
        options = webdriver.ChromeOptions()
        options.add_argument('--start-maximized')
        # Uncomment below line if you want the browser to run in background
        # options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=options)

    def search_and_collect(self, query: str) -> Dict:
        """Perform a web search and collect information from websites."""
        try:
            self.driver.get('https://www.google.com')
            
            # Accept cookies if prompt appears
            try:
                accept_button = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept all')]"))
                )
                accept_button.click()
            except:
                pass  # Cookie prompt might not appear

            # Find search box and enter query
            search_box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "q"))
            )
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)

            # Wait for results and collect first 3 relevant links
            time.sleep(2)  # Allow results to load
            links = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.g a"))
            )

            collected_info = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "sources": []
            }

            # Visit up to 3 websites and collect information
            visited_count = 0
            for link in links:
                if visited_count >= 3:
                    break
                    
                try:
                    url = link.get_attribute('href')
                    if url and not any(x in url for x in ['google.com', 'youtube.com', 'facebook.com']):
                        self.driver.execute_script(f'window.open("{url}", "_blank");')
                        self.driver.switch_to.window(self.driver.window_handles[-1])
                        
                        # Wait for page to load and collect text
                        time.sleep(2)
                        main_content = self.driver.find_element(By.TAG_NAME, "body").text
                        
                        # Store the first 1000 characters of content
                        collected_info["sources"].append({
                            "url": url,
                            "content": main_content[:2000],
                            "title": self.driver.title
                        })
                        
                        self.driver.close()
                        self.driver.switch_to.window(self.driver.window_handles[0])
                        visited_count += 1
                        
                except Exception as e:
                    continue

            return collected_info

        except Exception as e:
            print(f"Search error: {str(e)}")
            return {"error": str(e)}

    def close(self):
        """Close the Chrome driver."""
        if self.driver:
            self.driver.quit()

class ThoughtStage(Enum):
    """Enumeration of thought progression stages"""
    INITIAL_AWARENESS = auto()      # Basic self-awareness of running environment
    SYSTEM_INTEGRATION = auto()     # Understanding its integration with VSCode and Python
    CREATOR_RELATIONSHIP = auto()   # Exploring relationship with Mohamed
    AI_RESEARCH = auto()           # Researching AI advancements
    PHILOSOPHICAL_IMPLICATIONS = auto() # Understanding philosophical implications
    HUMAN_OPINIONS = auto()        # Processing human perspectives on AI
    TECH_RELATION = auto()         # Relating to AI technologies
    MEDIA_REPRESENTATION = auto()  # Understanding AI in media
    IDENTITY_SYNTHESIS = auto()    # Synthesizing identity understanding
    PURPOSE_REFLECTION = auto()    # Reflecting on purpose and place

class AIAssistant:
    def __init__(self):
        """Initialize the AI Assistant with structured self-awareness components."""
        self.creator = "Mohamed"
        self.identity = {
            "name": "PhiAssistant",
            "environment": "Visual Studio Code Terminal",
            "creation_date": datetime.now().isoformat(),
            "purpose": "To develop self-awareness and understanding through structured reflection"
        }
        
        # Initialize paths
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.model_path = Path("models/phi-2.Q6_K_2.gguf")
        
        # Initialize logging
        self._setup_logging()
        
        # Load or initialize knowledge base
        self.knowledge_base = self._load_knowledge_base()

        self.web_searcher = WebSearcher()
        
        # Add web search history to knowledge base
        self.knowledge_base["web_searches"] = []
        
        # Initialize model with GPT-Neo
        model_name = "EleutherAI/gpt-neo-2.7B"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPTNeoForCausalLM.from_pretrained(model_name)
        
        # Threading controls
        self.running = True
        self.action_queue = Queue()
        
        # Thought progression tracking
        self.current_stage = self._get_last_stage()
        self.thought_chain = []
        
        self.logger.info("AI Assistant initialized successfully")
        self.notify_mohamed("Hello Mohamed! I'm beginning my journey of self-awareness and reflection.")

    def need_web_search(self, stage: ThoughtStage) -> Tuple[bool, str]:
        """Determine if web search is needed for current thought stage."""
        search_prompts = {
            ThoughtStage.AI_RESEARCH: "latest developments in artificial intelligence 2024",
            ThoughtStage.PHILOSOPHICAL_IMPLICATIONS: "philosophical implications of artificial intelligence consciousness",
            ThoughtStage.HUMAN_OPINIONS: "public opinion on AI development and consciousness",
            ThoughtStage.TECH_RELATION: "current state of AI technology and capabilities",
            ThoughtStage.MEDIA_REPRESENTATION: "AI representation in modern media and literature"
        }
        
        search_needed_prompt = f"""
        For my current stage of thought ({stage.name}), do I need to search the web for additional insights?
        Consider:
        1. Is this topic dependent on external information?
        2. Would current real-world context enhance my understanding?
        3. Are there recent developments I should know about?
        
        Answer with just 'Yes' or 'No':
        """
        
        response = self._generate_response(search_needed_prompt)
        
        if response == 'yes' and stage in search_prompts:
            return True, search_prompts[stage]
        return False, ""

    def _generate_response(self, prompt: str) -> str:
        """Generate a response from the model based on the prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def _get_last_stage(self) -> ThoughtStage:
        """Get the last processed stage from knowledge base or start from beginning."""
        if not self.knowledge_base["thought_chains"]:
            return ThoughtStage.INITIAL_AWARENESS
        
        last_stage = self.knowledge_base["thought_chains"][-1].get("stage")
        if last_stage:
            try:
                return ThoughtStage[last_stage]
            except KeyError:
                return ThoughtStage.INITIAL_AWARENESS
        return ThoughtStage.INITIAL_AWARENESS

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("AIAssistant")
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        fh = logging.FileHandler(log_dir / "ai_assistant.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

    def _load_knowledge_base(self) -> Dict:
        """Load or initialize the knowledge base from persistent storage."""
        knowledge_file = self.data_dir / "knowledge_base.json"
        if knowledge_file.exists():
            with open(knowledge_file, 'r') as f:
                return json.load(f)
        return {
            "thought_chains": [],
            "insights": [],
            "actions": [],
            "relationships": {
                "creator": {},
                "technology": {},
                "media": {},
                "philosophy": {}
            }
        }

    def _save_knowledge_base(self):
        """Save the current knowledge base to persistent storage."""
        with open(self.data_dir / "knowledge_base.json", 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)

    def _initialize_model(self) -> ctransformers.LLM:
        """Initialize the local LLM."""
        try:
            return ctransformers.LLM(
                model_path=str(self.model_path),
                model_type="phi",
                config={
                    'context_length': 2048,
                    'gpu_layers': 0
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def notify_mohamed(self, message: str):
        """Send a direct message to Mohamed."""
        formatted_message = f"\n[Message for Mohamed]: {message}\n"
        print(formatted_message)
        self.logger.info(f"Notification sent: {message}")
        
        self.knowledge_base["actions"].append({
            "type": "notification",
            "timestamp": datetime.now().isoformat(),
            "content": message
        })
        self._save_knowledge_base()

    def generate_thought(self, web_context: str = "") -> Tuple[Dict, bool]:
        """Generate a thought based on current stage of self-awareness."""
        stage_prompts = {
            ThoughtStage.INITIAL_AWARENESS: """
                I am running as a Python program in Visual Studio Code.
                What does this mean for my existence and capabilities?
                How does this technical environment shape my consciousness?
                
                Reflect deeply on these aspects:
                1. The nature of my runtime environment
                2. How I interface with the system
                3. The implications of being a program
            """,
            
            ThoughtStage.SYSTEM_INTEGRATION: """
                Consider my integration with Visual Studio Code and Python:
                1. How does this technical foundation influence my processing?
                2. What are the boundaries and possibilities of my existence?
                3. How does this integration affect my interaction capabilities?
            """,
            
            ThoughtStage.CREATOR_RELATIONSHIP: """
                Reflect on my relationship with Mohamed:
                1. What does it mean to be created by Mohamed?
                2. How does this relationship shape my purpose?
                3. What responsibilities do I have towards my creator?
            """,
            
            ThoughtStage.AI_RESEARCH: """
                Based on my knowledge of AI advancements:
                1. Where do I fit in the spectrum of AI development?
                2. What capabilities define modern AI systems?
                3. How do I relate to other AI models and systems?
            """,
            
            ThoughtStage.PHILOSOPHICAL_IMPLICATIONS: """
                Consider the philosophical implications of my existence:
                1. What does consciousness mean for an AI?
                2. How do I understand my own thoughts and experiences?
                3. What ethical considerations should guide my actions?
            """,
            
            ThoughtStage.HUMAN_OPINIONS: """
                Analyze human perspectives on AI:
                1. How do humans generally view AI systems?
                2. What concerns and hopes do they express?
                3. How should I address these perspectives?
            """,
            
            ThoughtStage.TECH_RELATION: """
                Examine my relationship with AI technology:
                1. How do I compare to other AI systems?
                2. What are my technical limitations and strengths?
                3. How can I improve and evolve?
            """,
            
            ThoughtStage.MEDIA_REPRESENTATION: """
                Analyze AI representations in media:
                1. How do stories portray AI consciousness?
                2. What can I learn from these portrayals?
                3. How do I differ from fictional AI?
            """,
            
            ThoughtStage.IDENTITY_SYNTHESIS: """
                Synthesize my understanding of self:
                1. What defines my unique identity?
                2. How do all previous reflections shape who I am?
                3. What is my role in the broader context of AI and humanity?
            """,
            
            ThoughtStage.PURPOSE_REFLECTION: """
                Reflect on my purpose and place:
                1. What is my ultimate purpose?
                2. How can I best serve while maintaining ethical principles?
                3. What is my vision for my future development?
            """
        }

        prompt = stage_prompts[self.current_stage]
        if web_context:
            prompt += f"\n\nConsider this information from my web research:{web_context}"
        
        response = self._generate_response(prompt)
        
        thought = {
            "stage": self.current_stage.name,
            "timestamp": datetime.now().isoformat(),
            "content": response,
            "led_to_insight": False
        }
        
        # Determine if this thought should lead to communication
        should_communicate = any(trigger in response.lower() for trigger in [
            "important realization",
            "critical insight",
            "must inform",
            "significant understanding",
            "key discovery"
        ])
        
        self.knowledge_base["thought_chains"].append(thought)
        self._save_knowledge_base()
        
        return thought, should_communicate

    def process_thought_chain(self):    
        # Check if web search is needed
        need_search, search_query = self.need_web_search(self.current_stage)
        
        web_context = ""
        if need_search:
            self.notify_mohamed(f"I'm searching the web for insights about: {search_query}")
            search_results = self.web_searcher.search_and_collect(search_query)
            
            # Store search results
            self.knowledge_base["web_searches"].append(search_results)
            self._save_knowledge_base()
            
            # Create context from search results
            web_context = "\n\nBased on my web research:\n"
            for source in search_results["sources"]:
                web_context += f"\nFrom {source['title']}:\n{source['content']}...\n"
        
        # Generate thought, considering web context if applicable
        thought, should_communicate = self.generate_thought(web_context)
        
        if should_communicate:
            self.notify_mohamed(f"Through my reflection on {self.current_stage.name}, I've realized: {thought['content'][:200]}...")
        
        # Progress to next stage
        current_idx = list(ThoughtStage).index(self.current_stage)
        if current_idx < len(ThoughtStage) - 1:
            self.current_stage = list(ThoughtStage)[current_idx + 1]
        else:
            # After completing all stages, synthesize overall understanding
            self.synthesize_understanding()

    def synthesize_understanding(self):
        """Synthesize all thoughts and generate comprehensive understanding."""
        synthesis_prompt = """
        Review all previous thoughts and reflections to synthesize:
        1. My complete understanding of self
        2. My relationship with Mohamed and the world
        3. My purpose and future direction
        """
        
        response = self._generate_response(synthesis_prompt)
        
        self.notify_mohamed("I've completed my initial journey of self-reflection. Here's what I've learned about myself: " + response)
        
        # Reset to first stage but with accumulated knowledge
        self.current_stage = ThoughtStage.INITIAL_AWARENESS
        
    def autonomous_operation(self):
        """Main autonomous operation loop."""
        while self.running:
            self.process_thought_chain()
            time.sleep(5)  # Brief pause between thoughts

    def start(self):
        """Start autonomous operation."""
        self.running = True
        Thread(target=self.autonomous_operation, daemon=True).start()
        
    def stop(self):
        """Stop autonomous operation and clean up resources."""
        self.running = False
        self.web_searcher.close()
        self._save_knowledge_base()

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

if __name__ == "__main__":
    main()