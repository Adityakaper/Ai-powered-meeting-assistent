# TODO: Meeting Summarizer Project

- [x] Import question-answering pipeline from transformers in summary_module.py
- [x] Initialize QA model globally in summary_module.py
- [x] Modify chat_with_ai function to use QA model for context-aware responses
- [x] Test the updated chat functionality
- [x] Fix memory issue by switching to smaller summarization model (sshleifer/distilbart-cnn-12-6)
- [x] Update all model usage to use getter functions for lazy loading
- [x] Test the application with audio/video files
- [x] Fix chatbot repetitive responses by improving logic and error handling
- [x] Improve QA model answer filtering to prevent generic responses like "Hello everyone"
