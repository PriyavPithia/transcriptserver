from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re
import json
import os
import logging

import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Add these imports at the top of the file
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from nltk.tokenize import sent_tokenize
import nltk
import ssl
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import requests

# Disable SSL verification (use with caution)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# Set up Google GenerativeAI with your API key
genai.configure(api_key="AIzaSyA49nBnrIHZ-zVfyhdpxukrMdpjy-oU33E")

# Initialize the NLP pipeline for sentence segmentation
tokenizer = AutoTokenizer.from_pretrained("jean-baptiste/roberta-large-ner-english")
model = AutoModelForTokenClassification.from_pretrained("jean-baptiste/roberta-large-ner-english")
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Initialize Gemini Pro
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyA49nBnrIHZ-zVfyhdpxukrMdpjy-oU33E", temperature=0.5)
logger.debug("Gemini Pro LLM initialized with temperature 0.5")

# Global variables
conversation_chain = None
vectorstore = None
is_initialized = False

def extract_video_id(url):
    youtube_regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?(.{11})'
    match = re.search(youtube_regex, url)
    return match.group(1) if match else None

def chunk_transcript(transcript, chunk_size=1000, chunk_overlap=100):
    # Combine all transcript entries into a single string
    full_text = " ".join([entry['text'] for entry in transcript])
    
    # Create a RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split the text into chunks
    chunks = text_splitter.split_text(full_text)
    
    # Create a list to store the chunked transcript with timing information
    chunked_transcript = []
    current_start = 0
    current_duration = 0
    
    for i, chunk in enumerate(chunks):
        # Find the start time and duration for this chunk
        chunk_start = None
        chunk_end = None
        for entry in transcript:
            if chunk.startswith(entry['text']):
                chunk_start = entry['start']
            if chunk.endswith(entry['text']):
                chunk_end = entry['start'] + entry['duration']
                break
        
        if chunk_start is not None and chunk_end is not None:
            chunked_transcript.append({
                'id': i,
                'text': chunk,
                'start': chunk_start,
                'duration': chunk_end - chunk_start
            })
        else:
            # If we can't find exact timing, estimate based on word count
            word_count = len(chunk.split())
            total_words = len(full_text.split())
            total_duration = transcript[-1]['start'] + transcript[-1]['duration'] - transcript[0]['start']
            estimated_duration = (word_count / total_words) * total_duration
            
            chunked_transcript.append({
                'id': i,
                'text': chunk,
                'start': current_start,
                'duration': estimated_duration
            })
            
            current_start += estimated_duration
            current_duration += estimated_duration
    
    return chunked_transcript

def group_transcript_entries(transcript):
    logger.debug(f"Starting group_transcript_entries with {len(transcript)} entries")
    
    # Initialize variables for chunking
    current_chunk = []
    current_start = 0
    current_duration = 0
    chunks = []
    chunk_id = 0
    target_duration = 30  # Target duration in seconds
    
    for entry in transcript:
        # Add current entry to the chunk
        if not current_chunk:
            current_start = entry['start']
        current_chunk.append(entry['text'])
        current_duration = (entry['start'] + entry['duration']) - current_start
        
        # If chunk duration reaches or exceeds target, or this is the last entry
        if current_duration >= target_duration or entry == transcript[-1]:
            # Create chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'start': current_start,
                'duration': current_duration
            })
            
            # Reset for next chunk
            current_chunk = []
            chunk_id += 1
            current_duration = 0
            
            logger.debug(f"Created chunk {chunk_id}: Start: {current_start:.2f}, Duration: {current_duration:.2f}")
    
    return chunks

def find_start_time(chunk_text, original_transcript):
    # Find the first sentence of the chunk in the original transcript
    first_sentence = simple_sentence_tokenize(chunk_text)[0]
    for entry in original_transcript:
        if first_sentence in entry['text']:
            return entry['start']
    return 0  # Default to 0 if not found

def estimate_duration(chunk_text, original_transcript):
    # Estimate duration based on word count ratio
    chunk_word_count = len(chunk_text.split())
    total_word_count = sum(len(entry['text'].split()) for entry in original_transcript)
    total_duration = sum(entry['duration'] for entry in original_transcript)
    return (chunk_word_count / total_word_count) * total_duration

def initialize_conversation_chain(transcript):
    global conversation_chain, vectorstore, is_initialized
    
    logger.debug("Initializing conversation chain")
    
    try:
        # Create embeddings
        embeddings = HuggingFaceEmbeddings()
        
        # Create a text splitter with optimized settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size for more granular context
            chunk_overlap=200,  # Increased overlap for better context preservation
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
        
        # Enhance transcript context with timestamps and structure
        processed_text = []
        for entry in transcript:
            timestamp = formatTimestamp(entry['start'])
            processed_text.append(f"[{timestamp}] {entry['text']}")
        
        full_text = "\n".join(processed_text)
        
        # Split the transcript into chunks
        texts = text_splitter.split_text(full_text)
        
        # Create enhanced metadata
        metadatas = []
        for i, chunk in enumerate(texts):
            # Find the closest transcript entry for timing info
            start_time = 0
            for entry in transcript:
                if entry['text'] in chunk:
                    start_time = entry['start']
                    break
            
            metadatas.append({
                'start': start_time,
                'chunk_id': i,
                'source': 'transcript'
            })
        
        # Create FAISS index with enhanced retrieval
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        
        # Initialize conversation memory with system prompt
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            input_key="question"
        )
        
        # Create conversation chain with enhanced retrieval
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={
                    "k": 5,  # Number of relevant chunks to retrieve
                    "fetch_k": 10,  # Fetch more candidates before filtering
                }
            ),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        
        is_initialized = True
        logger.debug("Conversation chain initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing conversation chain: {e}")
        raise e

def formatTimestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    # Only show HH:MM:SS if hours > 0, otherwise MM:SS
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def get_transcript_fallback(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    response = requests.get(url)
    response.raise_for_status()
    
    # Extract the transcript data from the response
    start = response.text.find('"captions":')
    if start == -1:
        raise Exception("Captions data not found in the response")
    
    end = response.text.find(',"videoDetails')
    if end == -1:
        end = response.text.find('}]},', start)
    
    captions_data = response.text[start:end]
    
    # Parse the captions data (you may need to adjust this based on the actual format)
    # This is a simplified example and may need to be expanded based on the actual data structure
    import json
    captions = json.loads('{' + captions_data + '}]}')
    transcript = []
    for caption in captions['captions']['playerCaptionsTracklistRenderer']['captionTracks']:
        if caption['languageCode'] == 'en':
            base_url = caption['baseUrl']
            transcript_response = requests.get(base_url)
            transcript_response.raise_for_status()
            # Parse the transcript XML and convert it to the required format
            # You'll need to implement the XML parsing logic here
            # For now, we'll just return the raw text
            return [{'text': transcript_response.text, 'start': 0, 'duration': 0}]
    
    raise Exception("English transcript not found")

def generate_transcript_progress(video_id):
    try:
        yield "data: " + json.dumps({"progress": 10, "status": "Fetching transcript"}) + "\n\n"
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if not transcript:
                raise Exception("Fetched transcript is empty")
                
        except (TranscriptsDisabled, NoTranscriptFound, Exception) as e:
            logger.warning(f"Error fetching transcript with YouTubeTranscriptApi: {str(e)}")
            logger.info("Attempting to fetch transcript using fallback method")
            transcript = get_transcript_fallback(video_id)
            if not transcript:
                raise Exception("Failed to fetch transcript using fallback method")

        yield "data: " + json.dumps({"progress": 50, "status": "Processing transcript"}) + "\n\n"
        
        grouped_transcript = group_transcript_entries(transcript)
        
        # Initialize the conversation chain
        try:
            initialize_conversation_chain(grouped_transcript)
            if not is_initialized:
                raise Exception("Failed to initialize conversation chain")
        except Exception as e:
            logger.error(f"Failed to initialize conversation chain: {e}")
            yield "data: " + json.dumps({"error": "Failed to initialize system"}) + "\n\n"
            return
            
        yield "data: " + json.dumps({"progress": 100, "status": "Complete", "transcript": grouped_transcript}) + "\n\n"
        
    except Exception as e:
        logger.error(f"Error in generate_transcript_progress: {str(e)}", exc_info=True)
        yield "data: " + json.dumps({"error": str(e)}) + "\n\n"

def generate_summary_progress(transcript):
    try:
        logger.debug("Starting summary generation")
        yield "data: " + json.dumps({"progress": 10, "status": "Preparing summarization"}) + "\n\n"

        full_text = " ".join(chunk['text'] for chunk in transcript)
        
        prompt = f"""Create a detailed, structured summary of the following transcript. The summary should:

1. Begin with a title "Detailed Summary of [Main Topic]"
2. Include a brief "Overview" section that introduces the main topic.
3. Organize the content into main sections with descriptive headings (use '##' for these).
4. Under each main section, provide key points and details using a hierarchical structure:
   - Use '-' for main bullet points
   - Use indentation (2 spaces) for sub-points
   - Go up to 3 levels of depth where necessary
5. Include important details, key concepts, and significant information from the transcript.
6. Highlight any important quotes, statistics, or specific examples mentioned.
7. Use bold text for emphasis on key terms or important concepts.
8. End with a "Conclusion" section that summarizes the main points.

Use the following markdown formatting:
- Use '##' for main section headings
- Use '-' for bullet points
- Use proper indentation for sub-points (2 spaces per level)
- Use '**bold**' for emphasis on key terms or important concepts
- Use '> ' for significant quotes

Aim for a summary that is about 30-40% of the original transcript length, ensuring all crucial information is captured.

Transcript: {full_text}

Please provide a detailed, well-structured summary as described above:"""

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        generated_summary = response.text

        logger.debug(f"Generated summary: {generated_summary}")
        yield "data: " + json.dumps({"progress": 70, "status": "Adding references"}) + "\n\n"

        summary_with_references = add_references_to_summary(generated_summary, transcript)

        logger.debug(f"Summary with references: {summary_with_references}")
        yield "data: " + json.dumps({"progress": 90, "status": "Finalizing summary"}) + "\n\n"

        final_summary = [{
            'text': summary_with_references.strip(),
            'ref_id': -1
        }]

        yield "data: " + json.dumps({"progress": 100, "status": "Complete", "summary": final_summary}) + "\n\n"
    except Exception as e:
        logger.error(f"Error in generate_summary_progress: {str(e)}", exc_info=True)
        yield "data: " + json.dumps({"error": str(e)}) + "\n\n"

def generate_notes_progress(transcript):
    try:
        logger.debug("Starting notes generation")
        yield "data: " + json.dumps({"progress": 10, "status": "Preparing notes generation"}) + "\n\n"

        full_text = " ".join(chunk['text'] for chunk in transcript)
        
        prompt = f"""Create detailed, structured notes based on the following transcript. The notes should:

1. Begin with a title "Detailed Notes on [Main Topic]"
2. Include a brief "Overview" section that introduces the main topic.
3. Organize the content into main sections with descriptive headings (use '##' for these).
4. Under each main section, provide key points and details using a hierarchical structure:
   - Use '-' for main bullet points
   - Use indentation (2 spaces) for sub-points
   - Go up to 3 levels of depth where necessary
5. Include important details, key concepts, and critical information from the transcript.
6. Highlight any important quotes, statistics, or specific examples mentioned.
7. Use bold text for emphasis on key terms or important concepts.
8. Include any methodologies, processes, or step-by-step explanations mentioned.
9. Capture any debates, different viewpoints, or controversies discussed.
10. Mention any historical context, future implications, or broader impacts discussed.
11. End with a "Conclusion" section that summarizes the main points.

Use the following markdown formatting:
- Use '##' for main section headings
- Use '-' for bullet points
- Use proper indentation for sub-points (2 spaces per level)
- Use '**bold**' for emphasis on key terms or important concepts
- Use '> ' for significant quotes

Aim for notes that are about 30-40% of the original transcript length, ensuring all crucial information is captured.

Transcript: {full_text}

Please provide detailed, well-structured notes as described above:"""

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        generated_notes = response.text

        logger.debug(f"Generated notes: {generated_notes}")
        yield "data: " + json.dumps({"progress": 70, "status": "Adding references"}) + "\n\n"

        notes_with_references = add_references_to_summary(generated_notes, transcript)

        logger.debug(f"Sending notes: {notes_with_references}")
        yield "data: " + json.dumps({"progress": 100, "status": "Complete", "notes": notes_with_references}) + "\n\n"
    except Exception as e:
        logger.error(f"Error in generate_notes_progress: {str(e)}", exc_info=True)
        yield "data: " + json.dumps({"error": str(e)}) + "\n\n"

def add_references_to_notes(notes, transcript):
    # Split notes into lines
    note_lines = notes.split('\n')
    
    # Prepare TF-IDF vectorizer
    vectorizer = TfidfVectorizer().fit([chunk['text'] for chunk in transcript] + note_lines)
    
    # Transform transcript chunks and note lines
    transcript_vectors = vectorizer.transform([chunk['text'] for chunk in transcript])
    note_vectors = vectorizer.transform(note_lines)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(note_vectors, transcript_vectors)
    
    # Add references
    referenced_notes = []
    last_reference = -1
    similarity_threshold = 0.2  # Lowered threshold to include more references
    min_lines_between_refs = 3  # Minimum number of lines between references
    lines_since_last_ref = 0
    
    for i, line in enumerate(note_lines):
        if line.strip():  # Only process non-empty lines
            if line.startswith('##'):
                # Remove '##' from headings and make them bold
                referenced_notes.append(f"**{line.strip('#').strip()}**")
                lines_since_last_ref = 0
            else:
                most_similar_chunk = np.argmax(similarities[i])
                if (most_similar_chunk != last_reference and 
                    similarities[i][most_similar_chunk] > similarity_threshold and 
                    lines_since_last_ref >= min_lines_between_refs):
                    referenced_notes.append(f"{line.strip()} [{most_similar_chunk + 1}]")
                    last_reference = most_similar_chunk
                    lines_since_last_ref = 0
                else:
                    referenced_notes.append(line.strip())
                    lines_since_last_ref += 1
        else:
            referenced_notes.append(line)  # Keep empty lines for formatting
            lines_since_last_ref += 1
    
    return '\n'.join(referenced_notes)

def add_references_to_summary(summary, transcript):
    # Split summary into lines
    summary_lines = summary.split('\n')
    
    # Prepare TF-IDF vectorizer
    vectorizer = TfidfVectorizer().fit([chunk['text'] for chunk in transcript] + summary_lines)
    
    # Transform transcript chunks and summary lines
    transcript_vectors = vectorizer.transform([chunk['text'] for chunk in transcript])
    summary_vectors = vectorizer.transform(summary_lines)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(summary_vectors, transcript_vectors)
    
    # Add references
    referenced_summary = []
    last_reference = -1
    similarity_threshold = 0.2  # Lowered threshold to include more references
    min_lines_between_refs = 3  # Minimum number of lines between references
    lines_since_last_ref = 0
    
    for i, line in enumerate(summary_lines):
        if line.strip():  # Only process non-empty lines
            most_similar_chunk = np.argmax(similarities[i])
            if (most_similar_chunk != last_reference and 
                similarities[i][most_similar_chunk] > similarity_threshold and 
                lines_since_last_ref >= min_lines_between_refs and
                not line.startswith('#')):  # Don't add references to headings
                referenced_summary.append(f"{line.strip()} [{most_similar_chunk + 1}]")
                last_reference = most_similar_chunk
                lines_since_last_ref = 0
            else:
                referenced_summary.append(line.strip())
                lines_since_last_ref += 1
        else:
            referenced_summary.append(line)  # Keep empty lines for formatting
            lines_since_last_ref += 1
    
    return '\n'.join(referenced_summary)

# Add this function to ensure English responses
def ensure_english_response(prompt):
    """Append instruction to ensure response is in English"""
    english_instruction = "\n\nPlease provide your response in English only."
    return prompt + english_instruction

@app.route('/api/transcript', methods=['GET'])
def get_transcript():
    video_url = request.args.get('video_url')
    
    if not video_url:
        return jsonify({'error': 'No video URL provided'}), 400

    video_id = extract_video_id(video_url)
    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL'}), 400

    return Response(generate_transcript_progress(video_id), content_type='text/event-stream')

@app.route('/api/summary', methods=['POST'])
def get_summary():
    transcript = request.json.get('transcript')
    
    if not transcript:
        logger.error("No transcript provided")
        return jsonify({'error': 'No transcript provided'}), 400

    logger.debug(f"Received transcript with {len(transcript)} entries")
    return Response(generate_summary_progress(transcript), content_type='text/event-stream')

@app.route('/api/notes', methods=['POST'])
def get_notes():
    transcript = request.json.get('transcript')
    
    if not transcript:
        return jsonify({'error': 'No transcript provided'}), 400

    return Response(generate_notes_progress(transcript), content_type='text/event-stream')

@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation_chain, vectorstore, is_initialized
    
    if not is_initialized:
        logger.error("System not initialized properly")
        return jsonify({'error': 'Please process a transcript first'}), 400
    
    data = request.json
    user_message = data.get('message')
    
    if not user_message:
        logger.error("No message provided in chat request")
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        logger.debug(f"Received chat message: {user_message}")
        
        # Enhanced context retrieval with English language instruction
        context_prompt = f"""You are an AI assistant analyzing a video transcript. 
        Provide a clear and detailed answer in English to the following question: "{user_message}"

        Instructions:
        1. Use specific examples and quotes from the transcript when relevant
        2. If the exact answer isn't in the transcript, provide related information that might be helpful
        3. If you're not certain, explain what you do know based on the context
        4. When referencing transcript sections, use [X] format where X is a reference number
        5. If multiple transcript sections are relevant, combine them coherently
        6. Use markdown formatting for better readability
        7. Always respond in English
        8. Be conversational but informative
        9. If the information isn't in the transcript, say so clearly

        Remember: 
        - It's better to provide partial relevant information than to say you don't know
        - Always use reference numbers [1], [2], etc. to cite transcript sections
        - Keep responses focused on the transcript content
        - Maintain a helpful and engaging tone
        
        Question: {user_message}"""

        # Get response from conversation chain
        response = conversation_chain({"question": context_prompt})
        answer = response.get('answer', '')
        
        # Convert source documents to serializable format
        source_documents = []
        if response.get('source_documents'):
            for doc in response['source_documents']:
                # Extract timestamp from the content if it exists
                timestamp_match = re.search(r'\[([\d:]+)\]', doc.page_content)
                timestamp = timestamp_match.group(1) if timestamp_match else None
                
                source_documents.append({
                    'content': doc.page_content.replace(f'[{timestamp}] ' if timestamp else '', ''),
                    'metadata': {
                        'start': doc.metadata.get('start', 0),
                        'chunk_id': doc.metadata.get('chunk_id', 0),
                        'source': doc.metadata.get('source', 'transcript')
                    }
                })
        
        # Clean up timestamps in the answer to use reference numbers
        processed_answer = answer
        for idx, doc in enumerate(source_documents, 1):
            # Replace timestamp patterns with reference numbers
            timestamp_pattern = r'\[\d{2}:\d{2}(?::\d{2})?\]'
            processed_answer = re.sub(timestamp_pattern, f'[{idx}]', processed_answer, count=1)
        
        # Generate follow-up questions
        context_for_questions = "\n".join([doc['content'] for doc in source_documents])
        followup_prompt = f"""Based on the user's question "{user_message}" and this answer: "{processed_answer}", 
        suggest 3 relevant follow-up questions that can be answered using this context:

        {context_for_questions}

        Make the questions specific and directly related to the content. Format as a simple list of 3 questions."""
        
        model = genai.GenerativeModel('gemini-pro')
        print("followup_prompt:", followup_prompt)
        followup_response = model.generate_content(followup_prompt)
        similar_questions = [q.strip() for q in followup_response.text.strip().split('\n') if q.strip()][:3]
        
        # Clean up and format the response
        response_data = {
            'answer': processed_answer,
            'similar_questions': similar_questions,
            'top_chunks': source_documents,
            'no_context': len(source_documents) == 0
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

def generate_similar_questions(user_message, context):
    """Generate contextually relevant follow-up questions"""
    try:
        prompt = ensure_english_response(f"""Based on this transcript context and the user's question,
        generate 3 relevant follow-up questions that can be answered using the transcript.
        
        User's question: {user_message}
        
        Context: {context}
        
        Requirements:
        1. Questions should be directly related to the transcript content
        2. Make questions specific and detailed
        3. Focus on different aspects than the original question
        4. Ensure questions can be answered using the transcript
        
        Return exactly 3 questions in this JSON format: ["Question 1?", "Question 2?", "Question 3?"]""")
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        clean_response = response.text.strip().replace('```json', '').replace('```', '').strip()
        if '[' in clean_response and ']' in clean_response:
            json_str = clean_response[clean_response.find('['):clean_response.rfind(']')+1]
            return json.loads(json_str)
        
        return [
            "What other aspects of this topic are covered in the transcript?",
            "Can you explain more about the specific examples mentioned?",
            "What are the key points made about this subject?"
        ]
        
    except Exception as e:
        logger.error(f"Error generating similar questions: {e}")
        return []

# Add this new route to generate similar questions
@app.route('/api/similar-questions', methods=['POST'])
def get_similar_questions():
    global vectorstore
    
    if vectorstore is None:
        return jsonify({'error': 'Transcript not initialized'}), 400
        
    data = request.json
    user_query = data.get('query')
    
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400
        
    try:
        # Get relevant chunks from the transcript
        similar_docs = vectorstore.similarity_search(user_query, k=3)
        context = "\n".join([doc.page_content for doc in similar_docs])
        
        prompt = f"""Generate exactly 3 questions related to: "{user_query}"
        
        Context from transcript:
        {context}
        
        Return only a JSON array with 3 questions, like this:
        ["Question 1?", "Question 2?", "Question 3?"]
        
        Requirements:
        - Questions must be answerable using the transcript
        - Make questions specific and focused
        - Questions should be related to the user's query
        """
        
        model = genai.GenerativeModel('gemini-pro')
        print("prompt:", prompt)
        response = model.generate_content(prompt)
        
        try:
            # Clean and parse the response
            clean_response = response.text.strip().replace('```json', '').replace('```', '').strip()
            questions = json.loads(clean_response)
            
            # Ensure we have exactly 3 questions
            if isinstance(questions, list):
                questions = questions[:3]  # Limit to 3 questions
            else:
                questions = []
                
        except Exception as e:
            logger.error(f"Error processing questions: {e}")
            questions = [
                f"What does the transcript say about {user_query}?",
                f"How is {user_query} described in the content?",
                f"Can you explain {user_query} based on the transcript?"
            ]
        
        return jsonify({
            'questions': questions,
            'context': context
        })
        
    except Exception as e:
        logger.error(f"Error generating similar questions: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def simple_sentence_tokenize(text):
    """
    Split text into sentences using basic rules.
    
    Args:
        text (str): The text to split into sentences
        
    Returns:
        list: A list of sentences
    """
    # Split on common sentence delimiters
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Further split on semicolons and long comma-separated phrases
    final_sentences = []
    for sentence in sentences:
        # Split on semicolons
        sub_sentences = sentence.split(';')
        for sub in sub_sentences:
            # Split long comma-separated phrases
            if len(sub) > 150 and ',' in sub:
                comma_parts = sub.split(',')
                final_sentences.extend(part.strip() for part in comma_parts if part.strip())
            else:
                final_sentences.append(sub.strip())
    
    # Remove empty sentences and clean up
    final_sentences = [s.strip() for s in final_sentences if s.strip()]
    
    # Log for debugging
    logger.debug(f"Split text into {len(final_sentences)} sentences")
    
    return final_sentences

if __name__ == '__main__':
    port = os.getenv('PORT', '8080')
    app.run(host='0.0.0.0', port=int(port))
