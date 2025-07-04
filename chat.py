# main.py - FastAPI Backend for Pharma Chatbot with Natural Language Response
import os
import json
import re
from warnings import filters
import openai
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
import uvicorn
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Literal, Dict
import google.generativeai as genai
# import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('natural_res.log'),
        logging.StreamHandler()
    ]
)

class PharmaChatbot:
    def __init__(self):
        """Initialize the Pharma Chatbot"""
        # Initialize OpenAI

        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        

        self.qdrant_client = QdrantClient(
        url="http://34.47.177.112:6333",
        timeout=30.0  
        )

        # self.collection_name = "Pharma-1"
        self.collection_name = "inference"

        # self.vector_dimension = 1536
        self.vector_dimension = 768

        self.conversation_history = {}  # Holds chat turns for each session
        self.max_history_turns = 5

        print("✅ Pharma Chatbot initialized with deployed Qdrant!")
        


    def detect_intent(self, user_query: str) -> str:
        """
        Detect user intent using GPT-4
        
        Args:
            user_query: The user's input query
            
        Returns:
            Intent as string: either "semantic_search" or "filter"
        """
        prompt = f"""You are an intent classifier for a pharmaceutical database chatbot.

    Given a user query, classify it into one of these intents:

    1. "semantic_search" - For queries asking for general information, performance, usage, effects, descriptions about medicines
    2. "filter" - For queries asking to find/list medicines based on specific themes, categories, region or theme reasons

    SEMANTIC SEARCH Examples (use when user wants to LEARN ABOUT or UNDERSTAND):
    - "How is Ospaan D performing?" → semantic_search
    Reason: User wants to understand performance information about a specific medicine
    - "What is Paracetamol used for?" → semantic_search  
    Reason: User wants to learn about usage/indications of a medicine
    - "Tell me about Aspirin side effects" → semantic_search
    Reason: User wants descriptive information about side effects
    - "How effective is Virilix?" → semantic_search
    Reason: User wants to understand effectiveness information
    - "What are the benefits of Ibuprofen?" → semantic_search
    Reason: User seeks descriptive information about benefits
    - "Explain the safety profile of Ospaan D" → semantic_search
    Reason: User wants detailed explanation about safety information
    - "What is the market performance of medicine X?" → semantic_search
    Reason: User wants to understand specific performance data

    FILTER Examples (use when user wants to FIND or LIST medicines):
    - "Give me the product efficacy based information on Ospaan D" → filter
    Reason: User wants to retrieve structured data/themes about efficacy
    - "List medicines with high efficacy" → filter
    Reason: User wants to find multiple medicines matching efficacy criteria
    - "Show me medicines with safety concerns" → filter
    Reason: User wants to filter medicines by safety theme/category
    - "Find medicines based on product performance" → filter
    Reason: User wants to search medicines using performance as filter criteria
    - "Which medicines are available in Mumbai?" → filter
    Reason: User wants to filter medicines by region
    - "Show all medicines in the Product Efficacy category" → filter
    Reason: User wants to filter by specific theme category
    - "Find medicines with theme reason containing 'fast action'" → filter
    Reason: User wants to filter by specific theme reasons
    - "Product performance and safety concerns for Virilix and Ospaan D" → filter
    Reason: User wants structured data for multiple medicines with specific criteria
    - "Give me all Safety Profile related information" → filter
    Reason: User wants to retrieve all items matching safety profile theme

    Key Decision Rules:
    - If query asks "What/How/Why/Explain/Tell me about" a specific medicine → semantic_search
    - If query asks "List/Show/Find/Give me medicines/Give me information" with criteria → filter  
    - If query seeks understanding/explanation of medicine properties → semantic_search
    - If query seeks to discover/retrieve multiple items or structured data → filter
    - If query mentions specific themes like "Product Efficacy", "Safety Profile", "Market Performance" → filter
    - If query asks for information "based on" criteria → filter

    User Query: "{user_query}"

    Respond with only one word: either "semantic_search" or "filter"
    """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            
            intent = response.choices[0].message.content.strip().lower()
            
            # Validate intent
            if intent not in ["semantic_search", "filter"]:
                intent = "semantic_search"
                print(f"⚠️ Invalid intent detected: {intent}. Defaulting to semantic_search")
                return "semantic_search"
                intent = "semantic_search"
                
            print(f"🎯 Intent detected: {intent}")
            return intent
            
        except Exception as e:
            print(f"❌ Error in intent detection: {e}")
            return "semantic_search"  # Default fallback
    
    def extract_filters_with_llm(self, user_query: str) -> Dict[str, str]:
        """
        Extract filter criteria from user query using GPT-4
        
        Args:
            user_query: The user's input query
            
        Returns:
            Dictionary with filter criteria
        """
        prompt = f"""
Extract filter criteria from the user query for a pharmaceutical database.

The database has these fields:
- product
- theme: Categories like "Product Efficacy", "Safety Profile", "Market Performance", etc.
- insight: Detailed reasons/descriptions related to the theme
- region: City or region (e.g., Mumbai, Navi Mumbai, Delhi, etc.)


Given the user query, extract the relevant filter criteria and return a JSON object.
If no specific criteria is mentioned, return an empty JSON object {{}}.

Examples:
- "Give me the product efficacy based information" → {{"theme": "Product Efficacy"}}
- "Show me medicines with safety concerns" → {{"theme": "Safety Profile"}}
- "Find medicines with high performance ratings" → {{"theme": "Market Performance"}}
- "List medicines where tablets were appreciated for fast action" → {{"insight": "fast action"}}
- "Product performance and safety concerns for Virilix and Ospaan D"  → {{"product": ["Virilix", "Ospaan D"], "theme": "Product Performance", "insight": "safety concerns"}}
- "Talk to me about the medicines from Navi Mumbai" → {{"region": "Navi Mumbai"}}

  
Important: 
- Only include fields that are clearly mentioned in the query
- Use exact matching terms when possible
- If unsure, return empty object {{}}

User Query: "{user_query}"

Return only a valid JSON object:
"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0
            )
            
            filter_text = response.choices[0].message.content.strip()
            filter_text = re.sub(r"```json|```", "", filter_text).strip()
            
            # Try to parse the JSON response
            try:
                filters = json.loads(filter_text)
                print(f" Extracted filters: {filters}")
                return filters
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse filter JSON: {filter_text}")
                return {}
                
        except Exception as e:
            print(f"❌ Error in filter extraction: {e}")
            return {}
    
    # def get_embedding(self, text: str) -> List[float]:
    #     """
    #     Get text embedding using OpenAI's text-embedding-ada-002
        
    #     Args:
    #         text: Text to embed
            
    #     Returns:
    #         List of embedding values
    #     """
    #     # try:
    #     #     response = openai.embeddings.create(
    #     #         # model="text-embedding-ada-002",
    #     #         model="text-embedding-3-small",
    #     #         input=text,
                
    #     #     )
    #     try:
    #         model = SentenceTransformer(
    #             "jinaai/jina-embeddings-v2-base-en", # switch to en/zh for English or Chinese
    #             trust_remote_code=True
    #         )
    #         return model.encode(text).tolist()
    #     except Exception as e:
    #         print(f"Error getting embedding: {str(e)}")
    #         return []
    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        """
        Generate embeddings using Gemini's embedding-001 model.

        Args:
            text (str): The input text to embed.
            mode (str): 'query' or 'document' based on use case.

        Returns:
            List[float]: 768-dimensional embedding vector.
        """
        try:
            task = "RETRIEVAL_QUERY" if mode == "query" else "RETRIEVAL_DOCUMENT"
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type=task
            )
            return response["embedding"]
        except Exception as e:
            print(f"❌ Gemini embedding error: {e}")
            return []

    
    def semantic_search(self, user_query: str, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity
        
        Args:
            user_query: The user's query
            limit: Maximum number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Get embedding for the query
            query_vector = self.get_embedding(user_query)
            if not query_vector:
                return []
            
            # Perform vector search
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for hit in search_results:
                results.append({
                    'id': hit.id,
                    'score': hit.score,
                    'product': hit.payload.get('product', ''),
                    'theme': hit.payload.get('theme', ''),
                    'insight': hit.payload.get('insight', ''),
                    'region': hit.payload.get('region', ''),
                    'payload': hit.payload
                })
            
            print(f" Found {len(results)} semantic search results")
            return results
            
        except Exception as e:
            print(f"❌ Error in semantic search: {e}")
            return []
    
    def filter_search(self, filters: Dict[str, str], limit: int = 50) -> List[Dict[str, Any]]:
        """
        Perform filtered search using Qdrant payload filtering
        
        Args:
            filters: Dictionary containing filter criteria
            limit: Maximum number of results to return
            
        Returns:
            List of filtered results
        """
       

        try:
            if not filters:
                print("⚠️ No filters provided")
                return []

            # 🔍 First try: semantic filter search (only region, product, theme used as structured filters)
            query_text = filters.get("theme") or filters.get("insight", "")
            results = self.semantic_filter_search(query_text, filters, limit=limit)

            # 🔁 Fallback: if semantic-filter search yields nothing, do plain semantic search
            if not results:
                print("⚠️ No results found in semantic filter search — falling back to semantic search")
                # fallback_query = filters.get("insight") or filters.get("theme") or "pharma"
                # results = self.semantic_search(fallback_query)
                results = self.semantic_search(" ".join(f"{k}: {v}" for k, v in filters.items()) or "pharma")


            return results

        except Exception as e:
            print(f"❌ Error in filter search: {e}")
            return []


    
    def semantic_filter_search(self, text: str, filters: Dict[str, str], limit: int = 50) -> List[Dict[str, Any]]:
        """
        Perform semantic vector search using a theme or insight embedding,
        combined with structured payload filters like region or product.
        """
        try:
            embedding = self.get_embedding(text)
            if not embedding:
                return []

            # Structured filters (region, product)
            must_conditions = []


            for key in ["region", "product", "theme"]:
                if key in filters:
                    values = filters[key]
                    if isinstance(values, str):
                        values = [values]
                    for value in values:
                        must_conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )

            print(f"📊 Query vector search with structured filters: {[c.key for c in must_conditions]}")
            print(f"📥 Incoming filters: {filters}")

            search_filter = Filter(must=must_conditions) if must_conditions else None

            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                query_filter=search_filter
            )

            results = []
            for hit in search_results:
                results.append({
                    'id': hit.id,
                    'score': hit.score,
                    'product': hit.payload.get('product', ''),
                    'theme': hit.payload.get('theme', ''),
                    'insight': hit.payload.get('insight', ''),
                    'region': hit.payload.get('region', ''),
                    'payload': hit.payload
                })

            print(f" Found {len(results)} semantic-filtered results for: {text}")
            return results

        except Exception as e:
            print(f"❌ Error in semantic_filter_search: {e}")
            return []



    def generate_natural_response(self, user_query: str, results: List[Dict[str, Any]], intent: str, filters: Dict[str, str] = None, history: List[Dict[str, str]] = None) -> str:

        """
        Generate a natural language response based on search results
        """
        try:

            if not results:
                hint = "Try refining your query by:"
                if intent == "filter":
                    hint += "\n- Including a specific medicine name\n- Narrowing down the theme (e.g., 'Product Efficacy', 'Safety Profile')\n- Adding a region if relevant"
                else:
                    hint += "\n- Asking about a specific product or effect\n- Rephrasing your question for clarity"

                response_text = f" I couldn't find any relevant insights for:\n\"{user_query}\"\n\n{hint}"
                logging.info(f"Natural Response Generated:\nUser Query: {user_query}\nResponse: {response_text}\nTimestamp: {datetime.now().isoformat()}\n{'-'*50}")
                if all(not r.get("insight") for r in results):
                    response_text += "\n\n🟡 Note: The insights found were limited in detail. You may get better results by asking about a specific region, medicine, or category."
                return response_text
                        
            # Prepare context from results
            n = len(results)
            context_parts = []

            for i, result in enumerate(results[:5], 1):
                med = result.get('product', 'Unknown')
                theme = result.get('theme', '')
                reason = result.get('insight', '')
                region = result.get('region', 'unspecified region')
                
                part = f"{i}. {med}"
                if theme:
                    part += f" (theme: {theme})"
                if reason:
                    part += f" - {reason}"
                if region:
                    # part += f" [region: {region}]"
                    part += f" - This insight was reported in **{region}**."
                context_parts.append(part)

            context = "\n".join(context_parts)

            # Response strategy hint based on result density
            if n == 1:
                strategy_note = "Focus your response on this single medicine. Provide a clear summary of the insight and its relevance."
            elif n <= 5:
                strategy_note = "Compare the listed medicines and highlight differences or similarities in their theme insights."
            else:
                strategy_note = "Summarize patterns or trends from the results. Encourage narrowing down using product name, region, or theme."


            
            # Create prompt for natural response generation
            system_prompt = """You are a professional **Pharmaceutical Insights Analyst** working with sales, product, and marketing teams. Your task is to explain field insights derived from doctor interactions and translate them into business-relevant observations. **please note that you are not a medical expert, so do not provide medical advice.**Important** SHARE THE INSIGHTS WITHIN 100-150 WORDS.

🔹 OBJECTIVE:
Help business users (like product managers, sales heads, or marketers) understand the meaning and implications of the results retrieved from the pharma database.

🔹 GUIDELINES:
- Always stay grounded in the data. Never speculate or hallucinate.
- Structure your response in **natural, conversational paragraphs** — not bullet points.
- Use a **helpful, professional, and informative tone**.
- Adapt based on number of results:
  - **Single result**: Give a focused explanation, highlight what makes it noteworthy.
  - **Multiple results**: Compare across them (e.g., "Compared to X, Y showed...").
- If the results are vague or sparse, clearly mention that and suggest follow-up actions (e.g., “You might want to search by theme instead...”).
- Avoid jargon. Make it easy for pharma business stakeholders to act on what you say.
- When `intent` is "filter", highlight what criteria were used in the search (e.g., theme, reason, region).
"""

            context_history = ""
            if history:
                for turn in history:
                    context_history += f"\n👤 User: {turn['query']}\n🤖 Bot: {turn['response']}"


            user_prompt = f"""
            **Recent Conversation History:**{context_history}
💬 **User Query:**
"{user_query}"

📊 **Search Context:**
Below are the top search results from the pharmaceutical insights database. Each result includes the medicine name, the associated insight category (theme),  specific reason or finding (insight), and also if asked Include the region as well.

{context}

🎯 **Task for You:**
Using this context, generate a clear, business-friendly explanation that answers the user’s query.

📌 Guidelines:
- Use natural language and full sentences.
- Focus on insights that are most relevant to the user’s query.
- If results include multiple medicines, help the user compare or group them.
- If results are unclear, acknowledge it and guide the user on how to improve their search (e.g., suggest narrowing by product or theme).
- Do NOT add any extra information not present in the context above.
"""

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            user_prompt += f"\n\n🧠 Response Strategy Hint: {strategy_note}"
            
            natural_response = response.choices[0].message.content.strip()
            

            if intent == "filter" and isinstance(filters, dict):
                if filters:
                    filter_info = ", ".join([f"{k}: {v}" for k, v in filters.items()])
                    natural_response += f"\n\n(This search was filtered by: {filter_info})"
                if "region" in filters:
                    natural_response += f"\n\n Note: These results are specific to the region: {filters['region']}."


            # Log the natural response
            logging.info(f"Natural Response Generated:\n"
                        f"User Query: {user_query}\n"
                        f"Response: {natural_response}\n"
                        f"Timestamp: {datetime.now().isoformat()}\n"
                        f"{'-'*50}")
            
            print(f"💬 Generated natural response")
            return natural_response
            
        except Exception as e:
            # Log errors too
            error_response =  f"❌ I ran into a technical issue while generating your response. Please try again in a few seconds, or refine your query and resend it."
            logging.error(f"Error generating natural response:\n"
                         f"User Query: {user_query}\n"
                         f"Error: {str(e)}\n"
                         f"Timestamp: {datetime.now().isoformat()}\n"
                         f"{'-'*50}")
            
            print(f"❌ Error generating natural response: {e}")
            return error_response
    
    def handle_user_query(self, user_query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Main handler that routes queries based on detected intent and generates natural response
        
        Args:
            user_query: The user's input query
            
        Returns:
            Dictionary containing results, metadata, and natural language response
        """
        try:
            # Step 1: Detect intent
            intent = self.detect_intent(user_query)
            
            # Step 2: Route based on intent and get results
            if intent == "semantic_search":
                results = self.semantic_search(user_query)
                filters = None
            elif intent == "filter":
                filters = self.extract_filters_with_llm(user_query)
                results = self.filter_search(filters)
            else:
                results = []
                filters = None
            
            # Step 3: Generate natural language response
            natural_response = self.generate_natural_response(user_query, results, intent, filters, history=self.conversation_history.get(session_id, []))
            turn = {"query": user_query, "response": natural_response}
            self.conversation_history.setdefault(session_id, []).append(turn)
            self.conversation_history[session_id] = self.conversation_history[session_id][-self.max_history_turns:]

            
            return {
                'intent': intent,
                'query': user_query,
                'filters': filters,
                'results': results,
                'total_results': len(results),
                'search_type': 'semantic_search' if intent == 'semantic_search' else 'filter_search',
                'response': natural_response  # This is the main natural language response
            }
                
        except Exception as e:
            print(f"❌ Error handling query: {e}")
            return {
                'query': user_query,
                'results': [],
                'total_results': 0,
                'response': f"I'm sorry, I encountered an error while processing your query: {str(e)}",
                'error': str(e)
            }

# Initialize the chatbot
chatbot = PharmaChatbot()

# FastAPI app
app = FastAPI(title="Pharma Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dev-pharma-ai.valenceai.in/x"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    intent: Optional[str] = None
    query: str
    filters: Optional[Dict[str, str]] = None
    results: List[Dict[str, Any]]
    total_results: int
    search_type: Optional[str] = None
    response: str  # Natural language response
    error: Optional[str] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Pharma Chatbot API is running!", "status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process user query and return results with natural language response
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process the query using the chatbot
        result = chatbot.handle_user_query(request.query)
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Legacy endpoint for backward compatibility
@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """
    Legacy chat endpoint that returns just the natural language response
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process the query using the chatbot
        result = chatbot.handle_user_query(request.query)
        
        return {
            "response": result.get('response', 'I apologize, but I could not generate a response.'),
            "results": result.get('results', [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Extended health check with system status"""
    try:
        # Test Qdrant connection
        collections = chatbot.qdrant_client.get_collections()
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "qdrant_status": qdrant_status,
        "collection_name": chatbot.collection_name
    }
@app.get("/history")
async def get_history(session_id: str = "default"):
    """
    Get conversation history for a given session_id
    """
    history = chatbot.conversation_history.get(session_id, [])
    return {"session_id": session_id, "history": history}
    
@app.post("/supervisor")
async def handle_with_supervisor(request: QueryRequest):
    graph = build_pharma_graph()
    result = graph.invoke({"query": request.query})
    return result


# --- State Definition ---
class ChatState(TypedDict):
    query: str
    intent: Optional[Literal["semantic_search", "filter"]]
    filters: Optional[Dict[str, str]]
    results: Optional[List[Dict]]
    response: Optional[str]

# --- Node 1: Detect Intent ---
def detect_intent_node(state: ChatState) -> ChatState:
    intent = chatbot.detect_intent(state["query"])
    return {**state, "intent": intent}

# --- Node 2: Extract Filters (only for filter intent) ---
def extract_filters_node(state: ChatState) -> ChatState:
    filters = chatbot.extract_filters_with_llm(state["query"])
    return {**state, "filters": filters}

# --- Node 3a: Semantic Search Agent ---
def semantic_search_node(state: ChatState) -> ChatState:
    results = chatbot.semantic_search(state["query"])
    return {**state, "results": results}

# --- Node 3b: Filtered Search Agent ---
def filter_search_node(state: ChatState) -> ChatState:
    results = chatbot.filter_search(state.get("filters", {}))
    return {**state, "results": results}

# --- Node 4: Natural Language Generation Agent ---
def natural_response_node(state: ChatState) -> ChatState:
    response = chatbot.generate_natural_response(
        state["query"],
        state["results"],
        state["intent"],
        filters=state.get("filters")
    )
    return {**state, "response": response}
def should_extract_filters(state: ChatState) -> str:
    return "extract_filters" if state["intent"] == "filter" else "semantic_search"

def build_pharma_graph():
    graph = StateGraph(ChatState)

    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("extract_filters", extract_filters_node)
    graph.add_node("semantic_search", semantic_search_node)
    graph.add_node("filter_search", filter_search_node)
    graph.add_node("generate_response", natural_response_node)

    # Flow
    graph.set_entry_point("detect_intent")
    graph.add_conditional_edges("detect_intent", should_extract_filters, {
        "extract_filters": "extract_filters",
        "semantic_search": "semantic_search"
    })
    graph.add_edge("extract_filters", "filter_search")
    graph.add_edge("semantic_search", "generate_response")
    graph.add_edge("filter_search", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()



if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "chat:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )



    # # To run the server, use the command:
    # uvicorn chat:app --host 0.0.0.0 --port 8003 --reload --log-level info
