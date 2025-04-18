from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from langchain import PromptTemplate, OpenAI
import requests
import re
import json
from handlers.chat_model_start_handler import ChatModelStartHandler
from datetime import datetime

app = Flask(__name__)
CORS(app)

# === LLM Setup ===
handler = ChatModelStartHandler()
llm = OpenAI(callbacks=[handler], model_name="gpt-4")

architecture_sections = [
    {
        "sequence": "1.",
        "bgColor": "primary",
        "description": "ChatBot User navigates to the ChatBot Website http://localhost:5005"
    },
    {
        "sequence": "2.",
        "bgColor": "primary",
        "description": "ChatBot User enters his/her Question / Message and submits the ChatBot Form"
    },
    {
        "sequence": "3.",
        "bgColor": "primary",
        "description": "ChatBot Backend Flask API (POST) invokes OpenAI API and submits a PromptTemplate to parse the user's message into json"
    },
    {
        "sequence": "4.",
        "bgColor": "primary",
        "description": "OpenAI API returns the parsed json to the ChatBot Backend API"
    },
    {
        "sequence": "5.",
        "bgColor": "primary",
        "description": "ChatBot Backend parses the JSON and invokes the appropriate Flask REST API"
    },
    {
        "sequence": "6.",
        "bgColor": "primary",
        "description": "Flask REST API executes the appropriate SQL Query against the SQL Database"
    },
    {
        "sequence": "7.",
        "bgColor": "primary",
        "description": "SQL Operation Returns"
    },
    {
        "sequence": "8.",
        "bgColor": "primary",
        "description": "Flask REST API rerturns to the Flask Chatbot Backend API"
    },
    {
        "sequence": "9.",
        "bgColor": "primary",
        "description": "Flask Chatbot Backend API renders the ChatBot Front end with Chat Response"
    },
    {
        "sequence": "10.",
        "bgColor": "primary",
        "description": " Chatbot User sees the response."
    },
]

self_attention_sections = [
    {
        "title": "What is Self-Attention?",
        "bgColor": "primary",
        "description": "Self-attention is a mechanism that allows a model to weigh different words (tokens) in a sentence based on their relationships, regardless of their distance. It is key to models like BERT, GPT-4, and T5."
    },
    {
        "title": "How Self-Attention Works",
        "bgColor": "success",
        "description": "Each word in a sentence is converted into Query, Key, and Value matrices. A dot-product similarity computes attention scores, which weight the words dynamically for better context understanding."
    },
    {
        "title": "When Was Self-Attention Invented?",
        "bgColor": "danger",
        "description": "Self-attention was introduced in the Transformer model in June 2017 in the paper 'Attention Is All You Need', presented at NeurIPS 2017."
    },
    {
        "title": "Which Organization Developed It?",
        "bgColor": "warning",
        "description": "The Transformer model, which introduced self-attention, was developed by Google Brain, an AI research division of Google Research."
    },
    {
        "title": "Who Were the People Behind It?",
        "bgColor": "info",
        "description": "The paper was authored by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin."
    },
    {
        "title": "Why Is Self-Attention Important?",
        "bgColor": "dark",
        "description": "Self-attention enables efficient long-sequence handling, better word relationship modeling, and faster training. It powers Generative AI models like ChatGPT, BERT, and DALL·E."
    },
    {
        "title": "Real-Life Example: ChatGPT Understanding a User Question",
        "bgColor": "primary",
        "description": "If you ask: 'Can you tell me who wrote the Transformer paper and why it’s important?', self-attention ensures that words like 'who' and 'wrote' are linked to focus on people, and 'Transformer paper' is connected to 'important' to generate an accurate response."
    }
]

sections = [
    {
        "title": "Mathematics in NLP",
        "bgColor": "primary",
        "topics": [
            {"subTitle": "Linear Algebra",
             "description": "Used for word embeddings like Word2Vec, GloVe, and Transformers."},
            {"subTitle": "Probability & Statistics",
             "description": "Helps estimate the likelihood of words appearing together (Markov Chains, Hidden Markov Models)."},
            {"subTitle": "Optimization & Calculus",
             "description": "Gradient descent and backpropagation are used to train deep learning models."}
        ]
    },
    {
        "title": "Statistics in NLP",
        "bgColor": "success",
        "topics": [
            {"subTitle": "N-Gram Models",
             "description": "Predicts word sequences using frequency-based probabilities."},
            {"subTitle": "Bayesian Inference",
             "description": "Used in Naïve Bayes for spam filtering and sentiment analysis."},
            {"subTitle": "Latent Semantic Analysis",
             "description": "Uses Singular Value Decomposition (SVD) to detect hidden word relationships."}
        ]
    },
    {
        "title": "Linguistics in NLP",
        "bgColor": "danger",
        "topics": [
            {"subTitle": "Syntax & Grammar",
             "description": "Analyzes sentence structure using context-free grammars (CFGs)."},
            {"subTitle": "Morphology & Semantics",
             "description": "Lemmatization and stemming improve text understanding."},
            {"subTitle": "Phonetics & Speech Processing",
             "description": "Essential for speech recognition and text-to-speech synthesis."}
        ]
    },
    {
        "title": "History of NLP",
        "bgColor": "warning",
        "topics": [
            {"subTitle": "1950s-60s: Rule-Based NLP",
             "description": "Alan Turing's test and early machine translation systems."},
            {"subTitle": "1970s-90s: Statistical NLP",
             "description": "Hidden Markov Models (HMMs) and Statistical Machine Translation (SMT)."},
            {"subTitle": "2000s-Present: Deep Learning NLP",
             "description": "Word embeddings, transformers (BERT, GPT), and modern AI chatbots."}
        ]
    },
    {
        "title": "Generative AI & NLP",
        "bgColor": "info",
        "topics": [
            {"subTitle": "Language Models",
             "description": "GPT-4, ChatGPT, and other transformer-based models for text generation."},
            {"subTitle": "Pretrained Language Models (PLMs)",
             "description": "BERT, T5, and GPT trained on massive datasets for contextual understanding."},
            {"subTitle": "Multimodal AI",
             "description": "Combining text, images, and audio for AI-generated content (e.g., DALL\u00b7E)."}
        ]
    }
]

timeline = [
    {
        "year": "1950s-1970s",
        "title": "Early AI & Statistical Learning",
        "description": "The foundations of AI with rule-based systems, Bayesian inference, and the first neural networks.",
        "details": [
            "🔹 1950s: Foundations of Statistical Learning.",
            "🔹 1956: First AI programs (ELIZA, SHRDLU).",
            "🔹 1958: Perceptron by Frank Rosenblatt."
        ]
    },
    {
        "year": "1980s",
        "title": "Neural Networks & Machine Learning Foundations",
        "description": "The resurgence of neural networks with backpropagation and associative memory models.",
        "details": [
            "🔹 1982: Hopfield Networks for associative memory.",
            "🔹 1986: Backpropagation algorithm for multi-layer perceptrons."
        ]
    },
    {
        "year": "1990s",
        "title": "Statistical AI & Deep Learning Precursors",
        "description": "Introduction of CNNs and Reinforcement Learning.",
        "details": [
            "🔹 1992: Q-Learning and Reinforcement Learning (Sutton & Barto).",
            "🔹 1998: LeNet-5 (First Convolutional Neural Network)."
        ]
    },
    {
        "year": "2000s",
        "title": "Deep Learning Foundations",
        "description": "Breakthroughs in feature learning, convex optimization, and deep networks.",
        "details": [
            "🔹 2006: Hinton\u2019s Deep Belief Networks.",
            "🔹 2000s: Support Vector Machines (SVMs) & Kernel Methods."
        ]
    },
    {
        "year": "2010s",
        "title": "Deep Learning Boom",
        "description": "The rise of CNNs, Transformers, GANs, and reinforcement learning in deep AI models.",
        "details": [
            "🔹 2013: Variational Autoencoders (VAEs).",
            "🔹 2014: Generative Adversarial Networks (GANs) by Ian Goodfellow.",
            "🔹 2015: Residual Neural Networks (ResNet).",
            "🔹 2017: Transformer Networks ('Attention is All You Need')."
        ]
    },
    {
        "year": "2020s-Present",
        "title": "Generative AI Revolution",
        "description": "Large Language Models, Diffusion Models, and Multimodal AI.",
        "details": [
            "🔹 2020-Present: GPT-4, Claude, LLaMA, Gemini, Mistral.",
            "🔹 2020-Present: DALL\u00b7E, Stable Diffusion, Imagen.",
            "🔹 2020-Present: Self-Supervised Learning, Reinforcement Learning from Human Feedback (RLHF)."
        ]
    }
]

similarities = [
    {"title": "Neuron-like Units",
     "description": "Both use units that receive inputs, process them, and generate outputs."},
    {"title": "Connections and Layers",
     "description": "The brain has layers of neurons, just like neural networks have input, hidden, and output layers."},
    {"title": "Learning and Adaptation",
     "description": "The brain strengthens connections based on experience; ANNs adjust weights via backpropagation."},
    {"title": "Parallel Processing", "description": "Both process multiple pieces of information at the same time."},
    {"title": "Pattern Recognition", "description": "Both recognize patterns like images and speech."}
]

differences = [
    {"title": "Energy Efficiency",
     "description": "The brain uses only ~20W, while ANNs require high computational power."},
    {"title": "Learning Mechanism", "description": "The brain learns continuously; ANNs need retraining."},
    {"title": "Memory Storage", "description": "The brain stores memories flexibly, while ANNs need large datasets."},
    {"title": "Generalization",
     "description": "Humans generalize from few examples; ANNs struggle without large training sets."},
    {"title": "Chemical Processing", "description": "The brain uses neurotransmitters; ANNs rely purely on math."},
    {"title": "Robustness", "description": "The brain heals from damage, but ANNs can fail with minor errors."}
]

# === Prompt Templates ===
intent_classifier_template = """
You are a classifier. Given the user's message, identify the intent.
Possible intents are: cancel, update, create, status.
Respond with only one word.

Message: {user_message}
"""

order_status_template = """
You are a helpful chatbot. A user asked about the status of an order. Here is the order information:

Order ID: {order_id}
Order Status JSON: {order_json}

Answer the user's question based on the order status.
"""

order_creation_template = """
You are a helpful assistant. When a user sends a message, check if it's an intent to place an order.

If so, extract and return these fields as a JSON object:
- order_no
- order_date (format as YYYY-MM-DD, use today's date if needed)
- product_id
- qty_ordered (integer)
- unit_price (number only)
- customer_id
- order_status

Leave any field blank if the user hasn’t provided it. Do not guess values. The order_status field is an exception. If not present, set default value as accepted

If the message is not about placing an order, respond with: "not an order".

Message: {user_message}
"""

order_update_template = """
You are a helpful assistant. When a user wants to update an existing order, extract the following:
- order_id
- Any of the following fields if mentioned: product_id, qty_ordered, unit_price

Return a JSON object with the fields provided. If the message is not an update request, reply with: "not an update".

Message: {user_message}
"""

cancel_order_template = """
You are a helpful assistant. When a user wants to cancel an existing order, extract the following:
- order_id

Return a JSON object with the fields provided. If the message is not a cancel request, reply with: "not a cancel".

Message: {user_message}
"""


def extract_order_id(user_message):
    match = re.search(r'\b\d+\b', user_message)
    return match.group(0) if match else None


def try_cancel_order_from_text(user_message):
    print("🔍 Entered try_cancel_order_from_text()")
    print("🔍 user_message:", user_message)

    try:
        if "cancel" not in user_message.lower():
            return False, "not a cancel"
        # Debugging the cancel prompt
        print("✅ cancel_prompt template:", cancel_prompt)

        # Check if the formatting works correctly
        try:
            print("🔍 user_message before cancel prompt:", user_message)
            prompt = cancel_prompt.format(user_message=user_message)
            print("📝 Final prompt to LLM:", prompt)
        except KeyError as ke:
            print(f"❌ KeyError in formatting: {ke}")
            return False, f"❌ KeyError in formatting: {ke}"

        try:
            raw_output = llm(prompt)
            if hasattr(raw_output, 'content'):
                raw_response = raw_output.content
            elif hasattr(raw_output, 'text'):
                raw_response = raw_output.text
            else:
                raw_response = str(raw_output)
        except Exception as e:
            print("❌ LLM call failed:", repr(e))
            raw_response = str(e)

        print("🧠 Raw LLM response:", repr(raw_response))

        if "not a cancel" in raw_response.lower():
            return False, "not a cancel"

        cleaned = raw_response.strip().strip("`").strip()
        print("🧪 Cleaned response:", cleaned)

        # Fix broken output: add {} if missing
        if cleaned.startswith('"order_id"') or cleaned.startswith("'order_id'"):
            print("⚠️ Wrapping key in braces...")
            cleaned = "{" + cleaned + "}"

        cleaned = cleaned.replace("'", '"')

        print("🧪 Final cleaned JSON string:", cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("❌ JSON decode error:", str(e))
            return False, f"❌ Could not parse JSON: {str(e)}"

        print("✅ Parsed response:", parsed)

        # Check if the 'order_id' exists in the parsed response
        if not isinstance(parsed, dict):
            return False, f"❌ LLM response was not a JSON object. It was a {type(parsed).__name__}."

        if "order_id" not in parsed:
            return False, "❌ 'order_id' not found in parsed JSON."

        order_id = parsed["order_id"]

        cancel_payload = {"order_status": "cancelled"}
        print(f"📤 PUT /orders/{order_id} with payload: {cancel_payload}")

        cancel_response = requests.put(
            f"http://localhost:5004/orders/{order_id}",
            headers={"Content-Type": "application/json"},
            json=cancel_payload
        )

        if cancel_response.ok:
            return True, f"✅ Order {order_id} has been cancelled."
        else:
            print("❌ Backend error:", cancel_response.text)
            return False, f"❌ Failed to cancel order: {cancel_response.text}"

    except Exception as e:
        print("❌ UNCAUGHT ERROR in try_cancel_order_from_text:", repr(e))
        return False, f"❌ Unexpected error: {str(e)}"


def get_order_status(order_id):
    api_url = f"http://localhost:5004/orders/{order_id}"
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    }

    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Unable to fetch order status. Status code: {response.status_code}"}


def try_update_order_from_text(user_message):
    try:
        # 🔍 Ask LLM for update JSON
        raw_response = llm(update_prompt.format(user_message=user_message))
        print("🔍 Update raw response from LLM:", raw_response)

        # 🛡️ Handle unexpected response formats
        if isinstance(raw_response, str):
            if "not an update" in raw_response.lower():
                return False, "not an update"
            try:
                update_data = json.loads(raw_response)
            except json.JSONDecodeError:
                return False, "❌ I couldn't parse the update info."
        elif isinstance(raw_response, dict):
            update_data = raw_response
        else:
            return False, "❌ Unexpected format from LLM."

        # ✅ Extract and validate order_id
        order_id = update_data.get("order_id")
        if not order_id:
            return False, "❌ Please specify the order ID to update."

        # ✅ Prepare payload with only allowed updatable fields
        update_payload = {}
        for field in ["product_id", "qty_ordered", "unit_price"]:
            value = update_data.get(field)
            if value:
                if field == "unit_price":
                    # Clean price input
                    value = value.replace("$", "").strip() if isinstance(value, str) else value
                    try:
                        update_payload[field] = float(value)
                    except ValueError:
                        return False, f"❌ Invalid {field} value."
                elif field == "qty_ordered":
                    try:
                        update_payload[field] = int(value)
                    except ValueError:
                        return False, "❌ Quantity must be a number."
                else:
                    update_payload[field] = value

        if not update_payload:
            return False, "⚠️ No updatable fields found in the message."

        # 🚀 Send update to backend
        update_response = requests.put(
            f"http://localhost:5004/orders/{order_id}",
            headers={"Content-Type": "application/json"},
            json=update_payload
        )

        if update_response.ok:
            return True, "✅ Order updated successfully."
        else:
            return False, f"❌ Update failed: {update_response.text}"

    except Exception as e:
        return False, f"❌ Error updating order: {str(e)}"


def try_create_order_from_text(user_message):
    try:
        raw_response = llm(creation_prompt.format(user_message=user_message))

        # 🛡️ Attempt to parse response as JSON safely
        try:
            order_data = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
        except json.JSONDecodeError:
            if "not an order" in raw_response.lower():
                return False, "not an order"
            else:
                return False, "❌ I couldn't parse the order info. Please rephrase or provide more details."

        response = llm(creation_prompt.format(user_message=user_message))
        print(response)
        # Parse JSON
        order_data = json.loads(response) if isinstance(response, str) else response

        if isinstance(order_data, dict):
            if "not an order" in json.dumps(order_data).lower():
                return False, "not an order"

            # ✅ Step 1: Required fields
            required_fields = [
                "order_no",
                "order_date",
                "product_id",
                "qty_ordered",
                "unit_price",
                "customer_id",
                "order_status"
            ]

            # ✅ Step 2: Find missing or empty fields
            missing_fields = [
                field for field in required_fields
                if field not in order_data or not str(order_data[field]).strip()
            ]

            if missing_fields:
                field_list = ", ".join(missing_fields)
                return False, f"⚠️ I need the following details to create the order: {field_list}"

            # ✅ Step 3: Clean and format fields

            # Clean order_date
            date_str = order_data["order_date"].lower().strip()
            print(date_str)
            if "today" in date_str:
                order_data["order_date"] = datetime.today().strftime('%Y-%m-%d')
            else:
                try:
                    parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
                    order_data["order_date"] = parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    return False, "❌ Invalid date format. Please use YYYY-MM-DD."

            # Clean unit_price
            price = order_data["unit_price"]
            if isinstance(price, str):
                price = price.replace("$", "").strip()
            try:
                order_data["unit_price"] = float(price)
            except ValueError:
                return False, "❌ Invalid unit price. Please provide a number."

            # Clean qty_ordered
            try:
                order_data["qty_ordered"] = int(order_data["qty_ordered"])
            except ValueError:
                return False, "❌ Invalid quantity. Please provide a number."

            # ✅ Step 4: Send to backend
            create_response = requests.post(
                "http://localhost:5004/orders",
                headers={"Content-Type": "application/json"},
                json=order_data
            )

            if create_response.ok:
                return True, "✅ Order created successfully!"
            else:
                return False, f"❌ Order creation failed: {create_response.text}"

        else:
            return False, "❌ I couldn’t understand your request."
    except Exception as e:
        return False, f"❌ Error processing order: {str(e)}"


# === PromptTemplate objects ===
intent_prompt = PromptTemplate(input_variables=["user_message"], template=intent_classifier_template)
cancel_prompt = PromptTemplate(input_variables=["user_message"], template=cancel_order_template)
update_prompt = PromptTemplate(input_variables=["user_message"], template=order_update_template)
creation_prompt = PromptTemplate(input_variables=["user_message"], template=order_creation_template)
status_prompt = PromptTemplate(input_variables=["order_id", "order_json"], template=order_status_template)







# === Intent Router ===
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "❌ Please provide a message."})

    try:
        intent = llm(intent_prompt.format(user_message=user_message)).strip().lower()
        print(f"🧭 Detected intent: {intent}")

        if intent == "cancel":
            is_cancelled, cancel_response = try_cancel_order_from_text(user_message)
            return jsonify({"response": cancel_response})

        elif intent == "update":
            is_updated, update_response = try_update_order_from_text(user_message)
            return jsonify({"response": update_response})

        elif intent == "create":
            is_created, create_response = try_create_order_from_text(user_message)
            return jsonify({"response": create_response})

        elif intent == "status":
            order_id = extract_order_id(user_message)
            if not order_id:
                return jsonify({"response": "❌ Could not find an order ID in your message."})

            order_json = get_order_status(order_id)
            if "error" in order_json:
                return jsonify({"response": order_json["error"]})

            langchain_prompt = status_prompt.format(order_id=order_id, order_json=str(order_json))
            chatbot_response = llm(langchain_prompt)
            return jsonify({"response": chatbot_response})

        else:
            return jsonify({"response": "❌ I couldn’t determine what you want to do. Please try rephrasing."})

    except Exception as e:
        print("❌ Uncaught error in /chat:", repr(e))
        return jsonify({"response": f"❌ An unexpected error occurred: {str(e)}"})


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/home')
def serve_home_html():
    return render_template('index.html')


# gen-ai-timeline
@app.route('/gen-ai-timeline')
def serve_genai_timeline():
    return render_template("gen-ai-timeline.html", timeline=timeline)


@app.route('/nlp-fundamentals')
def nlp_fundamentals():
    return render_template("nlp-fundamentals.html", sections=sections)


@app.route('/neural-networks-and-human-brain')
def neural_network():
    return render_template("neural_network.html", similarities=similarities, differences=differences)


@app.route('/self-attention')
def self_attention():
    return render_template("self_attention.html", self_attention_sections=self_attention_sections)

@app.route('/architecture')
def architecture():
    return render_template("architecture.html", architecture_sections=architecture_sections)

@app.route('/elg')
def serve_html():
    return render_template('elegant_index.html')


@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')


# === LLM Setup ===

self_attention_sections = [
    {
        "title": "What is Self-Attention?",
        "bgColor": "primary",
        "description": "Self-attention is a mechanism that allows a model to weigh different words (tokens) in a sentence based on their relationships, regardless of their distance. It is key to models like BERT, GPT-4, and T5."
    },
    {
        "title": "How Self-Attention Works",
        "bgColor": "success",
        "description": "Each word in a sentence is converted into Query, Key, and Value matrices. A dot-product similarity computes attention scores, which weight the words dynamically for better context understanding."
    },
    {
        "title": "When Was Self-Attention Invented?",
        "bgColor": "danger",
        "description": "Self-attention was introduced in the Transformer model in June 2017 in the paper 'Attention Is All You Need', presented at NeurIPS 2017."
    },
    {
        "title": "Which Organization Developed It?",
        "bgColor": "warning",
        "description": "The Transformer model, which introduced self-attention, was developed by Google Brain, an AI research division of Google Research."
    },
    {
        "title": "Who Were the People Behind It?",
        "bgColor": "info",
        "description": "The paper was authored by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin."
    },
    {
        "title": "Why Is Self-Attention Important?",
        "bgColor": "dark",
        "description": "Self-attention enables efficient long-sequence handling, better word relationship modeling, and faster training. It powers Generative AI models like ChatGPT, BERT, and DALL·E."
    },
    {
        "title": "Real-Life Example: ChatGPT Understanding a User Question",
        "bgColor": "primary",
        "description": "If you ask: 'Can you tell me who wrote the Transformer paper and why it’s important?', self-attention ensures that words like 'who' and 'wrote' are linked to focus on people, and 'Transformer paper' is connected to 'important' to generate an accurate response."
    }
]

sections = [
    {
        "title": "Mathematics in NLP",
        "bgColor": "primary",
        "topics": [
            {"subTitle": "Linear Algebra",
             "description": "Used for word embeddings like Word2Vec, GloVe, and Transformers."},
            {"subTitle": "Probability & Statistics",
             "description": "Helps estimate the likelihood of words appearing together (Markov Chains, Hidden Markov Models)."},
            {"subTitle": "Optimization & Calculus",
             "description": "Gradient descent and backpropagation are used to train deep learning models."}
        ]
    },
    {
        "title": "Statistics in NLP",
        "bgColor": "success",
        "topics": [
            {"subTitle": "N-Gram Models",
             "description": "Predicts word sequences using frequency-based probabilities."},
            {"subTitle": "Bayesian Inference",
             "description": "Used in Naïve Bayes for spam filtering and sentiment analysis."},
            {"subTitle": "Latent Semantic Analysis",
             "description": "Uses Singular Value Decomposition (SVD) to detect hidden word relationships."}
        ]
    },
    {
        "title": "Linguistics in NLP",
        "bgColor": "danger",
        "topics": [
            {"subTitle": "Syntax & Grammar",
             "description": "Analyzes sentence structure using context-free grammars (CFGs)."},
            {"subTitle": "Morphology & Semantics",
             "description": "Lemmatization and stemming improve text understanding."},
            {"subTitle": "Phonetics & Speech Processing",
             "description": "Essential for speech recognition and text-to-speech synthesis."}
        ]
    },
    {
        "title": "History of NLP",
        "bgColor": "warning",
        "topics": [
            {"subTitle": "1950s-60s: Rule-Based NLP",
             "description": "Alan Turing's test and early machine translation systems."},
            {"subTitle": "1970s-90s: Statistical NLP",
             "description": "Hidden Markov Models (HMMs) and Statistical Machine Translation (SMT)."},
            {"subTitle": "2000s-Present: Deep Learning NLP",
             "description": "Word embeddings, transformers (BERT, GPT), and modern AI chatbots."}
        ]
    },
    {
        "title": "Generative AI & NLP",
        "bgColor": "info",
        "topics": [
            {"subTitle": "Language Models",
             "description": "GPT-4, ChatGPT, and other transformer-based models for text generation."},
            {"subTitle": "Pretrained Language Models (PLMs)",
             "description": "BERT, T5, and GPT trained on massive datasets for contextual understanding."},
            {"subTitle": "Multimodal AI",
             "description": "Combining text, images, and audio for AI-generated content (e.g., DALL\u00b7E)."}
        ]
    }
]

timeline = [
    {
        "year": "1950s-1970s",
        "title": "Early AI & Statistical Learning",
        "description": "The foundations of AI with rule-based systems, Bayesian inference, and the first neural networks.",
        "details": [
            "🔹 1950s: Foundations of Statistical Learning.",
            "🔹 1956: First AI programs (ELIZA, SHRDLU).",
            "🔹 1958: Perceptron by Frank Rosenblatt."
        ]
    },
    {
        "year": "1980s",
        "title": "Neural Networks & Machine Learning Foundations",
        "description": "The resurgence of neural networks with backpropagation and associative memory models.",
        "details": [
            "🔹 1982: Hopfield Networks for associative memory.",
            "🔹 1986: Backpropagation algorithm for multi-layer perceptrons."
        ]
    },
    {
        "year": "1990s",
        "title": "Statistical AI & Deep Learning Precursors",
        "description": "Introduction of CNNs and Reinforcement Learning.",
        "details": [
            "🔹 1992: Q-Learning and Reinforcement Learning (Sutton & Barto).",
            "🔹 1998: LeNet-5 (First Convolutional Neural Network)."
        ]
    },
    {
        "year": "2000s",
        "title": "Deep Learning Foundations",
        "description": "Breakthroughs in feature learning, convex optimization, and deep networks.",
        "details": [
            "🔹 2006: Hinton\u2019s Deep Belief Networks.",
            "🔹 2000s: Support Vector Machines (SVMs) & Kernel Methods."
        ]
    },
    {
        "year": "2010s",
        "title": "Deep Learning Boom",
        "description": "The rise of CNNs, Transformers, GANs, and reinforcement learning in deep AI models.",
        "details": [
            "🔹 2013: Variational Autoencoders (VAEs).",
            "🔹 2014: Generative Adversarial Networks (GANs) by Ian Goodfellow.",
            "🔹 2015: Residual Neural Networks (ResNet).",
            "🔹 2017: Transformer Networks ('Attention is All You Need')."
        ]
    },
    {
        "year": "2020s-Present",
        "title": "Generative AI Revolution",
        "description": "Large Language Models, Diffusion Models, and Multimodal AI.",
        "details": [
            "🔹 2020-Present: GPT-4, Claude, LLaMA, Gemini, Mistral.",
            "🔹 2020-Present: DALL\u00b7E, Stable Diffusion, Imagen.",
            "🔹 2020-Present: Self-Supervised Learning, Reinforcement Learning from Human Feedback (RLHF)."
        ]
    }
]

similarities = [
    {"title": "Neuron-like Units",
     "description": "Both use units that receive inputs, process them, and generate outputs."},
    {"title": "Connections and Layers",
     "description": "The brain has layers of neurons, just like neural networks have input, hidden, and output layers."},
    {"title": "Learning and Adaptation",
     "description": "The brain strengthens connections based on experience; ANNs adjust weights via backpropagation."},
    {"title": "Parallel Processing", "description": "Both process multiple pieces of information at the same time."},
    {"title": "Pattern Recognition", "description": "Both recognize patterns like images and speech."}
]

differences = [
    {"title": "Energy Efficiency",
     "description": "The brain uses only ~20W, while ANNs require high computational power."},
    {"title": "Learning Mechanism", "description": "The brain learns continuously; ANNs need retraining."},
    {"title": "Memory Storage", "description": "The brain stores memories flexibly, while ANNs need large datasets."},
    {"title": "Generalization",
     "description": "Humans generalize from few examples; ANNs struggle without large training sets."},
    {"title": "Chemical Processing", "description": "The brain uses neurotransmitters; ANNs rely purely on math."},
    {"title": "Robustness", "description": "The brain heals from damage, but ANNs can fail with minor errors."}
]

# === Prompt Templates ===

def extract_order_id(user_message):
    match = re.search(r'\b\d+\b', user_message)
    return match.group(0) if match else None


def try_cancel_order_from_text(user_message):
    print("🔍 Entered try_cancel_order_from_text()")
    print("🔍 user_message:", user_message)

    try:
        if "cancel" not in user_message.lower():
            return False, "not a cancel"
        # Debugging the cancel prompt
        print("✅ cancel_prompt template:", cancel_prompt)

        # Check if the formatting works correctly
        try:
            print("🔍 user_message before cancel prompt:", user_message)
            prompt = cancel_prompt.format(user_message=user_message)
            print("📝 Final prompt to LLM:", prompt)
        except KeyError as ke:
            print(f"❌ KeyError in formatting: {ke}")
            return False, f"❌ KeyError in formatting: {ke}"

        try:
            raw_output = llm(prompt)
            if hasattr(raw_output, 'content'):
                raw_response = raw_output.content
            elif hasattr(raw_output, 'text'):
                raw_response = raw_output.text
            else:
                raw_response = str(raw_output)
        except Exception as e:
            print("❌ LLM call failed:", repr(e))
            raw_response = str(e)

        print("🧠 Raw LLM response:", repr(raw_response))

        if "not a cancel" in raw_response.lower():
            return False, "not a cancel"

        cleaned = raw_response.strip().strip("`").strip()
        print("🧪 Cleaned response:", cleaned)

        # Fix broken output: add {} if missing
        if cleaned.startswith('"order_id"') or cleaned.startswith("'order_id'"):
            print("⚠️ Wrapping key in braces...")
            cleaned = "{" + cleaned + "}"

        cleaned = cleaned.replace("'", '"')

        print("🧪 Final cleaned JSON string:", cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("❌ JSON decode error:", str(e))
            return False, f"❌ Could not parse JSON: {str(e)}"

        print("✅ Parsed response:", parsed)

        # Check if the 'order_id' exists in the parsed response
        if not isinstance(parsed, dict):
            return False, f"❌ LLM response was not a JSON object. It was a {type(parsed).__name__}."

        if "order_id" not in parsed:
            return False, "❌ 'order_id' not found in parsed JSON."

        order_id = parsed["order_id"]

        cancel_payload = {"order_status": "cancelled"}
        print(f"📤 PUT /orders/{order_id} with payload: {cancel_payload}")

        cancel_response = requests.put(
            f"http://localhost:5004/orders/{order_id}",
            headers={"Content-Type": "application/json"},
            json=cancel_payload
        )

        if cancel_response.ok:
            return True, f"✅ Order {order_id} has been cancelled."
        else:
            print("❌ Backend error:", cancel_response.text)
            return False, f"❌ Failed to cancel order: {cancel_response.text}"

    except Exception as e:
        print("❌ UNCAUGHT ERROR in try_cancel_order_from_text:", repr(e))
        return False, f"❌ Unexpected error: {str(e)}"


def get_order_status(order_id):
    api_url = f"http://localhost:5004/orders/{order_id}"
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    }

    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Unable to fetch order status. Status code: {response.status_code}"}


def try_update_order_from_text(user_message):
    try:
        # 🔍 Ask LLM for update JSON
        raw_response = llm(update_prompt.format(user_message=user_message))
        print("🔍 Update raw response from LLM:", raw_response)

        # 🛡️ Handle unexpected response formats
        if isinstance(raw_response, str):
            if "not an update" in raw_response.lower():
                return False, "not an update"
            try:
                update_data = json.loads(raw_response)
            except json.JSONDecodeError:
                return False, "❌ I couldn't parse the update info."
        elif isinstance(raw_response, dict):
            update_data = raw_response
        else:
            return False, "❌ Unexpected format from LLM."

        # ✅ Extract and validate order_id
        order_id = update_data.get("order_id")
        if not order_id:
            return False, "❌ Please specify the order ID to update."

        # ✅ Prepare payload with only allowed updatable fields
        update_payload = {}
        for field in ["product_id", "qty_ordered", "unit_price"]:
            value = update_data.get(field)
            if value:
                if field == "unit_price":
                    # Clean price input
                    value = value.replace("$", "").strip() if isinstance(value, str) else value
                    try:
                        update_payload[field] = float(value)
                    except ValueError:
                        return False, f"❌ Invalid {field} value."
                elif field == "qty_ordered":
                    try:
                        update_payload[field] = int(value)
                    except ValueError:
                        return False, "❌ Quantity must be a number."
                else:
                    update_payload[field] = value

        if not update_payload:
            return False, "⚠️ No updatable fields found in the message."

        # 🚀 Send update to backend
        update_response = requests.put(
            f"http://localhost:5004/orders/{order_id}",
            headers={"Content-Type": "application/json"},
            json=update_payload
        )

        if update_response.ok:
            return True, "✅ Order updated successfully."
        else:
            return False, f"❌ Update failed: {update_response.text}"

    except Exception as e:
        return False, f"❌ Error updating order: {str(e)}"


def try_create_order_from_text(user_message):
    try:
        raw_response = llm(creation_prompt.format(user_message=user_message))

        # 🛡️ Attempt to parse response as JSON safely
        try:
            order_data = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
        except json.JSONDecodeError:
            if "not an order" in raw_response.lower():
                return False, "not an order"
            else:
                return False, "❌ I couldn't parse the order info. Please rephrase or provide more details."

        response = llm(creation_prompt.format(user_message=user_message))
        print(response)
        # Parse JSON
        order_data = json.loads(response) if isinstance(response, str) else response

        if isinstance(order_data, dict):
            if "not an order" in json.dumps(order_data).lower():
                return False, "not an order"

            # ✅ Step 1: Required fields
            required_fields = [
                "order_no",
                "order_date",
                "product_id",
                "qty_ordered",
                "unit_price",
                "customer_id",
                "order_status"
            ]

            # ✅ Step 2: Find missing or empty fields
            missing_fields = [
                field for field in required_fields
                if field not in order_data or not str(order_data[field]).strip()
            ]

            if missing_fields:
                field_list = ", ".join(missing_fields)
                return False, f"⚠️ I need the following details to create the order: {field_list}"

            # ✅ Step 3: Clean and format fields

            # Clean order_date
            date_str = order_data["order_date"].lower().strip()
            print(date_str)
            if "today" in date_str:
                order_data["order_date"] = datetime.today().strftime('%Y-%m-%d')
            else:
                try:
                    parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
                    order_data["order_date"] = parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    return False, "❌ Invalid date format. Please use YYYY-MM-DD."

            # Clean unit_price
            price = order_data["unit_price"]
            if isinstance(price, str):
                price = price.replace("$", "").strip()
            try:
                order_data["unit_price"] = float(price)
            except ValueError:
                return False, "❌ Invalid unit price. Please provide a number."

            # Clean qty_ordered
            try:
                order_data["qty_ordered"] = int(order_data["qty_ordered"])
            except ValueError:
                return False, "❌ Invalid quantity. Please provide a number."

            # ✅ Step 4: Send to backend
            create_response = requests.post(
                "http://localhost:5004/orders",
                headers={"Content-Type": "application/json"},
                json=order_data
            )

            if create_response.ok:
                return True, "✅ Order created successfully!"
            else:
                return False, f"❌ Order creation failed: {create_response.text}"

        else:
            return False, "❌ I couldn’t understand your request."
    except Exception as e:
        return False, f"❌ Error processing order: {str(e)}"


# === PromptTemplate objects ===
intent_prompt = PromptTemplate(input_variables=["user_message"], template=intent_classifier_template)
cancel_prompt = PromptTemplate(input_variables=["user_message"], template=cancel_order_template)
update_prompt = PromptTemplate(input_variables=["user_message"], template=order_update_template)
creation_prompt = PromptTemplate(input_variables=["user_message"], template=order_creation_template)
status_prompt = PromptTemplate(input_variables=["order_id", "order_json"], template=order_status_template)


# === Utilities ===
def extract_order_id(user_message):
    match = re.search(r'\b\d+\b', user_message)
    return match.group(0) if match else None


def get_order_status(order_id):
    api_url = f"http://localhost:5004/orders/{order_id}"
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    }
    response = requests.get(api_url, headers=headers)
    return response.json() if response.status_code == 200 else {
        "error": f"Unable to fetch order status. Status code: {response.status_code}"}


if __name__ == '__main__':
    app.run(debug=True, port=5005)

