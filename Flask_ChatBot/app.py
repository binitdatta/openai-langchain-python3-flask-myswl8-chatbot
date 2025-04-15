from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from langchain import PromptTemplate, OpenAI
import requests
import re
import json
from handlers.chat_model_start_handler import ChatModelStartHandler
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Initialize LangChain + OpenAI with handler
handler = ChatModelStartHandler()
llm = OpenAI(callbacks=[handler], model_name="gpt-4")

# Templates for LangChain
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

Leave any field blank if the user hasn’t provided it. Do not guess values. The order_status fiels is an exception. If not present, set default value as accepted

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

# cancel_order_template = """
# Extract the order id from the following message and format it as a JSON object and use underscore to separate words: {user_message}
# Unless the message contains the word cancel, return not a cancel
# """

# cancel_order_template = """
# You are a helpful assistant.
#
# Your task is to determine if the user is trying to cancel an order.
#
# Instructions:
# - If the message includes a clear cancellation intent (e.g. 'cancel', 'terminate', 'remove my order'), extract the order id from the user_message and return it as JSON like: {"order_id": 123}
# - If there's no clear cancellation intent, respond with: not a cancel
#
# User message: {user_message}
# """

cancel_order_template = """
You are a helpful assistant. When a user wants to cancel an existing order, extract the following:
- order_id

Return a JSON object with the fields provided. If the message is not an update request, reply with: "not a cancel".

Message: {user_message}
"""


cancel_prompt = PromptTemplate(
    input_variables=["user_message"],
    template=cancel_order_template
)

update_prompt = PromptTemplate(
    input_variables=["user_message"],
    template=order_update_template
)

# Prompt objects
status_prompt = PromptTemplate(
    input_variables=["order_id", "order_json"],
    template=order_status_template
)

creation_prompt = PromptTemplate(
    input_variables=["user_message"],
    template=order_creation_template
)

# === UTILITIES ===

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

# === ROUTES ===

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    print(data)

    if not user_message:
        return jsonify({"response": "❌ Please provide a message."})

    # 1. Handle cancellation FIRST
    is_cancelled, cancel_response = try_cancel_order_from_text(user_message)
    if is_cancelled or cancel_response != "not a cancel":
        return jsonify({"response": cancel_response})

    # 2. Check for update intent
    is_updated, update_response = try_update_order_from_text(user_message)
    if is_updated or update_response != "not an update":
        return jsonify({"response": update_response})

    # 3. Check for creation intent
    is_created, create_response = try_create_order_from_text(user_message)
    if is_created or create_response != "not an order":
        return jsonify({"response": create_response})

    # 4. Fallback to status query
    order_id = extract_order_id(user_message)
    print(order_id)
    if not order_id:
        return jsonify({"response": "❌ Could not find an order ID in your message."})

    order_json = get_order_status(order_id)
    if "error" in order_json:
        return jsonify({"response": order_json["error"]})

    langchain_prompt = status_prompt.format(order_id=order_id, order_json=str(order_json))
    chatbot_response = llm(langchain_prompt)

    return jsonify({"response": chatbot_response})

@app.route('/home', methods=['GET'])
def serve_home_html():
    return render_template('index.html')

@app.route('/elg', methods=['GET'])
def serve_html():
    return render_template('elegant_index.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# === MAIN ===

if __name__ == '__main__':
    app.run(debug=True, port=5005)
