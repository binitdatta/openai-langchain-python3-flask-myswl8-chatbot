# Sample Messages

` 
Please cancel order 2
What is the status of order 2
Hi, I’d like to place a new order. The order number is A123. I'm ordering 2 units of product 456 at $15.50 each for customer 789. Please process it today.
I want to cancel order 13
Please cancel order 13
Please update order 404: product to 678, quantity to 3, and price to $12.
`

Great — this is your **LangChain prompt template** for **order status inquiries**. Let's reverse-engineer it just like before.

---

## 🧠 Prompt Summary

This template is meant to guide GPT-4 to **answer questions like**:

> “What’s the status of order 1?”

---

### 🔁 Template Recap:
```jinja2
You are a helpful chatbot. A user asked about the status of an order. Here is the order information:

Order ID: {order_id}
Order Status JSON: {order_json}

Answer the user's question based on the order status.
```

LangChain will dynamically fill:
- `{order_id}` — e.g., `"1003"`
- `{order_json}` — the actual status data returned from your backend like:
```json
{
  "order_id": 1003,
  "order_status": "shipped",
  "product_id": "567",
  "qty_ordered": 2,
  "unit_price": 15.5,
  "customer_id": "2023",
  "order_date": "2025-04-07",
  "shipping_date": "2025-04-08",
  "tracking_no": "XYZ123"
}
```

---

## ✅ Example User Messages That Trigger This Prompt

These should **express an intent to check order status** (not cancel, not update, not create):

1. **Basic**  
> “What’s the status of order 1003?”

2. **Polite**  
> “Could you please let me know the current status of order number 12?”

3. **Detailed**  
> “I placed an order a few days ago. Can you tell me what’s happening with order ID 1003?”

4. **Casual**  
> “Hey, is order 12 shipped yet?”

5. **Vague with a number**  
> “Any update on 1003?”

As long as the intent is **to inquire**, and not **to change or cancel**, this template will be used.

---

## 🧪 Bonus Tip: How to improve the prompt

If you'd like the LLM to respond more conversationally, you could extend it slightly:

```text
You are a helpful chatbot. A user asked about the status of an order. Here is the order information:

Order ID: {order_id}
Order Status JSON: {order_json}

Respond with a short, friendly message that clearly answers their question based on the status.
```

Let me know if you want a version that includes delivery ETA, or rephrases based on status (e.g. delayed vs shipped vs processing).