from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError
from flasgger import Swagger
app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:localroot@localhost/python_flask_orders'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)


# Define the Order model
class Order(db.Model):
    __tablename__ = 'orders'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    order_no = db.Column(db.String(50), nullable=False)
    order_date = db.Column(db.DateTime, nullable=False)
    product_id = db.Column(db.String(50), nullable=False)
    qty_ordered = db.Column(db.Integer, nullable=False)
    unit_price = db.Column(db.Float, nullable=False)
    customer_id = db.Column(db.String(50), nullable=False)
    order_status = db.Column(db.String(20), nullable=False)


# REST API Endpoint to Get All Orders
@app.route('/orders', methods=['GET'])
def get_all_orders():
    try:
        orders = Order.query.all()
        results = [
            {
                "order_id": order.id,
                "order_no": order.order_no,
                "order_date": order.order_date,
                "product_id": order.product_id,
                "qty_ordered": order.qty_ordered,
                "unit_price": order.unit_price,
                "customer_id": order.customer_id,
                "order_status": order.order_status
            }
            for order in orders
        ]
        return jsonify(results), 200
    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500


# REST API Endpoint to Get an Order by ID
@app.route('/orders/<int:order_id>', methods=['GET'])
def get_order_by_id(order_id):
    try:
        order = Order.query.get(order_id)
        if order:
            result = {
                "order_id": order.id,
                "order_no": order.order_no,
                "order_date": order.order_date,
                "product_id": order.product_id,
                "qty_ordered": order.qty_ordered,
                "unit_price": order.unit_price,
                "customer_id": order.customer_id,
                "order_status": order.order_status
            }
            return jsonify(result), 200
        else:
            return jsonify({"message": "Order not found"}), 404
    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500


# REST API Endpoint to Add a New Order
@app.route('/orders', methods=['POST'])
def create_order():
    try:
        data = request.json
        new_order = Order(
            order_no=data.get('order_no'),
            order_date=data.get('order_date'),
            product_id=data.get('product_id'),
            qty_ordered=data.get('qty_ordered'),
            unit_price=data.get('unit_price'),
            customer_id=data.get('customer_id'),
            order_status=data.get('order_status')
        )
        db.session.add(new_order)
        db.session.commit()
        return jsonify({"message": "Order created successfully", "order_id": new_order.id}), 201
    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500


# REST API Endpoint to Update an Order
@app.route('/orders/<int:order_id>', methods=['PUT'])
def update_order(order_id):
    try:
        data = request.json
        order = Order.query.get(order_id)
        if order:
            order.order_no = data.get('order_no', order.order_no)
            order.order_date = data.get('order_date', order.order_date)
            order.product_id = data.get('product_id', order.product_id)
            order.qty_ordered = data.get('qty_ordered', order.qty_ordered)
            order.unit_price = data.get('unit_price', order.unit_price)
            order.customer_id = data.get('customer_id', order.customer_id)
            order.order_status = data.get('order_status', order.order_status)
            db.session.commit()
            return jsonify({"message": "Order updated successfully"}), 200
        else:
            return jsonify({"message": "Order not found"}), 404
    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500


# REST API Endpoint to Delete an Order
@app.route('/orders/<int:order_id>', methods=['DELETE'])
def delete_order(order_id):
    try:
        order = Order.query.get(order_id)
        if order:
            db.session.delete(order)
            db.session.commit()
            return jsonify({"message": "Order deleted successfully"}), 200
        else:
            return jsonify({"message": "Order not found"}), 404
    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5004)
    swagger = Swagger(app)
